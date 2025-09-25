# QwenTrainer.py
# Author: Andrew Larkin
# Summary: Custom class for training a Qwen 2.5VL model

# import libraries
import re
import numpy as np
import json
import lightning as L
from torch.optim import AdamW
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
from JSONLDataset import JSONLDataset
from transformers import Qwen2_5_VLProcessor

# Custom class for training a Qwen 2.5 VL model
class Qwen2_5_Trainer(L.LightningModule):

    # create an instance of the Qwen2_5_Trainer class
    # INPUTS:
    #    config (dict): configuration params
    #    processor (Qwen2_5_VLProcessor) - initialized Qwen2_5_VL processor
    #    model (Qwen2_5_VL model) - initialized Qwen2_5_VL model
    #    trainJsonFilepath (str) - absolute filepath where structured data for training records is located
    #    evalJsonFilepath (str) - absolute filepath where structured data for eval records is located
    #    imageFolderpath (str) - aboluste folderpath where street view image comparisons are stored
    def __init__(self, config, processor, model,filepaths,systemMessage,prompt):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])  # save config for resume
        self.config = config
        self.processor = processor
        self.model = model

        self.trainDataset = JSONLDataset(
            jsonlFilepath=filepaths.TRAIN_JSONL_FILEPATH,
            imageFolderpath=filepaths.IMAGE_FOLDERPATH,
            systemMessage=systemMessage,
            prompt = prompt
        )
        self.validDataset = JSONLDataset(
            jsonlFilepath=filepaths.VAL_JSONL_FILEPATH,
            imageFolderpath=filepaths.IMAGE_FOLDERPATH,
            systemMessage=systemMessage,
            prompt = prompt
        )

    # custom function for fine-tuning Qwen 2.5VL model
    # INPUTS:
    #    batch (array of tuples) - contains training records (imagery and structured data)
    # OUTPUTS:
    #    inputIds (int array) - unique records ids
    #    attentionMask (int matrix) - which prompt tokens to pay attention to
    #    pixelValues (int matrix) - image data
    #    imgeGridThw (int matrix) - image dimensions
    #    labels (int matrix) - training record labels, in token format
    def trainCollateFxn(self,batch):
        _, _, examples = zip(*batch)
        
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example
            in examples
        ]
        
        imageInputs = [
            process_vision_info(example)[0]
            for example
            in examples
        ]
        
        modelInputs = self.processor(
            text=texts,
            images=imageInputs,
            return_tensors="pt",
            padding=True
        )
        
        labels = modelInputs["input_ids"].clone()
        # mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        if isinstance(self.processor, Qwen2_5_VLProcessor):
            imageTokens = [151652, 151653, 151655]
        else:
            imageTokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
        # mask image token IDs in the labels
        for image_token_id in imageTokens:
            labels[labels == image_token_id] = -100
        
        inputIds = modelInputs["input_ids"]
        attentionMask = modelInputs["attention_mask"]
        pixelValues = modelInputs["pixel_values"]
        imageGridThw = modelInputs["image_grid_thw"]
        return inputIds, attentionMask, pixelValues, imageGridThw, labels

    # custom function for evaluating the fine-tuned Qwen 2.5 VL model
    # INPUTS:
    #    batch (array of tuples) - contains training records (imagery and structured data)
    # OUTPUTS:
    #    inputIds (int array) - unique records ids
    #    attentionMask (int matrix) - which label tokens to pay attention to during training
    #    pixelValues (int matrix) - image data
    #    imgeGridThw (int matrix) - image dimensions
    #    labels (int matrix) - eval record labels, in token format
    def evalCollateFxn(self,batch):
        _, data, examples = zip(*batch)
        suffixes = [d["label"] for d in data]
        # drop the assistant portion so the model must generate it
        examples = [e[:2] for e in examples]

        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example
            in examples
        ]
        
        imageInputs = [
            process_vision_info(example)[0]
            for example
            in examples
        ]
        
        modelInputs = self.processor(
            text=texts,
            images=imageInputs,
            return_tensors="pt",
            padding=True
        )
        inputIds = modelInputs["input_ids"]
        attentionMask = modelInputs["attention_mask"]
        pixelValues = modelInputs["pixel_values"]
        imageGridThw = modelInputs["image_grid_thw"]
        return inputIds, attentionMask, pixelValues, imageGridThw, suffixes

    # how to calculate loss after generating predictions for a single training batch
    # INPUTS:
    #    batch (array of tuples) - contains training records (imagery and structured data)
    # OUTPUTS:
    #    loss (float) - measure of the difference between model predictions and labels
    def training_step(self, batch, batch_idx):
        inputIds, attentionMask, pixelValues, imageGridThw, labels = batch
        outputs = self.model(
            input_ids=inputIds,
            attention_mask=attentionMask,
            pixel_values=pixelValues,
            image_grid_thw=imageGridThw,
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    # given LLM output as free form text, remove pad tokens and convert releveant text to json format
    # INPUTS:
    #    text (str) - free form text with varying number of characters
    # OUTPUTS:
    #    model prediction in {'direction':direction,'distance':distance} format if found in text, None otherwise
    def extractDirectionDistanceJson(self,text: str):
        """
        Extracts and validates the LAST direction and distance key-value pairs from noisy LLM output.
        Handles cases with or without surrounding braces.
        Returns None if malformed or invalid.
        """
        pattern = re.compile(
            r"""            # verbose regex mode for clarity
            \{?             # optional opening brace
            [^{}]*?         # non-greedy anything but braces
            ['"]?direction['"]?\s*[:=]\s*['"]?(left|right)['"]?   # direction key-value pair
            [^{}]*?         # non-greedy anything but braces
            ['"]?distance['"]?\s*[:=]\s*([0-9]+(?:\.[0-9]*)?)      # distance key-value pair
            \}?             # optional closing brace
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        matches = list(pattern.finditer(text))
        for match in reversed(matches):
            try:
                direction = match.group(1).lower().strip()
                distance = float(match.group(2).strip())

                # Validate values
                if direction not in {"left", "right"}:
                    continue
                if not (1 <= distance <= 50):  # adjust range if needed
                    continue

                return {"direction": direction, "distance": distance}
            except Exception as e:
                print("🛑 Extracted match but failed to parse:", e)
                continue

        return None

    # calculate model performance using validation data
    # INPUTS:
    #    batch (array of tuples) - contains training records (imagery and structured data)
    #    batchIdx (int array) - record ids
    #    datasetIDx (int) - dataset number
    # OUTPUTS:
    #    combinedScore (float) - combined error of the distance and direction predictions
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        inputIds, attentionMask, pixelValues, imageGridThw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=inputIds,
            attention_mask=attentionMask,
            pixel_values=pixelValues,
            image_grid_thw=imageGridThw,
            max_new_tokens=128
        )

        generatedIdsTrimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids
            in zip(inputIds, generated_ids)]
        
        generatedSuffixes = self.processor.batch_decode(
            generatedIdsTrimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        scores = []
        correctDirections, totalDistanceError, total = 0,0,0

        for generatedSuffix, targetSuffix in zip(generatedSuffixes, suffixes):
            # If either is a string, convert it to a dict
            if isinstance(generatedSuffix, str):
                try:
                    generatedSuffix = self.extractDirectionDistanceJson(generatedSuffix)
                except json.JSONDecodeError:
                    print("❌ Could not decode generated suffix:", generatedSuffix)
                    print(generatedSuffix)
                    continue

            if isinstance(targetSuffix, str):
                try:
                    targetSuffix = self.extractDirectionDistanceJson(generatedSuffix)
                except json.JSONDecodeError:
                    #print("❌ Could not decode ground truth suffix:", target_suffix)
                    continue
            try:
                print("prediction: %s, label: %s" %(str(generatedSuffix),str(targetSuffix)))
                predDirection = generatedSuffix.get("direction")
                labelDirection = targetSuffix.get("direction")
                directionCorrect = 0
                if(predDirection in ['left','left ']):
                    if(labelDirection in ['left','left ']):
                        directionCorrect = 1
                elif (predDirection == 'right' and labelDirection=='right'):
                    directionCorrect = 1

                distanceError = abs(generatedSuffix.get("distance", 0) - targetSuffix.get("distance", 0))
                distanceError = distanceError*distanceError

                correctDirections += int(directionCorrect)
                totalDistanceError += distanceError
                total += 1
                #print("sucessfully updated score")
            except Exception as e:
                print(str(e))

        directionAccuracy = correctDirections / total if total > 0 else 0
        avgDistanceError = totalDistanceError / total if total > 0 else 2500
        avgDistanceError = np.sqrt(avgDistanceError)
        # Log metrics
        self.log("val_direction_accuracy", directionAccuracy, prog_bar=True, logger=True)
        self.log("val_avg_distance_error", avgDistanceError, prog_bar=True, logger=True)
        combinedScore = (1 - directionAccuracy) + (avgDistanceError / 50.0)  # Normalize distance
        self.log("val_combined_score", combinedScore, prog_bar=False, logger=True)
        return combinedScore
        
    # set the learning rate for the Adam optimizer
    # OUTPUTS:
    #    optimizer (AdamW optimizer)
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr"))
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(
            self.trainDataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.trainCollateFxn,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validDataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.evalCollateFxn,
            num_workers=10,
            pin_memory=True
        )

# end of QwenTrainer.py