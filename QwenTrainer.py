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
    def __init__(self, config, processor, model,trainJsonlFilepath,evalJsonlFilepath,imageFolderpath):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])  # save config for resume
        self.config = config
        self.processor = processor
        self.model = model

        # load training dataset
        self.trainDataset = JSONLDataset(
            jsonlFilepath=trainJsonlFilepath,
            imageFolderpath=imageFolderpath,
        )

        # load eval dataset
        self.evalDataset = JSONLDataset(
            jsonlFilepath=evalJsonlFilepath,
            imageFolderpath=imageFolderpath,
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
        print(batch[0])
        _, _, examples = zip(*batch)

        # extract LLM prompts
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example
            in examples
        ]

        # extract street view imagery
        imageInputs = [
            process_vision_info(example)[0]
            for example
            in examples
        ]

        # combine prompts and imagery to create model inputs
        modelInputs = self.processor(
            text=texts,
            images=imageInputs,
            return_tensors="pt",
            padding=True
        )
        
        labels = modelInputs["input_ids"].clone()

        # mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        imageTokens = [151652, 151653, 151655]
        # mask image token IDs in the labels
        for imageTokenID in imageTokens:
            labels[labels == imageTokenID] = -100

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
        cat,image,query,label = batch
        
        print(label)
        suffixes = label#[d for d in label]#[d["label"] for d in data]
        print(suffixes)
        _,_, data, examples = zip(*batch)
        # drop the assistant portion so the model must generate it
        examples = [e[:2] for e in examples]

        # extract LLM prompts
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example
            in examples
        ]

        # extract street view imagery
        image_inputs = [
            process_vision_info(example)[0]
            for example
            in examples
        ]

        # combine imagery and prompts to create model inputs
        model_inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        image_grid_thw = model_inputs["image_grid_thw"]
        return input_ids, attention_mask, pixel_values, image_grid_thw, suffixes

    # how to calculate loss after generating predictions for a single training batch
    # INPUTS:
    #    batch (array of tuples) - contains training records (imagery and structured data)
    # OUTPUTS:
    #    loss (float) - measure of the difference between model predictions and labels
    def training_step(self, batch):
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
    def validation_step(self, batch, batchIdx, datasetIdx=0):
        input_ids, attentionMask, pixelValues, imageGridThw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attentionMask,
            pixel_values=pixelValues,
            image_grid_thw=imageGridThw,
            max_new_tokens=128
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids
            in zip(input_ids, generated_ids)]
        generated_suffixes = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # number of records where the direction (left or right) was correctly predicted        
        correct_directions = 0
        total_distance_error = 0
        total = 0

        for generated_suffix, target_suffix in zip(generated_suffixes, suffixes):
            # If either is a string, convert it to a dict
            if isinstance(generated_suffix, str):
                try:
                    generated_suffix = self.extract_direction_distance_json(generated_suffix)
                except json.JSONDecodeError:
                    print("❌ Could not decode generated suffix:", generated_suffix)
                    print(generated_suffix)
                    continue

            if isinstance(target_suffix, str):
                try:
                    target_suffix = self.extract_direction_distance_json(generated_suffix)
                except json.JSONDecodeError:
                    #print("❌ Could not decode ground truth suffix:", target_suffix)
                    continue
            try:
                print("prediction: %s, label: %s" %(str(generated_suffix),str(target_suffix)))
                predDirection = generated_suffix.get("direction")
                labelDirection = target_suffix.get("direction")
                direction_correct = 0
                if(predDirection in ['left','left ']):
                    if(labelDirection in ['left','left ']):
                        direction_correct = 1
                elif (predDirection == 'right' and labelDirection=='right'):
                    direction_correct = 1
                #direction_correct = generated_suffix.get("direction") == target_suffix.get("direction")
                distance_error = abs(generated_suffix.get("distance", 0) - target_suffix.get("distance", 0))
                distance_error = distance_error*distance_error

                correct_directions += int(direction_correct)
                total_distance_error += distance_error
                total += 1
                #print("sucessfully updated score")
            except Exception as e:
                print(str(e))

        direction_accuracy = correct_directions / total if total > 0 else 0
        avg_distance_error = total_distance_error / total if total > 0 else 2500
        avg_distance_error = np.sqrt(avg_distance_error)
        # Log metrics
        self.log("val_direction_accuracy", direction_accuracy, prog_bar=True, logger=True)
        self.log("val_avg_distance_error", avg_distance_error, prog_bar=True, logger=True)
        combined_score = (1 - direction_accuracy) + (avg_distance_error / 50.0)  # Normalize distance
        self.log("val_combined_score", combined_score, prog_bar=False, logger=True)
        return combined_score
        
    # set the learning rate for the Adam optimizer
    # OUTPUTS:
    #    optimizer (AdamW optimizer)
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr"))
        return optimizer
    
    # create a training dataset loader
    def train_dataloader(self):
        return DataLoader(
            self.trainDataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.trainCollateFxn,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )
    
    # create a validation dataset loader
    def val_dataloader(self):
        return DataLoader(
            self.evalDataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.evalCollateFxn,
            num_workers=10,
            pin_memory=True
        )

# end of QwenTrainer.py