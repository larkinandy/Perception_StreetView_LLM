# fineTuneQwenvl.py
# Author: Andrew Larkin
# Summary: Fine tune the qwenvl2.5 model for predicting perceptions from street view imagery.


########## IMPORT LIBRARIES ############

# import libraries
import torch
from pathlib import Path
from peft import get_peft_model, LoraConfig,PeftModel, get_peft_model, PeftConfig
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import lightning as L
#from nltk import edit_distance
#from lightning.pytorch.callbacks import Callback
#from lightning.pytorch.callbacks.early_stopping import EarlyStopping



# import custom libraries
from QwenTrainer import Qwen2_5_Trainer
from CustomCheckpoint import SaveCheckpointEveryNBatches

# set params
#torch.set_float32_matmul_precision('high')


######## LLM Prompts #########
PROMPT = '''You will be shown two images. Your job is to answer the following question by moving a virtual slider **left or right**.  
- Move the slider **right** if the **right image is better**, or **left** if the **left image is better**.  
- You move the slider farther (on a scale from 1 to 50) if you feel more strongly about your choice.  
- You **must** choose a direction and a distance. The images cannot be rated equally.

Respond using **only a valid JSON object** on a single line, exactly in the format: { "direction": direction, "distance": distance }

Your response **must start with {**

Now answer the following question:'''

SYSTEM_MESSAGE = """You are a human filling out a survey on Amazon Mechanical Turk. """

######## STORAGE LOCATIONS ##########

IMAGE_FOLDERPATH = "/mnt/c/users/larkinan/Desktop/LLM/combined_GSV"
TRAIN_JSONL_FILEPATH = "/mnt/c/users/larkinan/Desktop/LLM/trainMTurk.jsonl"
VAL_JSONL_FILEPATH="/mnt/c/users/larkinan/Desktop/LLM/evalMTurk.jsonl"
SAVE_DIR = "/mnt/c/users/larkinan/Desktop/LLMSaves"
RESULT_PATH = "/mnt/c/users/larkinan/Desktop/LLMSaves"

####### MODEL PARAMS ############

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fine tune just a small number of weights in the q_proj and v_proj layers

LORA_CONFIG = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# use 4bit precision to allow the model to fit in memory (RTX 6000)
USE_QLORA = True
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )

# define max and min image size
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# due to ram limitations of 1x RTX 6000, only 1 image can be processed at a time during training. 
# batch size can be 2 during implementation
BATCH_SIZE = 1
NUM_WORKERS = 0
resultPath = Path(RESULT_PATH)

config = {
    "max_epochs": 10,
    "batch_size": BATCH_SIZE,
    "lr": 1e-4,
    "val_check_interval": 1000,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "num_nodes": 1,
    "warmup_steps": 100,
    "result_path": "qwen2.5-32b-instruct-ft"
}

# recursively search for saved models :
checkpointDirs = list(resultPath.rglob("checkpoint_*"))

# train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, collate_fn=trainCollateFxn, num_workers=NUM_WORKERS, shuffle=True)
# valid_loader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, collate_fn=evalCollateFxn, num_workers=NUM_WORKERS)



# create an instance of the custom save checkpoint class
saveEveryNBatchesCallback = SaveCheckpointEveryNBatches(
    resultPath=SAVE_DIR,  
    everyNBatches=1000
)

# create a trainer 
trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    val_check_interval=1000,
    gradient_clip_val=config.get("gradient_clip_val"),
    limit_val_batches=125,
    num_sanity_val_steps=0,
    log_every_n_steps=1000,
    callbacks=[
        saveEveryNBatchesCallback,   
    ],
    default_root_dir=SAVE_DIR,
    enable_checkpointing=True,
)

# if a fine-tuned model already exists, load the fine-tuned model weights and resume fine-tuning. Otherwise, 
# load the default 32b qwen2.5VL model and start fine-tuning
if checkpointDirs:
    checkpointDirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_ckpt_dir = checkpointDirs[0]
    print(f"🔁 Loading from latest checkpoint folder: {latest_ckpt_dir}")

    # Load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(latest_ckpt_dir)
    processor.tokenizer.padding_side = "left"

    # Load PEFT config
    peft_config = PeftConfig.from_pretrained(latest_ckpt_dir)

    # Load base model from PEFT config base_model_name_or_path
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=bnb_config if USE_QLORA else None,
    )

    # Load LoRA adapter weights and attach to base model
    model = PeftModel.from_pretrained(base_model, latest_ckpt_dir)
    model.train()

    for name, param in model.named_parameters():
        # Only keep LoRA params trainable, freeze the rest
        if "lora" in name.lower():
            param.requires_grad = True
            print(f"LoRA trainable param: {name}")
        else:
            param.requires_grad = False

    modelModule = Qwen2_5_Trainer(config, processor, model,TRAIN_JSONL_FILEPATH,VAL_JSONL_FILEPATH,IMAGE_FOLDERPATH)

else:
    print("🚀 No checkpoint found, training from scratch")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config if USE_QLORA else None,
        dtype=torch.bfloat16
    )
    model = get_peft_model(model, LORA_CONFIG)
    
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left",
    modelModule = Qwen2_5_Trainer(config, processor, model,TRAIN_JSONL_FILEPATH,VAL_JSONL_FILEPATH,IMAGE_FOLDERPATH)

# optional: print trainable params. Good for debugging lora
#model.print_trainable_parameters()

# start fine-tuning the model
trainer.fit(modelModule)


# end of fineTuneQwenvl.py