# CustomCheckpoint.py 
# Author: Andrew Larkin
# Custom class for saving fine-tuned model weights every n batches

# import libraries
import time
from pathlib import Path
import lightning as L

# custom class for saving every n batches
class SaveCheckpointEveryNBatches(L.Callback):

    # create an instance of the custom checkpoint class
    # INPUTS:
    #    resultPath (str) - absolute folderpath where model weights should be stored on disk
    #    everyNBatches (int) - the save frequency, in units of n batches
    def __init__(self, resultPath, everyNBatches=100):
        self.resultPath = Path(resultPath)
        self.everyNBatches = everyNBatches
        self.batchCount = 0

    # what the custom save class does after the trainer has completed one batch
    # inerhited from the Lightning library
    # see https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html for inherited params
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batchCount += 1
        # if the batch number is divisible by 1000, save the model
        if self.batch_count % self.everyNBatches == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_dir = self.result_path / f"checkpoint_{timestamp}_batch{self.batchCount}"
            save_dir.mkdir(parents=True, exist_ok=True)
            pl_module.processor.save_pretrained(save_dir)
            pl_module.model.save_pretrained(save_dir)
            print(f"[Checkpoint] Saved at batch {self.batchCount} → {save_dir}")

    # what the custom save class does after the trainer has finished training
    # inerhited from the Lightning library
    # see https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html for inherited params
    def on_train_end(self, trainer, pl_module):
        save_dir = self.result_path / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        pl_module.processor.save_pretrained(save_dir)
        pl_module.model.save_pretrained(save_dir)
        print(f"[Final Checkpoint] Saved at training end → {save_dir}")

# end of CustomCheckpoint.py