import json
import os
from torch.utils.data import Dataset
from PIL import Image


# Custom class for loading and accessing street view image comparisons and structured data
class JSONLDataset(Dataset):

    # create an instance of the JSONLDataset class
    # INPUTS:
    #    jsonlFilepath (str) - absolute filepath where jsonl structured data is stored on disk
    #    imageFolderpath (str) - absolute folderpath where street view image comparisons are stored on disk
    def __init__(self, jsonlFilepath, imageFolderpath,systemMessage,prompt):
        self.jsonlFilepath = jsonlFilepath
        self.imageFolderpath = imageFolderpath
        self.entries = self.loadEntries()
        self.systemMessage = systemMessage
        self.prompt = prompt


    def formatData(self,sample):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.systemMessage}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.imageFolderpath + "/" + sample["image"],
                    },
                    {
                        "type": "text",
                        "text": self.prompt + sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}],
            },
        ]

    # 
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.imageFolderpath, entry['image'])
        image = Image.open(image_path)
        return image, entry, self.formatData(entry)

    # load structured data into memory
    # OUTPUTS:
    #    array of records in json format
    def loadEntries(self):
        entries = []
        with open(self.jsonlFilepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries