import json
from torch.utils.data import Dataset

# Custom class for loading and accessing street view image comparisons and structured data
class JSONLDataset(Dataset):

    # create an instance of the JSONLDataset class
    # INPUTS:
    #    jsonlFilepath (str) - absolute filepath where jsonl structured data is stored on disk
    #    imageFolderpath (str) - absolute folderpath where street view image comparisons are stored on disk
    def __init__(self, jsonlFilepath, imageFolderpath):
        self.jsonlFilepath = jsonlFilepath
        self.imageFolderpath = imageFolderpath
        self.entries = self.loadEntries()

    # 
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        (Implementation details for __getitem__ would depend on the specific data structure)
        """
        return self.entries[idx]

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