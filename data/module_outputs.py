from dataclasses import dataclass
from dataclasses import fields
from collections import namedtuple

from datasets import Dataset
from torch.utils.data import DataLoader

class Output:
    def __iter__(self):
        return iter([getattr(self, f.name) for f in fields(self)])

@dataclass
class DatasetOutput(Output):
    """ The output of the `get_datasets` function """
    
    train_labeled: Dataset | None = None
    train_unlabeled: Dataset | None = None
    test: Dataset | None = None
    val: Dataset | None = None
    
@dataclass
class DataLoaderOutput(Output):
    """ The output of the `get_dataloaders` function """
    
    train_loader: DataLoader | None = None
    test_loader_unlabeled: DataLoader | None = None
    test_loader_labeled: DataLoader | None = None
    train_labeled_loader_ind_mapping: DataLoader | None = None
    val_loader: DataLoader | None = None
    
    
    