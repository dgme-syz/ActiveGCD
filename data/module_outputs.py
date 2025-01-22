from dataclasses import dataclass

from datasets import Dataset
from torch.utils.data import DataLoader

@dataclass
class DatasetOutput:
    """ The output of the `get_datasets` function """
    
    train_labeled: Dataset | None = None
    train_unlabeled: Dataset | None = None
    test: Dataset | None = None
    val: Dataset | None = None
    
@dataclass
class DataLoaderOutput:
    """ The output of the `get_dataloaders` function """
    
    train_loader: DataLoader | None = None
    test_loader_unlabeled: DataLoader | None = None
    test_loader_labeled: DataLoader | None = None
    train_labeled_loader_ind_mapping: DataLoader | None = None
    val_loader: DataLoader | None = None
    
    
    