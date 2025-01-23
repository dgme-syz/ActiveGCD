from collections import OrderedDict


MAPPING_NAME_TO_HUB = OrderedDict([
    ('CIFAR10', 'uoft-cs/cifar10'), 
])

MAPPING_NAME_TO_KEY = OrderedDict([
    ('CIFAR10', ('img', 'label')),  
])

MAPPING_NAME_TO_SPLIT = OrderedDict([
    ('CIFAR10', ('train', 'test')),  
])

MAPPING_NAME_TO_CLS = OrderedDict([
    ('CIFAR10', 10),   
])

FGVC_DATASET = ('CUB', 'SCARS', 'AIRCRAFT')


def is_fgvc_dataset(name: str) -> bool:
    """ check if the dataset is an FGVC dataset """
    
    return name in FGVC_DATASET

def get_splits(name: str) -> tuple[str, str]:
    """ get the split names for the dataset """
    
    return MAPPING_NAME_TO_SPLIT[name]

def get_keys(name: str) -> tuple[str, str]:
    """ get the key names for the dataset """
    
    return MAPPING_NAME_TO_KEY[name]

def get_info(name: str) -> tuple[str, tuple[str, str], int]:
    """ get the hub, key, and number of classes for the dataset """
    
    return MAPPING_NAME_TO_HUB[name], MAPPING_NAME_TO_KEY[name], MAPPING_NAME_TO_CLS[name]
    

def register_dataset(name: str, hub: str, key: tuple, cls: int):
    """register a dataset to the mapping """
    
    MAPPING_NAME_TO_HUB[name] = hub
    MAPPING_NAME_TO_KEY[name] = key
    MAPPING_NAME_TO_CLS[name] = cls