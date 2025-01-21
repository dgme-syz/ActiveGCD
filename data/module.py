import pickle

import yaml
from datasets import load_dataset

from .mapping import get_info
from .mapping import is_fgvc_dataset


def extract_by_yaml(pkl_path: str, yaml_path: str) -> tuple[list[int], list[int]]:
    """ Extract the class splits by the yaml file """
    
    with open(yaml_path, 'r') as f:
        key_info = yaml.safe_load(f)
    with open(pkl_path, 'rb') as f:
        ssb_info = pickle.load(f)
        
    if (
        key_info.get("new", None) is None
        or key_info.get("old", None) is None
    ):
        raise ValueError(
            'The yaml file should contain the key "new" and "old" to specify the class splits.'
        )
        
    def recursive(k_info: list | dict, info: dict) -> list[int]:
        if isinstance(k_info, str):
            x = info.get(k_info, None)
            if isinstance(x, list):
                return x
            else:
                raise ValueError(
                    f'The key info is a string, but the info is not a dict[list[int]], '
                    f'got dict[{type(x)}], please check the yaml file.'
                )
        elif isinstance(k_info, list):
            x = []
            for k in k_info:
                y = info.get(k, [])
                if isinstance(y, list):
                    x += y
                else:
                    raise ValueError(
                        f'The key info is a list of strings, but the info is not a dict[list[int]], '
                        f'got dict[{type(y)}], please check the yaml file.'
                    )
            return x
        else:
            x = []
            for k, v in k_info.items():
                x += recursive(v, info[k])
            return x

    old = recursive(key_info["old"], ssb_info)
    new = recursive(key_info["new"], ssb_info)
    
    return old, new



class GcdData:
    def __init__(
        self, 
        dataset: str = 'CIFAR10',
        device: str = 'cuda',
        ssb_ratio: float = 0.5, 
        ssb_num_labels: int | None = None,
        use_ssb_splits: bool = False, 
        pkl_path: str | None = None,
        yaml_path: str | None = None,
    ): 
        self.name = dataset
        hub, keys, self.num_labels = get_info(dataset)
        self.image_key, self.label_key = keys
        
        self.data = load_dataset(hub, split='train')
        self.old_classes, self.new_classes = self.get_class_splits(
            name=dataset,
            ssb_ratio=ssb_ratio,
            ssb_num_labels=ssb_num_labels,
            use_ssb_splits=use_ssb_splits,
            pkl_path=pkl_path,
            yaml_path=yaml_path,
        )
    
    def get_class_splits(
        self, 
        name: str = None, 
        ssb_ratio: float | None = 0.5,
        ssb_num_labels: int | None = None,
        use_ssb_splits: bool = False,
        pkl_path: str | None = None,
        yaml_path: str | None = None,
    ) -> tuple[list[int], list[int]]:
        """ Get the class splits for the dataset

        Args:
            name (`str`, optional): The name of dataset. Defaults to None.
            ssb_ratio (`float | None`, optional): The ratio of old classes, the others used for 
                new classes. Defaults to 0.5.
            ssb_num_labels (`int | None`, optional): If ssb_ratio is not specified, the 
                parameter will be used to select the top ssb_num_labels. Defaults to None.
            use_ssb_splits (`bool`, optional): For the FGVC dataset, there are already 
                established cuts, and specifying the current parameter as True means using 
                these settings. Defaults to False.
            pickle_path (`str | None`, optional): The path to the pickle file that contains 
                the class splits, used when `use_ssb_splits` is `True`. Defaults to None.
            yaml_path (`str | None`, optional): The path to the yaml file that contains the
                class splits, used when `use_ssb_splits` is `True`. Defaults to None.
        Returns:
            `tuple[list[int], list[int]]`: The first list contains old classes, and the
                ohers contains new classes.
        """

        if not use_ssb_splits:
            if not ((ssb_ratio is None) ^ (ssb_num_labels is None)):
                raise ValueError('Either ssb_ratio or ssb_num_labels should be specified')
            if is_fgvc_dataset(name):
                Warning('The dataset is a FGVC dataset, but the use_ssb_splits is False')
            
            if ssb_ratio is not None:
                split_num = int(self.num_labels * ssb_ratio)
            else:
                split_num = ssb_num_labels
            
            old_classes = list(range(split_num))
            new_classes = list(range(split_num, self.num_labels))
        
        else:
            if not is_fgvc_dataset(name):
                raise ValueError('The dataset is not an FGVC dataset')
            if pkl_path is None or yaml_path is None:
                raise ValueError('The pickle file or yaml file path is not specified')
            
            # Now just use the pickle file
            old_classes, new_classes = extract_by_yaml(
                pkl_path=pkl_path, yaml_path=yaml_path
            )
        return old_classes, new_classes