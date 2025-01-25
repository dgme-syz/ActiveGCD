import pickle
from typing import Callable
from typing import Optional
from typing import Iterable
from PIL import Image
import warnings

import yaml
import torch
from datasets import load_dataset
from datasets import Dataset
from datasets import concatenate_datasets
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

from .mapping import get_info
from .mapping import is_fgvc_dataset
from .mapping import get_splits
from .module_outputs import DatasetOutput
from .module_outputs import DataLoaderOutput
from .module_outputs import DataOutput
from .augmix import get_transform


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

def get_class_splits(
    name: str, 
    num_labels: int,
    ssb_ratio: float | None = 0.5,
    ssb_num_labels: int | None = None,
    use_ssb_splits: bool = False,
    pkl_path: str | None = None,
    yaml_path: str | None = None,
) -> tuple[list[int], list[int]]:
    """ Get the class splits for the dataset

    Args:
        name (`str`, optional): The name of dataset. 
        num_labels (`int`, optional): The number of classes in the dataset.
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
            warnings.warn('The dataset is a FGVC dataset, but the use_ssb_splits is False')
        
        if ssb_ratio is not None:
            split_num = int(num_labels * ssb_ratio)
        else:
            split_num = ssb_num_labels
        
        old_classes = list(range(split_num))
        new_classes = list(range(split_num, num_labels))
    
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



def get_datasets(
    name: str, 
    dataset: Dataset, 
    train_classes: list[int],
    prop_train_labels: float = 0.2, 
    split_train_val: float = 0., 
) -> DatasetOutput:
    """ extract Dataset 
    
    We will take the prop_train_labels proportional part of the labeled data in 
    the training set as train_labeled and the rest as train_unlabeled, if you 
    specify to partition the validation set, it will be further partitioned in 
    the train_labeled dataset, which is generally not recommended, and will lead 
    to the supervised data to be further reduced.

    Args:
        name (`str`): The name of the dataset.
        dataset (`Dataset`): Embrace Face Dataset Objects.
        train_classes (`list[int]`): Used to specify which labels are existing
        prop_train_labels (`float`, optional): Used to specify what percentage 
            of the labeled data in the training set is to be used as train_la-
            beled. Defaults to 0.2.
        split_train_val (`float`): If specified as a positive real number(<1.), th-
            e specified percentage of data from the train_labeled data is sli-
            ced as the validation set.
    Returns:
        `tuple[Dataset, ...]`: _description_
    """
    if split_train_val < 0. or split_train_val >= 1.:
        raise ValueError('The split_train_val should be in the range [0, 1)')
    
    train_key, test_key = get_splits(name)
    train_data, test_data = dataset[train_key], dataset[test_key]
    
    # sample prop_train_labels of the labeled data in the training set as train_labeled
    labeled_train = train_data.filter(
        function=lambda x: x['label'] in train_classes)
    unlabeled_train = train_data.filter(
        function=lambda x: x['label'] not in train_classes)
    
    labeled_train = labeled_train.train_test_split(test_size=prop_train_labels)
    labeled_train, others = labeled_train['train'], labeled_train['test']
    
    # merge others and unlabeled_train
    # concatenate_datasets = copy, bot share the same memory
    unlabeled_train = concatenate_datasets([unlabeled_train, others])
    del others
    
    if split_train_val == 0.:
                
        return DatasetOutput(
            train_labeled=labeled_train,
            train_unlabeled=unlabeled_train,
            test=test_data,
        )
    else:
        labeled_train = labeled_train.train_test_split(test_size=split_train_val)
        labeled_train, val_data = labeled_train['train'], labeled_train['test']
                
        return DatasetOutput(
            train_labeled=labeled_train,
            train_unlabeled=unlabeled_train,
            test=test_data,
            val=val_data,
        )
        

class GcdData(object):
    def __init__(
        self, 
        dataset: str = 'CIFAR10',
        ssb_ratio: float = 0.5, 
        ssb_num_labels: int | None = None,
        use_ssb_splits: bool = False, 
        pkl_path: str | None = None,
        yaml_path: str | None = None,
        prop_train_labels: float = 0.2,
        split_train_val: float = 0.,
    ): 
        self.name = dataset
        hub, keys, self.num_labels = get_info(dataset)
        self.image_key, self.label_key = keys
        
        self.data = load_dataset(hub)
        self.old_classes, self.new_classes = get_class_splits(
            name=dataset,
            num_labels=self.num_labels, 
            ssb_ratio=ssb_ratio,
            ssb_num_labels=ssb_num_labels,
            use_ssb_splits=use_ssb_splits,
            pkl_path=pkl_path,
            yaml_path=yaml_path,
        )
        self.dataset_dict = get_datasets(
            name=dataset,
            dataset=self.data,
            train_classes=self.old_classes,
            prop_train_labels=prop_train_labels,
            split_train_val=split_train_val,
        )
    
    
    def get_dataloaders(
        self, 
        per_device_train_batch_size: int = 128,
        per_device_eval_batch_size: int = 256,
        drop_last: bool = True,
        pin_memory: bool = True,
        num_workers: int = 0,
        sampler: Sampler | Iterable | None = None,
        output_dataset: bool = False,
        train_tfm: Callable[[Image.Image], torch.Tensor] | None = None,
        test_tfm: Callable[[Image.Image], torch.Tensor] | None = None,
        **kwargs, 
    ) -> DataLoaderOutput:
        """ Get the dataloaders

            If you are suprised by the 'collate' results, please ref to 
            https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
            
        Args:
            per_device_train_batch_size (`int`, optional): how many samples p-
                er train batch in one device to load. Defaults to 128.
            per_device_eval_batch_size (`int`, optional): how many samples p-
                er eval(or test) batch in one device to load. Defaults to 256.
            drop_last (`bool`, optional): set to ``True`` to drop the last in-
                complete batch, if the dataset size is not divisible by the batch 
                size. If ``False`` and the size of dataset is not divisible by the 
                batch size, then the last batch will be smaller. Defaults to True.
            pin_memory (`bool`, optional): If ``True``, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them. Defaults to True.
            num_workers (`int`, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process. 
                Defaults to 0.
            sampler (`Sampler | Iterable | None`, optional): defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
                Defaults to None.

        Returns:
            `tuple[DataLoader, ...]`: The dataloaders for training, test, and validation.
            see `DataLoaderOutput` for more details.
        """
        if sampler is None:
            label_len, unlabel_len = (len(self.dataset_dict.train_labeled), 
                len(self.dataset_dict.train_unlabeled))
            train_len = label_len + unlabel_len
            sample_weights = [
                1 if i < label_len else label_len / unlabel_len for i in range(train_len)]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=train_len
            )
            
        if train_tfm is None and test_tfm is None:
            train_tfm, test_tfm = get_transform(**kwargs)
        elif (train_tfm is None) ^ (test_tfm is None):
            raise ValueError(
                f"We need to specify both train_tfm and test_tfm, "
                f"or neither of them (use the default transform)."
            )
            
        if kwargs.get("name", None) == None:
            kwargs["name"] = self.name # set by default
        elif kwargs["name"] != self.name:
            warnings.warn(
                f"The name in kwargs is different from the name in the dataset"
            )
            
        # add one new column to mark is labeled or not
        train = concatenate_datasets(
            [
                self.dataset_dict.train_labeled.map(
                    function=lambda x: {'is_labeled': [1]*len(x[next(iter(x))]), **x}, batched=True
                ), 
                self.dataset_dict.train_unlabeled.map(
                    function=lambda x: {'is_labeled': [0]*len(x[next(iter(x))]), **x}, batched=True
                )
            ]
        ) # copy, not share the same memory with the original data
        test, val = self.dataset_dict.test, self.dataset_dict.val
        
        test_unlabled_from_train = self.dataset_dict.train_unlabeled
        train_labeled_ind_mapping = self.dataset_dict.train_labeled
        
        # set the transform
        train.set_transform(train_tfm)
        test_unlabled_from_train.set_transform(test_tfm)
        test.set_transform(test_tfm)
        train_labeled_ind_mapping.set_transform(test_tfm)
        if val is not None:
            val.set_transform(test_tfm)
        
        if output_dataset:
            return DataOutput(
                train=train,
                test_unlabled_from_train=test_unlabled_from_train,
                test=test,
                train_labeled_ind_mapping=train_labeled_ind_mapping,
                val=val,
            )
        
        train_loader = DataLoader(
            train,
            batch_size=per_device_train_batch_size,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=False, 
        )
        
        test_loader_unlabeled_from_train = DataLoader(
            test_unlabled_from_train,
            batch_size=per_device_eval_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
        )
        
        test_loader = DataLoader(
            test,
            batch_size=per_device_eval_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
        )
        
        train_labeled_loader_ind_mapping = DataLoader(
            train_labeled_ind_mapping,
            batch_size=per_device_eval_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
        )
        
        if val is not None:
            val_loader = DataLoader(
                val,
                batch_size=per_device_eval_batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=False,
            )
        else:
            val_loader = None
        
        return DataLoaderOutput(
            train_loader=train_loader,
            test_loader_unlabeled_from_train=test_loader_unlabeled_from_train,
            test_loader=test_loader,
            train_labeled_loader_ind_mapping=train_labeled_loader_ind_mapping,
            val_loader=val_loader,
        )