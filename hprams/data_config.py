from dataclasses import dataclass, field

    
@dataclass
class DatasetConfig:
    
    dataset: str = field(
        default="cifar10", 
        metadata={
            "help": (
                "The name of the dataset to use for training. It should be one of the "
                "datasets supported by torchvision."
            )
        }
    )
    
    num_workers: int = field(
        default=2, 
        metadata={
            "help": (
                "The number of worker processes to use for data loading. Increasing this "
                "speeds up data loading but consumes more memory."
            )
        }
    )
    
    ssb_ratio: float = field(
        default=0.5, 
        metadata={
            "help": (
                "The ratio of the dataset to use for labeled data. The rest of the data "
                "is used for unlabeled data."
            )
        }
    )
    
    ssb_num_labels: int | None = field(
        default=None, 
        metadata={
            "help": (
                "The number of labeled classes to use for semi-supervised learning. The "
                "rest of the classes are considered unlabeled."
            )
        }
    )
    
    use_ssb_splits: bool = field(
        default=False, 
        metadata={
            "help": (
                "Whether to use the splits provided by Semi-Supervised Benchmark for "
                "training and evaluation."
            )
        }
    )
    
    pkl_path: str | None = field(
        default=None, 
        metadata={
            "help": (
                "The path to a pickle file containing the dataset splits. Used when "
                "use_ssb_splits is False."
            )
        }
    )
    
    pkl_info_path: str | None = field(
        default=None, 
        metadata={
            "help": (
                "The path to a YAML file containing .pkl info. Used when "
                "use_ssb_splits is False."
            )
        }
    )
    
    
    