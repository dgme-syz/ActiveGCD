import os
import argparse
import math
from copy import deepcopy
from typing import Literal

import torch
from torch import nn
from tqdm import tqdm

from util import logger
from util import DINOHead
from data import GcdData

def get_models(
    mlp_output_dim: int, 
    device: str, 
    grad_from_block: int = 0,
) -> tuple[nn.Module, nn.Module]:
    """ Load the DINO model and the DINO EMA model."""
    
    backbone = torch.hub.load(
        repo_or_dir='facebookresearch/dino:main', model='dino_vitb16',
    )
    for n, p in backbone.named_parameters():
        if 'block' in n and int(n.split('.')[1]) >= grad_from_block:
            p.requires_grad = True
        else:
            p.requires_grad = False
    backbone_ema = deepcopy(backbone)
                
    model = nn.Sequential(
        backbone, 
        DINOHead(
            in_dim=768, 
            out_dim=mlp_output_dim, 
            nlayers=3,
        )
    )
    model_ema = nn.Sequential(
        backbone_ema, 
        DINOHead(
            in_dim=768, 
            out_dim=mlp_output_dim, 
            nlayers=3,
        )
    )
    return model.to(device), model_ema.to(device)

def work(
    dataset: str = 'CIFAR10', 
    device: str = 'cuda',
    grad_from_block: int = 11, 
    num_workers: int = 2, 
    per_device_train_batch_size: int = 128, 
    per_device_eval_batch_size: int = 256, 
    output_dir: str = '/saves', 
    ssb_ratio: float = 0.5,
    ssb_num_labels: int | None = None,
    use_ssb_splits: bool = False, 
    pkl_path: str | None = None,
    yaml_path: str | None = None,
):
    r"""a main function to start ActiveGCD
    
    Process the dataset and call the training function.
    
    Args:
        dataset (`str`, `optional`, defaults to `'CIFAR10'`): 
            The dataset to use for training and evaluation. Choose between 'CIFAR10' and 'CIFAR100'.
        device (`str`, `optional`, defaults to `'cuda'`): 
            The device to run the model on (e.g., 'cuda' or 'cpu'). 
        grad_from_block (`int`, `optional`, defaults to `11`): 
            The block from which to start gradient calculations during training. 
        num_workers (`int`, `optional`, defaults to `2`): 
            The number of workers to use for data loading. 
        per_device_train_batch_size (`int`, `optional`, defaults to `128`): 
            The batch size per device during training. 
        per_device_eval_batch_size (`int`, `optional`, defaults to `256`): 
            The batch size per device during evaluation. 
        output_dir (`str`, `optional`, defaults to `'/saves'`): 
            The directory where the model and results will be saved. 
        split_ratio (`float`, `optional`, defaults to `0.5`):
            The ratio of the dataset used for labeled and the rest for unlabeled data.
    """
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    data = GcdData(
        dataset=dataset,
        device=device,
        ssb_ratio=ssb_ratio,
        ssb_num_labels=ssb_num_labels,
        use_ssb_splits=use_ssb_splits,
        pkl_path=pkl_path,
        yaml_path=yaml_path,
    )
    num_labeled_cls, num_unlabeled_cls = len(data.old_classes), len(data.new_classes)
    mlp_output_dim = num_labeled_cls + num_unlabeled_cls
    logger.info(
        f"{dataset} loaded, include {num_labeled_cls} labeled classes" 
        f"and {num_unlabeled_cls} unlabeled classes."
    )
    
    model, model_ema = get_models(mlp_output_dim, device)
    logger.info("Model loaded.")
    
    

if __name__ == '__main__':
    pass

