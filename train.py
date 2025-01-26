import argparse

import yaml
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from hprams import TrainConfig
from hprams import DatasetConfig
from hprams import split_params
from util import logger
from util import DINOHead
from util import train
from data import GcdData

def get_model(
    mlp_output_dim: int, 
    device: str, 
    grad_from_block: int = 0,
) -> nn.Module:
    """ Load the DINO model"""
    
    backbone = torch.hub.load(
        repo_or_dir='facebookresearch/dino:main', model='dino_vitb16',
    )
    for n, p in backbone.named_parameters():
        if 'block' in n and int(n.split('.')[1]) >= grad_from_block:
            p.requires_grad = True
        else:
            p.requires_grad = False
    precision_type = next(backbone.parameters()).dtype
    logger.info(f"Model use {precision_type}")
    
    model = nn.Sequential(
        backbone, 
        DINOHead(
            in_dim=768, 
            out_dim=mlp_output_dim, 
            nlayers=3,
        )
    )
    return model.to(device)


def work(train_args: TrainConfig, data_args: DatasetConfig):
    """ Process the dataset and call the training function. """
    
    print(train_args)
    print(data_args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    data = GcdData(
        dataset=data_args.dataset,
        ssb_ratio=data_args.ssb_ratio,
        ssb_num_labels=data_args.ssb_num_labels,
        use_ssb_splits=data_args.use_ssb_splits,
        pkl_path=data_args.pkl_path,
        yaml_path=data_args.pkl_info_path,
    )
    num_labeled_cls, num_unlabeled_cls = len(data.old_classes), len(data.new_classes)
    mlp_output_dim = num_labeled_cls + num_unlabeled_cls
    logger.info(
        f"{data_args.dataset} loaded, include {num_labeled_cls} labeled classes " 
        f"and {num_unlabeled_cls} unlabeled classes."
    )
    
    model = get_model(mlp_output_dim, device, train_args.grad_from_block)
    logger.info("Model loaded.")
    
    (
        train_loader, 
        test_loader_unlabeled_from_train,
        test_loader,
        _,
        _,
    ) = data.get_dataloaders(
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        # used when output_dataset is False
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=data_args.num_workers,
        bf16=True,
    )
    
    train(
        model=model,
        train=train_loader,
        test_unlabeled_from_train=test_loader_unlabeled_from_train,
        test=test_loader,
        old_classes=data.old_classes,
        **train_args.get_args(),
    )
    logger.info("Training completed.")
    

if __name__ == '__main__':
    # too many params, we get them by yaml
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str,
        help="The path to the configuration file, including the whole params.",
    )
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    work(*split_params(config))
     
