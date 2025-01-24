import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import transformers
import torch.nn.functional as F
from transformers import EvalPrediction

from .training_args import TrainingArgs
from train_module import (
    DistillLoss,
    info_nce_logits,
    SupConLoss
)

def get_params_groups(model: nn.Module) -> list[dict]:
    """ Do not regularize biases nor Norm parameters """
    
    reg, not_reg = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if n.endswith(".bias") or len(p.shape) == 1:
            not_reg.append(p)
        else:
            reg.append(p)
    
    return [
        {"params": reg},
        {"params": not_reg, "weight_decay": 0.0},
    ]
    
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """ custom update metrics for test """
    
    pass


class Trainer(transformers.Trainer):
    """ Use Trainer to train the model with the distillation loss. """
    
    def __init__(
        self, 
        eval_dataset: tuple[Dataset, Dataset],
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # use args from parent class
        self.args: TrainingArgs  = self.args
        
        self.cluster_criterion = DistillLoss(
            warmup_teacher_temp_epochs=self.args.warmup_teacher_temp_epochs,
            n_epochs=self.args.num_train_epochs,
            n_crops=self.args.n_views,
            warmup_teacher_temp=self.args.warmup_teacher_temp,
            teacher_temp=self.args.teacher_temp,
        )
        self.eval_dataset = eval_dataset
    
    def prediction_step(
        self, 
        model, 
        inputs, 
        prediction_loss_only, 
        ignore_keys = None
    ):
        self.evaluate(
            eval_dataset=self.eval_dataset[0], 
            metric_key_prefix="test"
        )
        self.evaluate(
            eval_dataset=self.eval_dataset[1], 
            metric_key_prefix="train unlabeled"
        )
    
    def compute_loss(
        self, model, batch, return_outputs=False
    ):
        (
            images, 
            labels, 
            is_labeled
        ) = batch["image"], batch["label"], batch["is_labeled"]
        
        n_views, B, C, H, W = images.shape
        
        labels = labels.to(model.device)
        is_labeled = is_labeled.to(model.device).bool()
        # reshape to (n_views * B, C, H, W)
        images = torch.cat(images, dim=0).to(model.device)
        
        student_proj, student_outs = model(images) # stdent_proj: (n_views * B, hidden_dim)
        teacher_outs = student_outs.detach()
        
        # 1. supervised clustering
        supervised_logits = torch.cat(
            [student_outs[i][is_labeled] / 0.1 for i in range(n_views)], dim=0
        )
        supervised_labels = torch.cat(
            [labels[is_labeled] for _ in range(n_views)], dim=0
        )
        supervised_cluster_loss = nn.CrossEntropyLoss()(
            input=supervised_logits, target=supervised_labels
        )
        
        # 2. unsupervised clustering
        unsupervised_cluster_loss = self.cluster_criterion(
            student_outs, teacher_outs, int(self.state.epoch)
        )
        average_probs = (student_outs / 0.1).softmax(dim=1).mean(dim=0)
        log_probs = torch.log(average_probs ** (-average_probs))
        me_max_loss = -torch.sum(log_probs) + math.log(float(len(average_probs)))
        unsupervised_cluster_loss += self.args.memax_weight * me_max_loss
        
        # 3. unsupervised representation learning
        unsup_con_logits, unsup_con_labels = info_nce_logits(
            features=student_proj, n_views=n_views,
        )
        unsup_con_loss = nn.CrossEntropyLoss()(
            input=unsup_con_logits, target=unsup_con_labels
        )
        
        # 4. supervised representation learning
        student_proj = torch.cat(
            [student_proj[i][is_labeled] for i in range(n_views)], dim=1
        ) # (B, hidden_dim * n_views)
        student_proj = F.normalize(student_proj, dim=-1)
        sup_con_labels = labels[is_labeled]
        sup_con_loss = SupConLoss()(
            features=student_proj, labels=sup_con_labels
        )
        
        # 5. total loss
        loss = (
            (1 - self.args.sup_weight) * unsupervised_cluster_loss
            + self.args.sup_weight * supervised_cluster_loss
            + self.args.sup_weight * sup_con_loss +
            (1 - self.args.sup_weight) * unsup_con_loss 
        )
        
        return (loss, student_outs) if return_outputs else loss

                


def train(
    model: nn.Sequential,
    train_loader: DataLoader,
    test_loader_labeled: DataLoader,
    test_loader_unlabeled: DataLoader,
    do_eval: bool = True,
    learning_rate: float = 0.1, 
    momentum: float = 0.9,
    wight_decay: float = 1e-4,
    epochs: int = 200,
    warmup_teacher_temp_epochs: int = 30, 
    n_views: int = 2,
    warmup_teacher_temp: float = 0.07, 
    teacher_temp: float = 0.04,
    memax_weight: float = 2., 
):
    # model = (feature_extractor, projection_head) expected
    if not (isinstance(model, nn.Sequential) and len(model) == 2):
        raise ValueError(
            "The model must be a nn.Sequential object with two modules: "
            "a feature extractor and a projection head."
        )
    
    params_groups = get_params_groups(model)
    optimizer = torch.optim.SGD(
        params=params_groups,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=wight_decay,
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=learning_rate * 1e-3,
    )
    
    
    
       
                