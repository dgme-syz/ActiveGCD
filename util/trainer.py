import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import transformers

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
    
class DistillLoss(nn.Module):
    def __init__(
        self,
        warmup_teacher_temp_epochs: int, 
        n_epochs: int,
        n_crops: int = 2, 
        warmup_teacher_temp: float = 0.07,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1, 
    ):
        """
        Args:
            warmup_teacher_temp_epochs (int): The number of epochs during which the teacher 
                model's temperature gradually increases from `warmup_teacher_temp` to `teacher_temp`.
            n_epochs (int): The total number of epochs for training the student model.
            n_crops (int, optional): The number of crops (sub-images) per image for data 
                augmentation. Defaults to 2.
            warmup_teacher_temp (float, optional): The initial temperature for the teacher model 
                during the warmup phase. Defaults to 0.07.
            teacher_temp (float, optional): The final temperature for the teacher model after 
                the warmup phase. Defaults to 0.04.
            student_temp (float, optional): The temperature used for the student model to control 
                the smoothness of the output logits. Defaults to 0.1.
        """

        super().__init__()
        self.student_temp = student_temp
        self.n_crops = n_crops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(
                warmup_teacher_temp, 
                teacher_temp,
                warmup_teacher_temp_epochs
            ), 
            np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    def forward(
        self, 
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        epoch: int,
    ):
        """ Compute the distillation loss between the student and teacher outputs.  """

        pass


class Trainer(transformers.Trainer):
    """ Use Trainer to train the model with the distillation loss. """
    
    def __init__(
        self, 
        **kwargs
    ):
        pass
    
    def log(self, logs):
        pass
    
    def prediction_step(
        self, 
        model, 
        inputs, 
        prediction_loss_only, 
        ignore_keys = None
    ):
        pass
    
    
    def compute_loss(
        self, model, batch, return_outputs=False
    ):
        images, labels, is_labeled = batch["image"], batch["label"], batch["is_labeled"]
        n_views, B, C, H, W = images.shape
        labels = labels.to(model.device)
        is_labeled = is_labeled.to(model.device).bool()
        images = images.to(model.device) # n_views x B x C x H x W
        
        student_proj = model[0](images)
        student_outs = model[1](student_proj)
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
        unsupervised_cluster_loss = cluster_criterion(
            student_outs, teacher_outs, epoch
        )
        average_probs = (student_outs / 0.1).softmax(dim=1).mean(dim=0)
        me_max_loss = -torch.sum(torch.log(average_probs ** (-average_probs))) + math.log(float(len(average_probs)))
        unsupervised_cluster_loss += memax_weight * me_max_loss
        
        # 3. unsupervised representation learning
        contrastive_logits, constrastive_labels = 


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
    
    cluster_criterion = DistillLoss(
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        n_epochs=epochs,
        n_crops=n_views,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
    )
    
       
                