import math
from typing import Optional, Any
import warnings

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
from .train_module import (
    DistillLoss,
    info_nce_logits,
    SupConLoss
)
from .eval_module import split_cluster_acc_v2
from .log import logger

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
    



class Trainer(transformers.Trainer):
    """ Use Trainer to train the model with the distillation loss. """
    
    def __init__(
        self, 
        train_dataloader: DataLoader, 
        eval_dataloader: dict[str, DataLoader],
        labeled_classes: list[int],
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        *args, 
        **kwargs
    ):
        super().__init__(
            optimizers=optimizers, *args, **kwargs
        )
        # use args from parent class
        self.args: TrainingArgs = self.args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.cluster_criterion = DistillLoss(
            warmup_teacher_temp_epochs=self.args.warmup_teacher_temp_epochs,
            n_epochs=self.args.num_train_epochs,
            n_crops=self.args.n_views,
            warmup_teacher_temp=self.args.warmup_teacher_temp,
            teacher_temp=self.args.teacher_temp,
        )
        self.labeled_classes = labeled_classes
        self.label_names = ["label"]
        self.compute_metrics = self.compute_metrics_fn
    
    def get_train_dataloader(self):
        """ use custom dataloader """
        
        return self.accelerator.prepare(self.train_dataloader)
    
    def get_eval_dataloader(self, eval: str):
        """ use custom dataloader """
        if not isinstance(eval, str):
            raise NotImplementedError("Unexpected!")
        
        return self.accelerator.prepare(self.eval_dataloader[eval])
    
    def compute_metrics_fn(self, eval_preds: EvalPrediction) -> dict:
        """ custom update metrics for test """
        
        y_pred, y_true = eval_preds # not return inputs
        if y_pred.shape != y_true.shape:
            y_pred = np.argmax(y_pred, axis=1) # convert to class index
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")
        
        mask = np.isin(y_true, self.labeled_classes).astype(bool)
        
        res = {}
        if self.args.eval_func in ["normal", "mixed"]:
            prefix = "normal"
            total_acc, old_acc, new_acc = split_cluster_acc_v2(
                y_true=y_true, y_pred=y_pred, mask=mask, use_balanced=False
            )
            res.update({
                f"{prefix}_total_acc": total_acc,
                f"{prefix}_old_acc": old_acc,
                f"{prefix}_new_acc": new_acc,
            })
            
        if self.args.eval_func in ["balanced", "mixed"]:
            prefix = "balanced"
            total_acc, old_acc, new_acc = split_cluster_acc_v2(
                y_true=y_true, y_pred=y_pred, mask=mask, use_balanced=True
            )
            res.update({
                f"{prefix}_total_acc": total_acc,
                f"{prefix}_old_acc": old_acc,
                f"{prefix}_new_acc": new_acc,
            })
        
        return res  
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool = False,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Custom Version does not return loss.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        if prediction_loss_only:
            prediction_loss_only = False
        
        images, labels = inputs["image"], inputs["label"]
        with torch.no_grad():
            _, logits = model(images)
        
        logits = logits.detach()
        return (None, logits, labels)
        
    def compute_loss(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], return_outputs=False, **kwargs
    ):
        (
            images, 
            labels, 
            is_labeled
        ) = inputs["image"], inputs["label"], inputs["is_labeled"]
        
        n_views = len(images)
        device = next(model[0].parameters()).device
        labels = labels.to(device)
        is_labeled = is_labeled.to(device).bool()
        # reshape to (n_views * B, C, H, W)
        images = torch.cat(images, dim=0).to(device)
        
        if len(images) == 0:
            raise ValueError("Unexpected!")
        
        student_proj, student_outs = model(images) # stdent_proj: (n_views * B, hidden_dim)
        teacher_outs = student_outs.detach()
        
        # 1. supervised clustering
        supervised_logits = torch.cat(
            [s[is_labeled] / 0.1 for s in student_outs.chunk(n_views)], dim=0
        )
        supervised_labels = torch.cat(
            [labels[is_labeled] for _ in range(n_views)], dim=0
        )
        supervised_cluster_loss = nn.CrossEntropyLoss()(
            input=supervised_logits, target=supervised_labels
        ) if any(is_labeled) else 0.
        
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
            [s[is_labeled].unsqueeze(1) for s in student_proj.chunk(n_views)], dim=1
        ) # B x n_views x embed_dim
        
        student_proj = F.normalize(student_proj, dim=-1)
        sup_con_labels = labels[is_labeled]
        sup_con_loss = SupConLoss()(
            features=student_proj, labels=sup_con_labels
        ) if any(is_labeled) else 0.
        
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
    train: DataLoader,
    test: DataLoader,
    test_unlabeled_from_train: DataLoader,
    old_classes: list[int], 
    learning_rate: float, 
    momentum: float,
    weight_decay: float,
    epochs: int,
    warmup_teacher_temp_epochs: int, 
    n_views: int,
    warmup_teacher_temp: float, 
    teacher_temp: float,
    memax_weight: float, 
    logging_steps: int, 
    output_dir: str = "./saves", 
    logging_dir: str = "./logs", 
    **kwargs, 
):
    """
    Train the model for the specified number of epochs, evaluate the model, 
    and log the results. The function supports incremental learning, 
    semi-supervised learning, and regularization techniques such as MEMAX.

    Args:
        model (`nn.Sequential`): The neural network model, typically a `nn.Sequential` object.
        train (`DataLoader`): `DataLoader` for the training set, provides batches of training data.
        test (`DataLoader`): `DataLoader` for the test set, used for evaluating model performance.
        test_unlabeled_from_train (`DataLoader`): `DataLoader` for unlabeled data from the training set.
        old_classes (list[`int`]): List of old classes, used in incremental learning.
        learning_rate (`float`): Learning rate for the optimizer.
        momentum (`float`): Momentum term for the optimizer.
        weight_decay (`float`): Weight decay parameter for L2 regularization.
        epochs (`int`): Total number of training epochs.
        warmup_teacher_temp_epochs (`int`): Number of epochs for warming up the teacher model's temperature.
        n_views (`int`): Number of augmented views of data used in self-supervised learning.
        warmup_teacher_temp (`float`): Initial teacher model temperature value for warmup.
        teacher_temp (`float`): Final teacher model temperature for controlling softmax smoothness.
        memax_weight (`float`): Weight for MEMAX regularization.
        logging_steps (`int`): Number of steps between each logging of training progress.
        output_dir (`str`, optional): Directory for saving model checkpoints (default is `./saves`).
        logging_dir (`str`, optional): Directory for saving logs (default is `./logs`).
        **kwargs: Additional keyword arguments for customization (e.g., device settings, optimizer configurations).

    Returns:
        None
    """
    
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
        weight_decay=weight_decay,
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=learning_rate * 1e-3,
    )
    
    training_args = TrainingArgs(
        output_dir=output_dir, 
        eval_strategy="epoch", 
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        n_views=n_views,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        memax_weight=memax_weight,
        save_strategy="epoch",
        logging_steps=logging_steps, # for train stage
        logging_dir=logging_dir,
        num_train_epochs=epochs,
        bf16=True, 
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train,
        eval_dataloader={
            "test": test, 
            "unlabled from train": test_unlabeled_from_train,
        }, 
        eval_dataset={"test": None, "unlabled from train": None}, 
        labeled_classes=old_classes, 
        optimizers=(optimizer, exp_lr_scheduler), 
        args=training_args,
    )
    trainer.train()                