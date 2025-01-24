import torch
from torch import nn
import numpy as np


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
