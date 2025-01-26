from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from transformers import TrainingArguments

@dataclass
class TrainingArgs(TrainingArguments):
    """ Training Arguments """
    
    warmup_teacher_temp_epochs: int = field(
        default=30, 
        metadata={
            "help": (
                "The number of epochs to warm up the teacher temperature. Gradually increases "
                "the temperature of the teacher model during training."
            )
        }
    )
    
    n_views: int = field(
        default=2, 
        metadata={
            "help": (
                "The number of augmented views per input sample, typically used in contrastive "
                "learning to generate multiple views for the same image."
            )
        }
    )
    
    warmup_teacher_temp: float = field(
        default=0.07, 
        metadata={
            "help": (
                "The initial value for the teacher model's temperature. It controls the sharpness "
                "of the probability distribution in knowledge distillation."
            )
        }
    )
    
    teacher_temp: float = field(
        default=0.04, 
        metadata={
            "help": (
                "The final value for the teacher model's temperature. It controls the sharpness "
                "of the output probability distribution."
            )
        }
    )
    
    memax_weight: float = field(
        default=2., 
        metadata={
            "help": (
                "The weight applied to the memory bank or memory mechanism, often used in models "
                "that rely on memory-based methods."
            )
        }
    )
    
    sup_weight: float = field(
        default=0.35, 
        metadata={
            "help": (
                "The weight applied to the supervised contrastive loss. It balances the importance "
                "of supervised learning with unsupervised learning."
            )
        }
    )

    # "normal" or "balanced" or "mixed"
    eval_func: Literal["normal", "balanced", "mixed"] = field(
        default="normal", 
        metadata={
            "help": (
                "The evaluation function to use. 'normal' computes the accuracy as a whole, 'balanced' "
                "computes the accuracy for each class separately and averages, 'mixed' computes both."
            )
        }
    )
    
    warmup_teacher_temp_epochs: int = field(
        default=30, 
        metadata={
            "help": (
                "The number of epochs to warm up the teacher temperature. Gradually increases "
                "the temperature of the teacher model during training."
            )
        }
    )