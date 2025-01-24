from dataclasses import dataclass
from dataclasses import field

from transformers import TrainingArguments

@dataclass
class TrainingArgs(TrainingArguments):
    """ Training Arguments """
    
    do_eval: bool = field(
        default=True, 
        metadata={
            "help": (
                "Whether to evaluate the model during training. Set to True for evaluation "
                "after each epoch."
            )
        }
    )
    
    learning_rate: float = field(
        default=0.1, 
        metadata={
            "help": (
                "The rate at which the model learns. A higher value means the model updates "
                "its weights more drastically."
            )
        }
    )
    
    momentum: float = field(
        default=0.9, 
        metadata={
            "help": (
                "The momentum factor for the optimizer. It helps accelerate gradient descent "
                "in the right direction, leading to faster converging."
            )
        }
    )
    
    weight_decay: float = field(
        default=1e-4, 
        metadata={
            "help": (
                "The weight decay (L2 penalty) for regularization. It helps prevent overfitting "
                "by penalizing large weights."
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
