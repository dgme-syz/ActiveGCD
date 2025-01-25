from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    """ A configuration class to manage hyperparameters and settings for training. """
    
    grad_from_block: int = field(
        default=11,
        metadata={
            "help": (
                "The block from which to start calculating gradients during training. "
                "Useful for gradient checkpointing."
            )
        }
    )
    
    learning_rate: float = field(
        default=1e-1, 
        metadata={
            "help": (
                "The learning rate for the optimizer. It determines how large a "
                "step the optimizer takes during training."
            )
        }
    )
    
    momentum: float = field(
        default=0.9, 
        metadata={
            "help": (
                "Momentum factor for optimizers like SGD. It helps accelerate gradient "
                "vectors in the right direction, preventing oscillations."
            )
        }
    )
    
    weight_decay: float = field(
        default=1e-4, 
        metadata={
            "help": (
                "The weight decay (L2 penalty) to apply during optimization. It helps "
                "prevent overfitting by adding a penalty to large weights."
            )
        }
    )
    
    epochs: int = field(
        default=200, 
        metadata={
            "help": (
                "The number of epochs for training. Each epoch consists of one complete "
                "pass over the entire dataset."
            )
        }
    )
    
    warmup_teacher_temp_epochs: int = field(
        default=30, 
        metadata={
            "help": (
                "The number of epochs to warm up the teacher model temperature. The "
                "temperature controls the smoothness of the soft targets."
            )
        }
    )
    
    n_views: int = field(
        default=2, 
        metadata={
            "help": (
                "The number of views (augmented data representations) for each sample "
                "during training. Used in contrastive learning."
            )
        }
    )
    
    warmup_teacher_temp: float = field(
        default=0.07, 
        metadata={
            "help": (
                "The initial value for teacher temperature during warm-up. This helps "
                "gradually soften the target distributions."
            )
        }
    )
    
    teacher_temp: float = field(
        default=0.04, 
        metadata={
            "help": (
                "The final temperature value for the teacher model. This controls how "
                "soft or sharp the teacher's predictions are."
            )
        }
    )
    
    memax_weight: float = field(
        default=2., 
        metadata={
            "help": (
                "The weight factor for Memax during training. It controls the contribution "
                "of Memax in the overall loss."
            )
        }
    )
    
    output_dir: str = field(
        default="./saves", 
        metadata={
            "help": (
                "Directory path where the model checkpoints and training outputs will be "
                "stored."
            )
        }
    )
    
    save_steps: int = field(
        default=5, 
        metadata={
            "help": (
                "Number of steps between saving model checkpoints. Saving checkpoints allows "
                "recovery from interrupted training."
            )
        }
    )
    
    logging_steps: int = field(
        default=10, 
        metadata={
            "help": (
                "Number of steps between logging outputs. Used for monitoring training "
                "progress."
            )
        }
    )
    
    logging_dir: str = field(
        default="./logs", 
        metadata={
            "help": (
                "Directory path where training logs will be saved. Useful for tracking "
                "training metrics."
            )
        }
    )
    
    per_device_train_batch_size: int = field(
        default=128, 
        metadata={
            "help": (
                "The batch size to use per device during training. Larger batch sizes "
                "speed up training but consume more memory."
            )
        }
    )
    
    per_device_eval_batch_size: int = field(
        default=256, 
        metadata={
            "help": (
                "The batch size to use per device during evaluation. Larger batch sizes "
                "speed up evaluation but consume more memory."
            )
        }
    )
    def get_args(self):
        
        return self.__dict__