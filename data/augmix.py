from typing import Callable
from PIL import Image
import warnings

import torch
from torchvision import transforms

# ImageNet mean and std
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CLViewGen(object):
    """Take two random crops of one image as the query and key."""
    
    def __init__(
        self, 
        base_tfm: Callable[[Image.Image], torch.Tensor], 
        n_views: int = 2,
    ):
        self.base_tfm = base_tfm
        self.n_views = n_views
    
    def __call__(self, x: Image.Image) -> list[torch.Tensor]:
        """Generate two views of the image."""
        
        return [self.base_tfm(x) for _ in range(self.n_views)]



def get_transform(
    image_size: int = 32,
    crop_pct: float = 0.875, 
    interpolation: int = 3, 
    use_contrast: bool = True, 
    n_views: int = 2,
    **kwargs,
) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    """ Get the train and test transforms """
    
    train_transform = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean), std=torch.tensor(std)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean), std=torch.tensor(std)
        )
    ])
    
    if use_contrast:
        train_transform = CLViewGen(train_transform, n_views=n_views)
    
    return train_transform, test_transform

