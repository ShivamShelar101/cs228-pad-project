"""
PAD dataset loader.

Expects a directory of the form:
    root/
      train/live/*.jpg
      train/spoof/*.jpg
      val/live/*.jpg
      val/spoof/*.jpg
      test/live/*.jpg
      test/spoof/*.jpg

(prepare_celeba_spoof.py below builds this layout from raw CelebA-Spoof.)
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders(root: str, batch_size: int = 32, num_workers: int = 2):
    """
    Returns (train_loader, val_loader, test_loader, class_to_idx).
    class_to_idx tells you which integer ImageFolder assigned to 'live'
    vs 'spoof' (alphabetical by default: live=0, spoof=1) -- always read
    it rather than assuming, since a re-run with different folder naming
    would silently flip your labels otherwise.
    """
    train_ds = datasets.ImageFolder(f"{root}/train", transform=get_transforms(True))
    val_ds = datasets.ImageFolder(f"{root}/val", transform=get_transforms(False))
    test_ds = datasets.ImageFolder(f"{root}/test", transform=get_transforms(False))

    for ds in (train_ds, val_ds, test_ds):
        assert set(ds.class_to_idx.keys()) == {"live", "spoof"}, ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx
