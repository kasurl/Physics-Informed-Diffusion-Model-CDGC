import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import argparse


class RoadCrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_names = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
        mask_names = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
        assert len(image_names) == len(mask_names), "Image/mask count mismatch"

        for img_name, mask_name in zip(image_names, mask_names):
            assert img_name.split('.')[0] == mask_name.split('.')[0], f"Filename mismatch: {img_name} and {mask_name}"

        self.image_filenames = [os.path.join(images_dir, f) for f in image_names]
        self.mask_filenames = [os.path.join(masks_dir, f) for f in mask_names]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")

        # Convert mask to binary
        mask = np.array(mask)
        if np.unique(mask).size > 2:
            threshold = np.max(mask) * 0.5
            mask = (mask > threshold).astype(np.uint8) * 255

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()

        # Debug info on first call
        if idx == 0 and not hasattr(self, 'debug_printed'):
            print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Mask shape: {mask.shape}, unique values: {torch.unique(mask)}")
            self.debug_printed = True

        return image, mask


def create_data_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size=4):
    """Create train and validation data loaders"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = RoadCrackDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        transform=transform
    )

    val_dataset = RoadCrackDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create data loaders for training')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Training images directory')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='Training masks directory')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_mask_dir', type=str, required=True, help='Validation masks directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

    args = parser.parse_args()

    train_dataloader, val_dataloader = create_data_loaders(
        args.train_img_dir,
        args.train_mask_dir,
        args.val_img_dir,
        args.val_mask_dir,
        args.batch_size
    )

    print(f"Train dataset: {len(train_dataloader.dataset)} images")
    print(f"Val dataset: {len(val_dataloader.dataset)} images")