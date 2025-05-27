import os

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomImageMaskDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.transform = transform
        self.target_transform = target_transform

        self.filenames = sorted(os.listdir(self.image_dir))
        total_len = len(self.filenames)
        train_end = int(0.7 * total_len)
        valid_end = int(0.85 * total_len)

        if split == "train":
            self.filenames = self.filenames[:train_end]
        elif split == "valid":
            self.filenames = self.filenames[train_end:valid_end]
        elif split == "test":
            self.filenames = self.filenames[valid_end:]
        else:
            raise ValueError("Invalid split: choose from 'train', 'valid', or 'test'")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Replace extension to .png for label
        label_name = os.path.splitext(image_name)[0] + ".png"
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        state = torch.get_rng_state()  # Save the random state for reproducibility
        if self.transform:
            image = self.transform(image)
        # Restore the random state to ensure consistent transformations
        torch.set_rng_state(state)
        if self.target_transform:
            label = self.target_transform(label)

        return {"image": image, "mask": label, "filename": image_name}


def generate_dataset(root):
    transform = T.Compose(
        [
            T.RandomCrop((1024, 1024)),
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    target_transform = T.Compose(
        [
            T.RandomCrop((1024, 1024)),
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    train_dataset = CustomImageMaskDataset(root, "train", transform, target_transform)
    valid_dataset = CustomImageMaskDataset(root, "valid", transform, target_transform)
    test_dataset = CustomImageMaskDataset(root, "test", transform, target_transform)

    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset


def generate_dataloaders(train_dataset, valid_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    return train_dataloader, valid_dataloader, test_dataloader


# Helper function to visualize image and mask
def save_sample(dataset, index):
    image, mask = dataset[index]  # returns a tuple
    image = image.permute(1, 2, 0)  # CxHxW → HxWxC
    mask = mask.squeeze()  # 1xHxW → HxW

    fig = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    return fig


def main():
    from pathlib import Path

    path = Path("./examples")
    path.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = generate_dataset("./dataset")
    fig = save_sample(train_ds, 0)
    fig.savefig(path / "sample_train.png")
    fig = save_sample(val_ds, 0)
    fig.savefig(path / "sample_valid.png")
    fig = save_sample(test_ds, 0)
    fig.savefig(path / "sample_test.png")
    train_dl, val_dl, test_dl = generate_dataloaders(train_ds, val_ds, test_ds)
    print("Data loaders created successfully.")
    print(f"Train DataLoader size: {len(train_dl)}")
    print(f"Valid DataLoader size: {len(val_dl)}")
    print(f"Test DataLoader size: {len(test_dl)}")


if __name__ == "__main__":
    main()
