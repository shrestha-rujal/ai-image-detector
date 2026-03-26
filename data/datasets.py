import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class NPRTransform:
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        h, w = img_np.shape[:2]

        half = np.array(img.resize((w // 2, h // 2), Image.NEAREST))

        half_up = np.array(Image.fromarray(half).resize(
            (w, h), Image.NEAREST)).astype(np.float32)

        npr = img_np - half_up

        npr = npr - npr.min()
        npr = npr / (npr.max() + 1e-8) * 255
        npr = Image.fromarray(npr.astype(np.uint8))

        return npr


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        NPRTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = ImageFolder(
        root=data_dir,
        transform=get_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader, dataset
