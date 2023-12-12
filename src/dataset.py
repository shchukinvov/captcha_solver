import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CaptchaDataset(Dataset):
    def __init__(self, images, image_dir, transforms=None):
        super(CaptchaDataset, self).__init__()
        self.images = images
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        image = cv2.imread(image_path)
        label = self.images[item].split('.')[0]
        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label


def init_dataloaders(
        image_dir,
        train_transforms,
        val_transforms,
        test_size=0.25,
        batch_size=32,
        rand_state=0,
        num_workers=2,
):
    data = os.listdir(image_dir)
    train_data, val_data = train_test_split(data, test_size=test_size, shuffle=True, random_state=rand_state)

    train_dataset = CaptchaDataset(train_data, image_dir, train_transforms)
    val_dataset = CaptchaDataset(val_data, image_dir, val_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader


""" TESTING """
if __name__ == "__main__":
    from transforms import TRAIN_TRANSFORMS
    captcha_dir = '../data/captchas'
    train, val = init_dataloaders(captcha_dir, train_transforms=TRAIN_TRANSFORMS, val_transforms=TRAIN_TRANSFORMS)
    for im, lbl in train:
        print(im.shape)
        print(lbl)
        break
