import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import defaultdict


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

    def get_weights(self) -> dict:
        """
        Captcha name should look like 'type?_label.png'.
        For example 'type1_1y32fx.png'.
        :return: list of weights.
        """
        counter = defaultdict(int)
        for label in self.images:
            if '_' not in label:
                raise RuntimeError("Images is not separated by types")
            counter[label.split('_')[0]] += 1

        return {'weights': [1 / counter[label.split('_')[0]] for label in self.images],
                'num_samples': len(self.images)}


def init_dataloaders(
        image_dir,
        test_image_dir,
        train_transforms,
        val_transforms,
        test_size=0.2,
        batch_size=16,
        rand_state=0,
        num_workers=2,
):
    data = os.listdir(image_dir)
    test_data = os.listdir(test_image_dir)
    train_data, val_data = train_test_split(data,
                                            test_size=test_size,
                                            shuffle=True,
                                            random_state=rand_state)

    train_dataset = CaptchaDataset(train_data, image_dir, train_transforms)
    val_dataset = CaptchaDataset(val_data, image_dir, val_transforms)
    test_dataset = CaptchaDataset(test_data, test_image_dir, val_transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=8,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=1)

    return train_dataloader, val_dataloader, test_dataloader


""" TESTING """
if __name__ == "__main__":
    from transforms import TRAIN_TRANSFORMS, VAL_TRANSFORMS
    from captcha_solver.config import TRAIN_IMAGE_DIR, TEST_IMAGE_DIR
    loaders = init_dataloaders(TRAIN_IMAGE_DIR,
                               TEST_IMAGE_DIR,
                               TRAIN_TRANSFORMS,
                               VAL_TRANSFORMS)
    pass
