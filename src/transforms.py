import albumentations as A
from albumentations.pytorch import ToTensorV2


TRAIN_TRANSFORMS = A.Compose([
    A.Resize(75, 200, always_apply=True),
    A.ToFloat(max_value=255),
    ToTensorV2(),
])

VAL_TRANSFORMS = A.Compose([
    A.Resize(75, 200, always_apply=True),
    A.ToFloat(max_value=255),
    ToTensorV2(),
])
