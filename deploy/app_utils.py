import cv2
import numpy as np
from itertools import groupby

CHARACTERS_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
CHARACTERS_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMS = "0123456789"
CHARACTERS = NUMS + CHARACTERS_UPPERCASE + CHARACTERS_LOWERCASE
CHAR_LIST = ["-"] + [ch for ch in CHARACTERS]


class LabelEncoder:
    def __init__(self, char_list):
        self.encoder = {}
        self.decoder = {}
        for idx, char in enumerate(char_list):
            self.encoder[char] = idx
            self.decoder[idx] = char

    def encode(self, seq: str) -> list[int]:
        return [self.encoder[char] for char in seq]

    def decode(self, seq: list[int]):
        return ''.join([self.decoder[idx] for idx in seq])

    def decode_prediction(self, prediction: np.ndarray) -> str:
        blank_idx = 0
        prediction = prediction.argmax(axis=2)
        a = [k.item() for k, _ in groupby(prediction) if k != blank_idx]
        return self.decode(a)


def preprocess_image(image: str) -> np.ndarray:
    image = cv2.imread(image)
    image = cv2.resize(image, (200, 75)).transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    return (image / 255.).astype(np.float32)
