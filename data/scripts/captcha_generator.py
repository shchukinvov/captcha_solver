import random
import numpy as np
import cv2
import argparse
from generator_config import NOISE_PARAMS, CHARACTERS, NUMS, FONTS


class ImageCaptchaGenerator:
    """
    Create a color image CAPTCHA.
    :param width: The width of the captcha image
    :param height: The height of the captcha image
    :param min_length: Minimum length of generated characters sequence
    :param max_length: Maximum length of generated characters sequence
    :param noise_level: Captcha's noise level min | medium | high
    :return Captcha image
    """
    def __init__(self,
                 width: int = 300,
                 height: int = 150,
                 min_length: int = 5,
                 max_length: int = 8,
                 noise_level: str = "low",
                 ):
        self._width = width
        self._height = height
        if min_length > max_length:
            raise ValueError("max length should be higher or equal than min length")
        self._min_length = min_length
        self._max_length = max_length
        if noise_level not in ("low", "medium", "high"):
            raise ValueError("Chose noise level from 'low'|'medium'|'high'")
        self._noise_level = noise_level
        self._noise_params = NOISE_PARAMS
        self._fonts = FONTS
        self._characters = CHARACTERS
        self._nums = NUMS

    def _create_background(self) -> np.ndarray:
        scale, fq_low, fq_high = self._noise_params["bg_params"][self._noise_level].values()
        d1, d2, d3 = (np.random.uniform(self._width / fq_low, self._width / fq_high),
                      np.random.uniform(self._width / fq_low, self._width / fq_high),
                      np.random.uniform(self._width / fq_low, self._width / fq_high))
        bg_idx = np.zeros((2, self._height, self._width, 3))
        bg_idx[..., :, 0] = np.indices((self._height, self._width)) / d1
        bg_idx[..., :, 1] = np.indices((self._height, self._width)) / d2
        bg_idx[..., :, 2] = np.indices((self._height, self._width)) / d3
        bg = 1 - ((np.sin(bg_idx[0]) + np.sin(bg_idx[1]) + 2) / 4) * scale
        return bg

    def _create_noise_curve(self, image: np.ndarray) -> np.ndarray:
        num_lines, thickness, fq = self._noise_params["curve_params"][self._noise_level].values()
        crv_x = np.arange(0, self._width, fq)
        crv_y = np.full(fill_value=self._height, shape=crv_x.shape)
        crvs = np.stack((crv_x, crv_y), axis=1)
        crvs = np.array([
            [(x + np.random.randint(0, 19), y - np.random.randint(1, self._height-1)) for (x, y) in crvs]
            for _ in range(num_lines)]
        )
        for crv in crvs:
            cv2.polylines(img=image,
                          pts=[crv],
                          isClosed=False,
                          color=[np.random.rand() for _ in range(3)],
                          thickness=thickness)
        return image

    def _draw_character(self, char: str, box_size: int = 60) -> np.ndarray:
        N = box_size
        box = np.zeros((N, N, 3))

        # Affine transform
        src_mat = np.array([
            [0, 0],
            [box.shape[1] - 1, 0],
            [0, box.shape[0] - 1]
        ]).astype(np.float32)

        dst_mat = np.array([
            [0, 0],
            [np.random.randint(N // 2, N), np.random.randint(0, N // 2)],
            [np.random.randint(0, N // 2), np.random.randint(N // 2, N)]
        ]).astype(np.float32)

        warp_mat = cv2.getAffineTransform(src_mat, dst_mat)
        color = [np.random.rand() / 2 for _ in range(3)]
        font = random.choice(self._fonts)
        cv2.putText(box, char, (0, N-N//3), font, 1.4, color=color, thickness=3)
        box = cv2.warpAffine(box, warp_mat, (N, N))

        return box

    def _create_sequence(self) -> str:
        l = random.randint(self._min_length, self._max_length)
        seq = []
        for _ in range(l):
            seq.append(random.choice(random.choice([self._nums, self._characters])))
        return "".join(seq)

    def _create_captcha(self) -> tuple[np.ndarray, str]:
        box_size = 50
        captcha_bg = self._create_background()
        seq = self._create_sequence()
        cords = np.arange(10, self._width - box_size, (self._width - box_size) // len(seq))
        for char, pos in zip(seq, cords):
            captcha_bg[self._height//2-25:self._height//2+25, pos:pos+50, :] += self._draw_character(char, box_size)

        captcha = self._create_noise_curve(captcha_bg)
        return 255*captcha.clip(0, 1), seq

    def __iter__(self):
        return self

    def __next__(self):
        return self._create_captcha()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nums', default=10000, type=int, help='amount of images')
    parser.add_argument('--width', default=200, type=int, help='width of captcha image')
    parser.add_argument('--height', default=50, type=int, help='height of captcha image')
    parser.add_argument('--min_length', default=5, type=int, help='min length of char sequence')
    parser.add_argument('--max_length', default=8, type=int, help='max length of char sequence')
    parser.add_argument('--noise_level', default='low', type=str, help='noise level low|medium|high')
    parser.add_argument('--save_dir', default='./captcha_solver/data/captchas/', type=str, help='directory to save')
    args = parser.parse_args()
    captcha_images = iter(ImageCaptchaGenerator(width=args.width,
                                                height=args.height,
                                                min_length=args.min_length,
                                                max_length=args.max_length,
                                                noise_level=args.noise_level)
                          )
    for i in range(args.nums):
        im, char_seq = next(captcha_images)
        fpath = args.save_dir + char_seq + '.png'
        cv2.imwrite(fpath, im)
