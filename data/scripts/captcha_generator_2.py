import random
from captcha.image import ImageCaptcha
from PIL import Image
import argparse
from generator_config import CHARACTERS, NUMS


def create_sequence(min_length: int = 5, max_length: int = 5) -> str:
    if min_length > max_length:
        raise ValueError("max length should be higher or equal than min length")
    l = random.randint(min_length, max_length)
    seq = []
    for _ in range(l):
        seq.append(random.choice(random.choice([NUMS, CHARACTERS])))
    return "".join(seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nums', default=10000, type=int, help='amount of images')
    parser.add_argument('--width', default=200, type=int, help='width of captcha image')
    parser.add_argument('--height', default=50, type=int, help='height of captcha image')
    parser.add_argument('--min_length', default=5, type=int, help='min length of char sequence')
    parser.add_argument('--max_length', default=8, type=int, help='max length of char sequence')
    parser.add_argument('--save_dir', default='./captcha_solver/data/captchas/', type=str, help='directory to save')
    args = parser.parse_args()
    captcha = ImageCaptcha(width=args.width,
                           height=args.height)
    for _ in range(args.nums):
        char_seq = create_sequence(min_length=args.min_length,
                                   max_length=args.max_length)
        fpath = args.save_dir + char_seq + '.png'
        data = captcha.generate(char_seq)
        im = Image.open(data)
        im.save(fpath)
