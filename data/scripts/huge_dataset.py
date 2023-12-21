import requests
import pandas as pd
from PIL import Image
import io
import argparse

from captcha_solver.config import API_TOKEN


API_URL = "https://datasets-server.huggingface.co/parquet?dataset=hammer888/captcha-data"
DIR_TO_SAVE = '../huge_captchas/'


def underscore_remover(filename: str) -> str:
    """
    Save only the label in filename.
    For example 'Gs5y12_sqwehdfs23aeew2.png -> Gs5y12.png'
    :param filename:
    :return:
    """
    name, ext = filename.split('.')
    label = name.split('_')[0]
    return label + '.' + ext


def save_dataset(save_dir: str, k: int = 20) -> None:
    """
    Load parts of 'hammer888/captcha-data' and save it as images in local storage.
    :param save_dir: directory to save images.
    :param k: Save every k-th image, use k=1 to save all images.
    :return:
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.get(API_URL, headers=headers)
    data = response.json()
    for ind, parq_file in enumerate(data['parquet_files']):
        print('Part {}/{}:'.format(ind + 1, len(data['parquet_files'])), flush=True)
        url = parq_file['url']
        print('Loading parquet file.')
        df = pd.read_parquet(url, engine='pyarrow')
        print('Loading is complete. Start saving images.')
        for image_data in df.iloc[::k, :]['image'].values:
            image, path = image_data['bytes'], image_data['path']
            im = Image.open(io.BytesIO(image))
            path = underscore_remover(path)
            im.save(save_dir + path)
        print('Complete.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=20, type=int, help='Save every k-th image of all dataset')
    parser.add_argument('--save_dir', type=str, help='directory to save')
    args = parser.parse_args()
    save_dataset(args.save_dir, args.k)


