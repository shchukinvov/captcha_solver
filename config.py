from dotenv import load_dotenv
import os

load_dotenv()

API_TOKEN = os.environ.get("API_HUB_TOKEN")

TRAIN_IMAGE_DIR = '../data/captchas'
TEST_IMAGE_DIR = '../data/test'

