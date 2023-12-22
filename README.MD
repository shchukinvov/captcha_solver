# captcha-solver
Solving some captchas using Neural Networks.

## Description
Study project for practice with **OCR** tasks by solving 
[captchas](https://en.wikipedia.org/wiki/CAPTCHA) using different CNN and RNN architectures.

## Tasks
- [x] Collect and prepare data
- [ ] Test different models and algorithms
- [ ] Create inference tools
- [ ] Deploy the model

## Progress
### Data collection
- 1070 images from kaggle [dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images).
- 1.7kk captcha [dataset](https://huggingface.co/datasets/hammer888/captcha-data)
- [Captcha](https://github.com/lepture/captcha) library for python.
- Self-written captcha generator using `Numpy` and `OpenCV` libraries.
#### Downloading data
```commandline
kaggle datasets download -d fournierp/captcha-version-2-images
```
#### Generating data
```commandline
python captcha_solver/data/scripts/captcha_generator.py --num=2000 --min_length=4 --max_length=6
```
```commandline
python captcha_solver/data/scripts/captcha_generator_2.py --num=2000 --min_length=4 --max_length=6
```

#### Examples is given below.
![1](https://github.com/shchukinvov/captcha_solver/blob/main/figure/2bg48.png)
![2](https://github.com/shchukinvov/captcha_solver/blob/main/figure/2C16.png)
![3](https://github.com/shchukinvov/captcha_solver/blob/main/figure/2C518y.png)


### Models
1. Image -> `CNN -> LSTM -> FC layer` -> Label decoder -> text.
2. Image -> `CNN -> Self Attention -> FC layer` -> Label decoder -> text.`

Loss function - `CTCLoss`
