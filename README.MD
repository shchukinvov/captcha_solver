# captcha-solver
Solving some captchas using Neural Networks.

## Description
Project for practice with **OCR** tasks by solving 
[captchas](https://en.wikipedia.org/wiki/CAPTCHA) using different CNN and RNN architectures.

## Tasks
- [x] Collect and prepare data
- [x] Test different models and algorithms
- [x] Create inference tools
- [x] Deploy the model

## Progress
### Data collection
- 1070 images from kaggle [dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images).
- 1.37kk captcha [dataset](https://huggingface.co/datasets/hammer888/captcha-data)
- [Captcha](https://github.com/lepture/captcha) library for python.
- Self-written captcha generator using `Numpy` and `OpenCV` libraries.
#### Downloading data
```commandline
kaggle datasets download -d fournierp/captcha-version-2-images
```
```commandline
python captcha_solver/data/scripts/huge_dataset.py -k 20 --save_dir=captcha_solver/data/captchas/
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

### Metrics
- Character Error Rate (CER).
- Accuracy (percent of fully corrected labels)

### Models
1. CNN &#8594; LSTM &#8594; FC layer
2. CNN &#8594; Self Attention &#8594; FC layer

### Loss function
- CTCLoss

### Results
Achieved with **CNN** &#8594; **LSTM** &#8594; **FC layer** architecture.
- `CER = 0.197`
- `Accuracy = 0.75`

### Deploy model
Demo of the model presented [here](https://huggingface.co/spaces/shchukinvov/captcha_solver).
Deployment code presented in `deploy` directory.

