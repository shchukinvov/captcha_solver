# captcha-solver
Solving some captchas using Neural Networks.

## Description
Study project for practice with **OCR** tasks by solving 
[captchas](https://en.wikipedia.org/wiki/CAPTCHA) using different CNN and RNN architectures.

## Tasks
- [ ] Collect and prepare data
- [ ] Test different models and algorithms
- [ ] Create inference tools
- [ ] Deploy the model

## Progress
### Data collection
- 1070 images from kaggle [dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images).
- [Captcha](https://github.com/lepture/captcha) library for python.
- Self-written captcha generator using `Numpy` and `OpenCV` libraries.
#### Downloading data
```commandline
kaggle datasets download -d fournierp/captcha-version-2-images
```
```commandline
tar -xf captcha-version-2-images.zip -C ./captcha_solver/data/captchas/
```
```commandline
mv ./captcha_solver/data/captchas/samples/samples/* ./captcha_solver/data/captchas/
```
```commandline
rm captcha-version-2-images.zip 
rm -r ./captcha_solver/data/captchas/samples
```
#### Generating data
```commandline
python captcha_solver/data/scripts/captcha_generator.py --num=2000 --min_length=4 --max_length=6
```
```commandline
python captcha_solver/data/scripts/captcha_generator_2.py --num=2000 --min_length=4 --max_length=6
```

Examples is given below.
- 
![1](https://github.com/shchukinvov/captcha_solver/blob/main/figure/generated_captcha_0.png)
![2](https://github.com/shchukinvov/captcha_solver/blob/main/figure/generated_captcha_1.png)
![3](https://github.com/shchukinvov/captcha_solver/blob/main/figure/generated_captcha_2.png)