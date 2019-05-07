# Implicit-VAE
This code repository presents the pytorch implementation of the paper "Implicit variational auto-encoder for text generation". 

Each folder corresponds to a specific experiment with varied dataset or topic. To run the code, download the zip file, extract it and cd to `Implicit-VAE/`. The file named as "train_xxx.py" in each folder is the main code to train and test our implicit VAE. 

# 1. Toy_onehot
The toy dataset contains 4 data points **x**: 4 different one-hot four-dimensional vectors, and we learn corresponding latent code **z** in 2D space for each **x**. Run the following in cmd lime:
```
cd toy_onehot/
python vae_onehot.py
python train_onehot.py
```
The result is as following:

VAE             |  Implicit-VAE
:--------------:|:-------------------------:
![](toy_onehot/results_vae/070000.png)  |  ![](toy_onehot/results/075000.png)

# Reference
  NA so far.
