# Implicit-VAE
This code repository presents the pytorch implementation of the paper "Implicit variational auto-encoder for text generation". 

Each folder corresponds to a specific experiment with varied dataset or topic. To run the code, download the zip file, extract it and cd to `Implicit-VAE/`. The file named as `train_xxx.py` in each folder is the main code to train and test our implicit VAE. 

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

# 2. Language modeling on PTB
After downloading, run
```
cd lang_model_ptb/
python preprocess_ptb.py --trainfile data/train.txt --valfile data/val.txt --testfile data/test.txt --outputfile data/ptb
```
This will create the `*.hdf5` files (data tensors) to be used by the model, as well as the `*.dict` file which contains the word-to-integer mapping for each word.

Specify '--model' in cmd line, with '--model mle' for implicit vae (default if not specified) and '--model mle_mi' for implicit vae with mutual information maximized. The command for training is for example
```
python train_ptb.py
```
The command for evaluation after the 30th epoch is
```
python train_ptb.py --test --train_from results_mle/030.pt
```
The command for training VAEs ('vae', 'beta_vae', 'savae', 'cyc_vae') is for example
```
python train_ptb_vaes.py --model vae
```
For interpolating between 2 sentences after training the 40th epoch, run
```
python interpolation.py
```
For evaluating decoders from prior codes after training and calculate forward & reverse PPL, we need install [KenLM Language Model Toolkit](https://github.com/kpu/kenlm) and run
```
python generative_model.py --model ['mle', 'mle_mi', 'vae','beta_vae', 'savae', 'cyc_vae']
```
# 3. Language modeling on Yahoo
After downloading, run
```
cd lang_model_yahoo/
python preprocess_yahoo.py --trainfile data/train.txt --valfile data/val.txt --testfile data/test.txt --outputfile data/yahoo
```
This will create the `*.hdf5` files (data tensors) to be used by the model, as well as the `*.dict` file which contains the word-to-integer mapping for each word.

Specify '--model' in cmd line, with '--model mle' for implicit vae (default if not specified) and '--model mle_mi' for implicit vae with mutual information maximized. The command for training is for example
```
python train_yahoo.py
```
The command for evaluation after the 30th epoch is
```
python train_yahoo.py --test --train_from results_mle/030.pt
```
# 4. Language modeling on Yelp
Specify '--model' in cmd line, with '--model mle' for implicit vae (default if not specified) and '--model mle_mi' for implicit vae with mutual information maximized. The command for training is for example
```
python train_yelp.py
```
The command for evaluation after, e.g., the 30th epoch is
```
python train_yelp.py --test --train_from results_mle/030.pt
```
The command for training an autoencoder (AE) is
```
python train_yelp_ae.py
```
The command for training VAEs ('vae', 'beta_vae', 'cyc_vae') is for example
```
python train_yelp_vaes.py --model vae
```

# Reference
  NA so far.
