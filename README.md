This repository contains a Pytorch implementation of an image captioning model that is inspired by the Show, Attend and Tell paper (https://arxiv.org/abs/1502.03044) and the Sequence Generative Adversarial Network (SeqGAN) paper (https://arxiv.org/abs/1609.05473). For readability and convenience, the 3 main stages of the training process have been divided into 3 python scripts:
* train_mle.py: pretraining the generator using Maximum Likelihood estimation (essentially the same as the Show, Attent and Tell model)
* pretrain_discriminator.py: pretrain the discriminator, which is a GRU that takes the features of an image as its first input followed by its corresponding caption
* train_pg.py: adversarial training using policy gradients as proposed in the SeqGAN paper

The code is functional (if I do find bugs, I will try to fix it immediately). For the MLE stage, I have been able to get a BLEU-4 score of 0.197 on the validation set (without beam search) for Flickr8k. For the adversarial training stage, I was able to get the BLEU-4 score to rise to 0.21 from 0.197; however, I am still tuning the hyperparameters of the model. Research papers proposing similar models have been vague about how many iterations the discriminator is trained on for every iteration that the generator is trained on during adversarial training i.e., the generator-discriminator iterations ratio. During adversarial training, it is better to train the discriminator on more iterations than the generator (I am yet to find the ideal ratio). I have currently set the ratio to 10:1, mostly based on observations provided in https://arxiv.org/abs/1804.00861. 

To run the program:
* Place the data splits (http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) provided by Andrej Karpathy in the karpathy_splits folder
* Place images from flickr8k, flickr30, or coco within the images folder (make sure to place it in the correct subdirectory)
* Run preprocess.py
* Run train_mle.py
* Run pretrain_discriminator.py
* Run train_pg.py
