This repository contains code for an image captioning model that is inspired by the Show, Attend and Tell paper (https://arxiv.org/abs/1502.03044) and the Sequence Generative Adversarial Network (SeqGAN) paper (https://arxiv.org/abs/1609.05473). For readability and convenience, the 3 main stages of the training process have been divided into 3 python scripts:
* train_mle.py: pretraining the generator using Maximum Likelihood estimation (essentially the same as the Show, Attent and Tell model)
* pretrain_discriminator.py: pretrain the discriminator, which is a GRU that takes the features of an image as its first input followed by its corresponding caption
* train_pg.py: adversarial training using policy gradients as proposed in the SeqGAN paper

Although the code is functional, the hyperparameters of the model are yet to be tuned (this model is extremely sensitive to the hyperparameters that are chosen). I could not find a reliable implementation of the seqgan for image captioning. Therefore,  the best hyperparameters have to be found from scratch (research papers proposing similar models have been vague about how many iterations the discriminator is trained on for every iteration that the generator is trained on during adversarial training i.e., the generator-discriminator iterations ratio). When I say an iteration, I mean training on a single batch of data.

The code I have written is based on the following awesome repositories:
* https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
* https://github.com/X-czh/SeqGAN-PyTorch
