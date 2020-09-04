To run the program:
* Place the data splits (http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) provided by Andrej Karpathy in the karpathy_splits folder
* Place images from flickr8k, flickr30, or MSCOCO within the images folder (make sure to place it in the correct subdirectory). For MSCOCO, place the train2014 and val2014 folders as is into the correct image folder
* Run preprocess.py
* Run train_mle.py
* Run pretrain_discriminator.py. Make sure to provide the name of one of the checkpoints that were created when train_mle.py was run
* Run train_pg.py. Again, please provide the names of checkpoints for both the generator and discriminator as command-line arguments
