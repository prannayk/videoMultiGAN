# Video Multi GAN
Video Generation from Text using Tree like decision using GANs. The text annotation or statement is encoded using the LM into a embedding, which then is combined with random vector to generate relevant videos and images. 

## Video Generation models
1. VAEGAN
2. VAEGAN with Latent Variable optimization
3. VAEGAN with anti reconstruction loss
4. VAEGAN + Anti reconstruction loss + Latent variable models
5. variants of above models with different Hyper parameters

## Model structure
* LSTM based model for next frame creation
* Wasserstein GAN setting discriminator 
* Word embedding based LM
* Attention based model for classification structure

## Training model
* The relevant models are in ``` Tensorflow >= v1.2 ```
* Experimentation with above mentioned models
* The training is done over self generated Bouncing MNIST with sentence based annotation
* The gensim pre trained fastText wikipedia work embeddings are used for embedding tokens as vectors
* Non attention based models are used initially to generate starting frames. 
* The GAN tree trains to look for discriminative features (unverified)

## Datasets
1. UCF101 : 3 channel image
2. Bouncing MNIST

## Documentation
1. We use Sync-DRAW to develop our datasets (https://github.com/syncdraw/Sync-DRAW)
2. UCF101 is available from University of Montreal
3. We use multiple GPU training (or a single K80 or Titan X)
4. Cluster traning is impossible for now

### Results will not be updated here since there might be related publications.
