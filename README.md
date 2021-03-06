# Conditional COT-GAN for Video Prediction with Kernel Smoothing

This repository contains an implementation and further details of kernel Conditional COT-GAN (KCCOT-GAN).

## Results

The green border indicates context sequences, and the red border shows the KCCOT-GAN predictions conditioned on the context sequences. 

### Moving MNIST results
![](./gifs/mmnist.gif)

### GQN Mazes results
![](./gifs/mazes.gif)

### BAIR Push Small results
![](./gifs/bair.gif)


## Data 

All datasets used in the experiments are publicly available.  

- The Moving MNIST dataset can be downloaded from [here](http://www.cs.toronto.edu/~nitish/unsupervised_video/).

- All GQN datasets are available at [here](https://github.com/deepmind/gqn-datasets) and the GQN mazes dataset can be downloaded from [this GCP bucket](https://console.cloud.google.com/storage/browser/gqn-dataset/mazes?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). 
- BAIR Push Small dataset is available in TensorFlow Datasets module, see [here](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small).
