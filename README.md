# Semantic Segmentation of Cancer Cells
PyTorch implementation of several CNN-based models for segmentation of cancer cells.
# Requirement
Python 3.6.4

PyTorch 0.4.0

Some other libraries: NumPy 1.14.0, SciPy 1.0.0, Matpotlib 2.1.2

I recommend installation of [PyTorch](https://pytorch.org/) with CUDA using [Anaconda](https://anaconda.org/), which includes most of the libraries required. For example, Linux users can run the following command in the terminal:
```
conda install pytorch torchvision cuda91 -c pytorch
```

(This is my environment, but others may also work)
# Models
The idea of this model is from [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/abs/1505.04366/). Unlike the original model, fully connected layers as well as some other layers are dropped in the **DeConvNet**. The output stride is 16. A similar CNN model with skip connections was also implemented as **SegSkipNet**.

20-layer **DeConvNet** configuration:


![alt text](https://github.com/CoserU/cancer-cell-semantic-segmentation/blob/master/visulization/figures/deconvnet.png)


| Layer | ![](https://latex.codecogs.com/gif.latex?C%5Ctimes%20H%5Ctimes%20W) | Activations | Weights |
| ------------- |:-------------:| -----:| -----:|
| input | ![](https://latex.codecogs.com/gif.latex?3%5Ctimes%20320%5Ctimes%20320) | 307,200 | 0 |
| conv1-1 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6,553,600 | 1,728 |
| conv1-2 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6,553,600 | 36,864 |
| pool1 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20160%5Ctimes%20160) | 1,638,400 | 0 |
| conv2-1 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%20160%5Ctimes%20160) | 3,276,800 | 73,728 |
| conv2-2 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%20160%5Ctimes%20160) | 3,276,800 | 147,456 |
| pool2 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%2080%5Ctimes%2080) | 819,200 | 0 |
| conv3-1 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2080%5Ctimes%2080) | 1,638,400 | 294,912 |
| conv3-2 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2080%5Ctimes%2080) | 1,638,400 | 589,824 |
| pool3 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2040%5Ctimes%2040) | 409,600 | 0 |
| conv4-1 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2040%5Ctimes%2040) | 819,200 | 1,179,648 |
| conv4-2 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2040%5Ctimes%2040) | 819,200 | 2,359,296 |
| pool4 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2020%5Ctimes%2020) | 204,800 | 0 |
| conv5-1 | ![](https://latex.codecogs.com/gif.latex?1024%5Ctimes%2020%5Ctimes%2020) | 409,600 | 4,718,592 |
| conv5-2 | ![](https://latex.codecogs.com/gif.latex?1024%5Ctimes%2020%5Ctimes%2020) | 409,600 | 9,437,184 |
| deconv5-3 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2020%5Ctimes%2020) | 204,800 | 4,718,592 |
| unpool4 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2040%5Ctimes%2040) | 819,200 | 0 |
| deconv6-1 | ![](https://latex.codecogs.com/gif.latex?512%5Ctimes%2040%5Ctimes%2040) | 819,200 | 2,359,296 |
| deconv6-2 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2040%5Ctimes%2040) | 409,600 | 1,179,648 |
| unpool3 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2080%5Ctimes%2080) | 1,638,400 | 0 |
| deconv7-1 | ![](https://latex.codecogs.com/gif.latex?256%5Ctimes%2080%5Ctimes%2080) | 1,638,400 | 589,824 |
| deconv7-2 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%2080%5Ctimes%2080) | 819,200 | 294,912 |
| unpool2 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%20160%5Ctimes%20160) | 3,276,800 | 0 |
| deconv8-1 | ![](https://latex.codecogs.com/gif.latex?128%5Ctimes%20160%5Ctimes%20160) | 3,276,800 | 147,456 |
| deconv8-2 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20160%5Ctimes%20160) | 1,638,400 | 73,728 |
| unpool1 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6,553,600 | 0 |
| deconv9-1 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6,553,600 | 36,864 |
| deconv9-2 | ![](https://latex.codecogs.com/gif.latex?64%5Ctimes%20320%5Ctimes%20320) | 6,553,600 | 36,864 |
| output | ![](https://latex.codecogs.com/gif.latex?2%5Ctimes%20320%5Ctimes%20320) | 204,800 | 128 |


**U-Net** from [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597/) was implemented in the models. Here the paddings of 3 by 3 convolutional layers were set as 1 to preserve the spatial size.

The `model_stats.py` file in the misc folder can generate the configuration of the models, i.e., layers, activations and Mult-Adds.

# Visualization
The maximum activations of deconvolutional layers can be visualized by running `max_activation.py`, which illustrates coarse-to-fine pixel-wise segmentation via these learned layers. Besides, the features of the first convolutional layer can be visualized by running `visualize_features.py`.
