# Triangular network for density estimation and data generation
A highly compact and modular monotonic triangular network is implemented. We applied it to density estimation, and achieved the-state-of-art results on challenging image density estimation benchmarks in the category of general density estimators. Please check [A Triangular Network For Density Estimation](https://arxiv.org/pdf/2004.14593.pdf) for details.   

Please check utility.MonoTriNetInit and utility.MonoTriNet for usage of this network. The default activation function is tanh; use log, sign(x)log(1 + abs(x)), if bijection is a must. When several monotonic triangular network units are cascaded, by default, outputs of a unit is flipped before feeding to its successive one. You may set flip to False to disable it.     

### Density estimation demos
##### Toy demo
Please check toy_density_estimate_demo for details. 
##### MNIST demo
Run MNIST_L1Reg_Adam_Tanh_NoFlip for test results with L1 regularization. Test bits-per-dimension should be slightly below 1.13.

Run MNIST_Cshift_Adam_Log_NoFlip for test results with data augmentation. Test bits-per-dimension should be below 1.1.
##### CIFAR demo
Run CIFAR10_L1Reg_Adam_Log for test results with L1 regularization. Test bits-per-dimension should be slightly below 3.7. 

Run CIFAR10_Cshift_Adam_Log for test results with data augmentation. Test bits-per-dimension should be below 3.7. 

These results are significantly better than Transformation Autoregressive Networks (TAN), one of the best general density estimators we know. Still, we suspect that there might be further room for improvement as the train bits-per-dimension can go below 3.4 without early stopping.     

### The optimizer
I used [PSGD](https://github.com/lixilinx/psgd_torch) for some time. Replaced trtrs in the original copy with triangular_solve due to Pytorch's API changes. Later I switched to Adam because now I squeeze all parameters of a MonoTriNetUnit into one dense matrix, and those sparse (scaling/normalization/whitening) preconditioners defined in PSGD lose their meanings. 

### Some earlier code and experimental results (not maintained)
There is some eariler work left here, mainly for data generation with ordinary triangular networks. It looks difficult to scale up them to problems with dimensions up to thousands. Still, we successfully trained some generative models in the latent spaces.
###### MNIST example
Dimension of the latent variable is 16. Samples of the original (top), reconstructed (middle) and randomly generated digits (bottom) are shown as below. To reproduce these results, set random_init = True, and run mnist_dim_reduction, mnist_density and mnist_demo successively.         

![alt text](https://github.com/lixilinx/TriNet4PdfEst/blob/master/misc/mnist_demo.png)

##### CelebA example
Dimension of the latent variable is 128. Samples of the original (top), reconstructed (middle) and randomly generated faces (bottom) are shown as below. To reproduce these results, download CelebA, set the datasets.ImageFolder correctly and random_init = True, and run celeba_dim_reduction, celeba_density and celeba_demo successively.

![alt text](https://github.com/lixilinx/TriNet4PdfEst/blob/master/misc/celeba_demo.png)
