# Triangular network for density estimation and data generation
Mainly prepared for reproducing the experimental results in report "A Triangular Network For Density Estimation". You need Pytorch to run these demos. Please check utility.MonoTriNetInit and utility.MonoTriNet for usages. 

### Density estimation demos
##### Toy demo
Please check toy_density_estimate_demo for details. 
##### MNIST demo
Please check MNIST_Cshift_Adam_Log_NoFlip for details. Test bits-per-dimension should be below 1.1.
##### CIFAR demo
Please check CIFAR10_Cshift_Adam_Log for details. Test bits-per-dimension should be below 3.7. 

### The optimizer
I used [PSGD](https://github.com/lixilinx/psgd_torch) for some time. Replaced trtrs in the original copy with triangular_solve due to Pytorch's API changes. Later I switched to Adam because now I squeeze all parameters of a MonoTriNetUnit into one dense matrix, and those sparse (scaling/normalization/whitening) preconditioners defined in PSGD lose their meanings. 

### Old data generation demos (not maintained)
Some old stuff that I do not clean up them yet. Many high dimensional data have very low intrinsic dimensions. It is way easier to estimate their densities in their latent spaces.    
###### MNIST example
Dimension of the latent variable is 16. Samples of the original (top), reconstructed (middle) and randomly generated digits (bottom) are shown as below. To reproduce these results, set random_init = True, and run mnist_dim_reduction, mnist_density and mnist_demo successively.         

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/mnist_demo.png)

##### CelebA example
Dimension of the latent variable is 128. Samples of the original (top), reconstructed (middle) and randomly generated faces (bottom) are shown as below. To reproduce these results, download CelebA, set the datasets.ImageFolder correctly and random_init = True, and run celeba_dim_reduction, celeba_density and celeba_demo successively.

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/celeba_demo.png)
