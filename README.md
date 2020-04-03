# Triangular network for density estimation and data generation
You need Pytorch to run these demos. Trained models are put [here](https://drive.google.com/open?id=10ducfJ8RcicJ548CWkjWVe7rxrxrLi18).  
### Toy examples
Check toy_density_estimate_demo and toy_data_generation_demo. 

### Density estimation demos
We consider two typical benchmarks.
##### MNIST
Preprocessing includes: 28x28 handwritten digit images -> 784 vector -> logit transform -> add noise. Data is normalized before feeding to the network, and normalization is a part of the model. The test negative-log-likelihood by our model is about 60, i.e., 0.11 bits per dimension. Check mnist_mono_tri_network_nll for details.
##### CIFAR
Preprocessing includes: 3x32x32 images -> 3072 vector -> add noise -> to range [-1, 1]. Data is normalized before feeding to the network, and normalization is a part of the model. Test negative-log-likelihood by our model is about -6120. Check cifar10_mono_tri_network_nll for details.   

### Data generation demos
These are the typical examples for testing autoencoders. 
##### MNIST example
Dimension of the latent variable is 16. Samples of the original digits (top), reconstructed (middle) and randomly generated ones (bottom) are shown as below. These generated images are sharper than those from many popular methods like variational auto-encoders. Unlike GAN, the training is stable as there is just one cost to minimize. To reproduce these results, set random_init = True, and run mnist_dim_reduction, mnist_density and mnist_demo successively.         

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/mnist_demo.png)

##### CelebA example
Dimension of the latent variable is 128. Samples of the original faces (top), reconstructed (middle) and randomly generated ones (bottom) are shown as below. To reproduce these results, download CelebA, set the datasets.ImageFolder correctly and random_init = True, and run celeba_dim_reduction, celeba_density and celeba_demo successively.

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/celeba_demo.png)

### The optimizer
Except for the cifar density demo, I always use the [PSGD](https://github.com/lixilinx/psgd_torch) method. Replaced trtrs in the original copy with triangular_solve due to Pytorch's API changes. It's a second-order method using normalized step size. I used step size 0.01 without any tweaking.

PSGD needs Hessian-vector product, and runs out of memory in the cifar density demo on my machine (one 1080 ti GPU). I used RMSProp for this problem, and manually reduced the learning rate by one order of magnitude after 100 epochs.   

###### To the COVID-19 lockup days
