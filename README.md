# Triangular network for density estimation and data generation
You need Pytorch to run these demos. Trained models are put [here](https://drive.google.com/open?id=10ducfJ8RcicJ548CWkjWVe7rxrxrLi18).  
### Toy examples
Check toy_density_estimate_demo and toy_data_generation_demo. 

### Density estimation demos
We consider two typical benchmarks.
#### MNIST
Preprocessing before density estimation: 28x28 handwritten digit images -> 784 vecor -> logit transform. The negative-log-likelihood by our model should be below 80, i.e., below 0.15 bits per dimension. Check mnist_mono_tri_network_nll for details.
#### CIFAR

### Data generation demos
### MNIST example
Dimension of the latent variable is 16. Samples of the original digits (top), reconstructed (middle) and randomly generated ones (bottom) are shown as below. These generated images are sharper than those from many popular methods like variational auto-encoders. Unlike GAN, the training is stable as there is just one cost to minimize. To reproduce these results, set random_init = True, and run mnist_dim_reduction, mnist_density and mnist_demo successively.         

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/mnist_demo.png)

### CelebA example
Dimension of the latent variable is 128. Samples of the original faces (top), reconstructed (middle) and randomly generated ones (bottom) are shown as below. To reproduce these results, download CelebA, set the datasets.ImageFolder correctly and random_init = True, and run celeba_dim_reduction, celeba_density and celeba_demo successively. (Note: In the [prepared models](https://drive.google.com/open?id=10ducfJ8RcicJ548CWkjWVe7rxrxrLi18), I interrupted celeba_density after about one day. Only tens of epoches, not 100 as in the code, are done.)

![alt text](https://github.com/lixilinx/DensityEstimateWithEmpiricallyBijectiveMapping/blob/master/misc/celeba_demo.png)

### The optimizer
The [PSGD](https://github.com/lixilinx/psgd_torch) method. Replaced trtrs in the original copy with triangular_solve due to Pytorch's API changes. Why PSGD? One point is to save all those optimization relate parameters tuning efforts. I used step size 0.01 in all these demos.

###### To the COVID-19 lockup days
