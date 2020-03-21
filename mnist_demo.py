import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import utility as U

device = torch.device('cuda')

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([transforms.RandomCrop(26), transforms.ToTensor()])),    
                        batch_size=100, shuffle=True)
        
Ws_dr_encoder, Ws_dr_decoder = pickle.load(open('mnist_dr_codec.pkl', 'rb'))
Ws_encoder, Ws_decoder = pickle.load(open('mnist_codec.pkl', 'rb'))

data, _ = iter(train_loader).next()
x = data.reshape([100, -1]).to(device)

# original
y = x.cpu().data.numpy()
image1 = np.zeros((5*26, 20*26))
for i in range(5):
    for j in range(20):
        image1[26*i:26*i+26, 26*j:26*j+26] = y[20*i+j].reshape([26, 26])

# reconstructed through: original-->latent-->Gaussian-->latent-->reconstructed
z = U.encoding_fnn(x, Ws_dr_encoder)
v_gauss, nll = U.encoding_fnn(z, Ws_encoder, return_nll=True)
y = U.decoding_fnn(v_gauss, Ws_decoder)   
y = torch.sigmoid(U.decoding_fnn(y, Ws_dr_decoder)).cpu().data.numpy()
image2 = np.zeros((5*26, 20*26))
for i in range(5):
    for j in range(20):
        image2[26*i:26*i+26, 26*j:26*j+26] = y[20*i+j].reshape([26, 26])
        
# randomly sampled by Gaussian-->latent-->image
y = torch.sigmoid(U.decoding_fnn(U.decoding_fnn(torch.randn(100, 16, device=device), Ws_decoder), Ws_dr_decoder)).cpu().data.numpy()
image3 = np.zeros((5*26, 20*26))
for i in range(5):
    for j in range(20):
        image3[26*i:26*i+26, 26*j:26*j+26] = y[20*i+j].reshape([26, 26])
        
image = np.concatenate([image1, np.ones((26, 20*26)), image2, np.ones((26, 20*26)), image3])
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('mnist_demo.png', bbox_inches='tight')