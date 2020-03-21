import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import utility as U
import celeba_dim_reduction as C

Us, Vs = pickle.load(open('celeba_dr_codec.pkl', 'rb'))#dimension reduction model
Ws_encoder, Ws_decoder = pickle.load(open('celeba_codec.pkl', 'rb'))#density model 

I, J = 4, 16
train_loader = torch.utils.data.DataLoader(C.celeba, batch_size=I*J, shuffle=True)
data, _ = iter(train_loader).next()

device = torch.device('cuda')
x = data.to(device)           

# original 
y = np.transpose(x.cpu().data.numpy(), [0,2,3,1])
image1 = np.zeros((I*64, J*64, 3))
for i in range(I):
    for j in range(J):
        image1[64*i:64*i+64, 64*j:64*j+64] = y[i*J+j]

# reconstructed through: original-->latent-->Gaussian-->latent-->reconstructed
z = C.conv_coder(x, Us)
v_gauss, nll = U.encoding_fnn(z, Ws_encoder, return_nll=True)
y = U.decoding_fnn(v_gauss, Ws_decoder) 
y = C.conv_decoder(y, Vs)
y = np.transpose(y.cpu().data.numpy(), [0,2,3,1])
image2 = np.zeros((I*64, J*64, 3))
for i in range(I):
    for j in range(J):
        image2[64*i:64*i+64, 64*j:64*j+64] = y[i*J+j]

# randomly sampled by Gaussian-->latent-->image
y = C.conv_decoder(U.decoding_fnn(torch.randn(I*J, 128, device=device), Ws_decoder), Vs).cpu().data.numpy()
y = np.transpose(y, [0,2,3,1])
image3 = np.zeros((I*64, J*64, 3))
for i in range(I):
    for j in range(J):
        image3[64*i:64*i+64, 64*j:64*j+64] = y[i*J+j]
        
image = np.concatenate([image1, np.ones((64, J*64, 3)), image2, np.ones((64, J*64, 3)), image3])
plt.imshow(image)
plt.axis('off')
plt.savefig('celeba_demo.png', bbox_inches='tight')