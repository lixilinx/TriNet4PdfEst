import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 
import utility as U

device = torch.device('cuda')
random_init = False#set to True to retrain the model

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([transforms.RandomCrop(26), transforms.ToTensor()])),    
                        batch_size=96, shuffle=True)

if random_init:
    Ws_dr_encoder = U.fnn_init([26**2, 2048, 2048, 2048, 16], device)#will use 26x26 crops
    Ws_dr_decoder = U.fnn_init([16, 2048, 2048, 2048, 26**2], device)
else:
    Ws_dr_encoder, Ws_dr_decoder = pickle.load(open('mnist_dr_codec.pkl', 'rb'))
   
def main():
    Ws = Ws_dr_encoder + Ws_dr_decoder        
    Qs = [[0.1*torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
    step_size, num_epoch, grad_norm_clip_thr = 0.01, 1000, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
    MSE, echo_every, num_iter = [], 10, 0
    for epoch in range(num_epoch):
        for W in Ws:
            W.requires_grad = True
        for _, (data, target) in enumerate(train_loader):
            num_iter += 1
            num_samples = len(data)
            x = data.reshape([num_samples, -1]).to(device)
            z = U.encoding_fnn(x, Ws_dr_encoder)
            y = torch.sigmoid(U.decoding_fnn(z, Ws_dr_decoder))#sigmoid to make output in range [0, 1]
            mse = torch.sum((x - y)**2)/num_samples    
            MSE.append(mse.item())
            
            # PSGD optimizer
            Q_update_gap = max(math.floor(math.log10(num_iter)), 1)
            if num_iter % Q_update_gap == 0:
                grads = grad(mse, Ws, create_graph=True)     
                v = [torch.randn(W.shape, device=device) for W in Ws]
                Hv = grad(grads, Ws, v)      
                with torch.no_grad():
                    Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
            else:
                grads = grad(mse, Ws)
                
            with torch.no_grad():
                pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
                grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
                step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
                for i in range(len(Ws)):
                    Ws[i] -= step_adjust*step_size*pre_grads[i]
                    
                if num_iter % echo_every == 0:
                    print('MSE: {}'.format(mse.item()))
            
        if (epoch+1) % round(0.01*num_epoch) == 0:    
            with torch.no_grad():
                # check reconstructed image   
                y = y.cpu().numpy()
                I = math.floor(num_samples**0.5)
                image = np.zeros((I*26, I*26))
                for i in range(I):
                    for j in range(I):
                        image[26*i:26*i+26, 26*j:26*j+26] = y[I*i+j].reshape([26, 26])
                        
                plt.clf()
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.savefig(str(epoch+1)+'.png', bbox_inches='tight')
                
        for W in Ws:
            W.requires_grad = False
        pickle.dump([Ws_dr_encoder, Ws_dr_decoder], open('mnist_dr_codec.pkl', 'wb'))
    
    plt.clf()
    plt.semilogy(MSE)
        
if __name__ == '__main__':
    main()