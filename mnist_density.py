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
random_init = False

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([transforms.RandomCrop(26), transforms.ToTensor()])),    
                        batch_size=96, shuffle=True)

Ws_dr_encoder, Ws_dr_decoder = pickle.load(open('mnist_dr_codec.pkl', 'rb'))
if random_init:  
    Ws_encoder = U.fnn_init([16, 512, 512, 512, 16], device)
    Ws_decoder = U.fnn_init([16, 2048, 2048, 2048, 16], device)
else:
    Ws_encoder, Ws_decoder = pickle.load(open('mnist_codec.pkl', 'rb')) 

def main():   
    Ws = Ws_encoder + Ws_decoder    
    Qs = [[0.1*torch.eye(W.shape[0], device=device), 0.1*torch.eye(W.shape[1], device=device)] for W in Ws]
    step_size, num_epoch, grad_norm_clip_thr = 0.01, 1000, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
    thr, eta = 0.2, 10
    LOSS, NLL, MSE, num_iter, echo_every = [], [], [], 0, 10
    for epoch in range(num_epoch):
        for W in Ws:
            W.requires_grad = True
        for _, (data, target) in enumerate(train_loader):
            num_iter += 1
            num_samples = len(data)
            x = data.reshape([num_samples, -1]).to(device)
            z = U.encoding_fnn(x, Ws_dr_encoder)
            #print(U.median_dist(z))
            v_gauss, nll = U.encoding_fnn(z, Ws_encoder, return_nll=True)
            y = U.decoding_fnn(v_gauss, Ws_decoder)        
            mse = torch.sum((z - y)**2)/num_samples
            loss =  nll + eta*torch.clamp(mse**0.5 - thr, 0)**2
            
            LOSS.append(loss.item())
            NLL.append(nll.item())
            MSE.append(mse.item())
            
            Q_update_gap = max(math.floor(math.log10(num_iter)), 1)
            if num_iter % Q_update_gap == 0:
                grads = grad(loss, Ws, create_graph=True)     
                v = [torch.randn(W.shape, device=device) for W in Ws]
                Hv = grad(grads, Ws, v)      
                with torch.no_grad():
                    Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
            else:
                grads = grad(loss, Ws)
                
            with torch.no_grad():
                pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
                grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
                step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
                for i in range(len(Ws)):
                    Ws[i] -= step_adjust*step_size*pre_grads[i]
                    
                if num_iter % echo_every == 0:
                    y = torch.sigmoid( U.decoding_fnn( y, Ws_dr_decoder) )#sigmoid to make output in range [0, 1]
                    mse_image = torch.sum((x - y)**2)/num_samples
                    print('Total loss: {}; NLL: {}; MSE: {}; Image MSE: {}'.format(loss.item(), nll.item(), mse.item(), mse_image.item()))
            
        if (epoch+1) % round(0.01*num_epoch) == 0:    
            with torch.no_grad():
                # sampling the trained generative model   
                y = torch.sigmoid(U.decoding_fnn(U.decoding_fnn(torch.randn(100, 16, device=device), Ws_decoder), Ws_dr_decoder)).cpu().data.numpy()
                image = np.zeros((10*26, 10*26))
                for i in range(10):
                    for j in range(10):
                        image[26*i:26*i+26, 26*j:26*j+26] = y[10*i+j].reshape([26, 26])
                        
                plt.clf()
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.savefig(str(epoch+1)+'.png', bbox_inches='tight')
    
        for W in Ws:
            W.requires_grad = False 
        pickle.dump([Ws_encoder, Ws_decoder], open('mnist_codec.pkl', 'wb'))
                    
    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(LOSS)
    plt.subplot(3,1,2)
    plt.plot(NLL)
    plt.subplot(3,1,3)
    plt.semilogy(MSE)
    
if __name__ == '__main__':
    main()