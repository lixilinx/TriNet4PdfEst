import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 

device = torch.device('cuda')
random_init = False#set to True to retrain the model

transform = transforms.Compose([transforms.CenterCrop(155), transforms.Resize(67), transforms.RandomCrop(64), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
celeba = datasets.ImageFolder('C:/Users/Admin/Downloads/celeba', transform)#set to the correct folder on your computer
train_loader = torch.utils.data.DataLoader(celeba, batch_size=128, shuffle=True)

def conv_coder(x, Us):
    U1,U2,U3,U4,U5,U6,U7 = Us
    x = torch.tanh(F.conv2d(x, U1[:,:-1].view(128,3,4,4), bias=U1[:,-1], stride=2, padding=1))
    x = torch.tanh(F.conv2d(x, U2[:,:-1].view(128,128,4,4), bias=U2[:,-1], stride=2, padding=1))
    x = torch.tanh(F.conv2d(x, U3[:,:-1].view(128,128,4,4), bias=U3[:,-1], stride=2, padding=1))
    x = torch.tanh(F.conv2d(x, U4[:,:-1].view(128,128,4,4), bias=U4[:,-1], stride=2, padding=1))
    x = torch.tanh(x.view(len(x), -1)@U5[:-1] + U5[-1:])
    x = torch.tanh(x@U6[:-1] + U6[-1:])
    x = x@U7[:-1] + U7[-1:]
    return x

def conv_decoder(x, Vs):
    V7,V6,V5,V4,V3,V2,V1 = Vs
    x = torch.tanh(x@V7[:-1] + V7[-1:])
    x = torch.tanh(x@V6[:-1] + V6[-1:])
    x = torch.tanh(x@V5[:-1] + V5[-1:]).view(-1, 128, 4, 4)
    x = torch.tanh(F.conv2d(F.interpolate(x, scale_factor=2), V4[:,:-1].view(128,128,3,3), bias=V4[:,-1], padding=1))
    x = torch.tanh(F.conv2d(F.interpolate(x, scale_factor=2), V3[:,:-1].view(128,128,3,3), bias=V3[:,-1], padding=1))
    x = torch.tanh(F.conv2d(F.interpolate(x, scale_factor=2), V2[:,:-1].view(128,128,3,3), bias=V2[:,-1], padding=1))
    x = F.conv2d(F.interpolate(x, scale_factor=2), V1[:,:-1].view(3,128,3,3), bias=V1[:,-1], padding=1)
    return torch.sigmoid(x)

if random_init:
    U1 = torch.randn(128, 3*4*4+1, device=device)/math.sqrt(3*4*4+1)
    U2 = torch.randn(128, 128*4*4+1, device=device)/math.sqrt(128*4*4+1)
    U3 = torch.randn(128, 128*4*4+1, device=device)/math.sqrt(128*4*4+1)
    U4 = torch.randn(128, 128*4*4+1, device=device)/math.sqrt(128*4*4+1)
    U5 = torch.randn(4*4*128+1, 2048, device=device)/math.sqrt(4*4*128+1)
    U6 = torch.randn(2048+1, 2048, device=device)/math.sqrt(2048+1)
    U7 = torch.randn(2048+1, 128, device=device)/math.sqrt(2048+1)
    Us = [U1,U2,U3,U4,U5,U6,U7]
    
    V7 = torch.randn(128+1, 2048, device=device)/math.sqrt(128+1)
    V6 = torch.randn(2048+1, 2048, device=device)/math.sqrt(2048+1)
    V5 = torch.randn(2048+1, 2048, device=device)/math.sqrt(2048+1)
    V4 = torch.randn(128, 128*3*3+1, device=device)/math.sqrt(128*3*3+1)
    V3 = torch.randn(128, 128*3*3+1, device=device)/math.sqrt(128*3*3+1)
    V2 = torch.randn(128, 128*3*3+1, device=device)/math.sqrt(128*3*3+1)
    V1 = torch.randn(3, 128*3*3+1, device=device)/math.sqrt(128*3*3+1)
    Vs = [V7,V6,V5,V4,V3,V2,V1]
else:
    Us, Vs = pickle.load(open('celeba_dr_codec.pkl', 'rb'))
    
def main():
    Ws = Us + Vs  
    # PSGD optimizer. Second order method. I have no need to tune the optimizer    
    Qs = [[0.1*torch.eye(W.shape[0], device=device), 0.1*torch.eye(W.shape[1], device=device)] for W in Ws]
    step_size, num_epoch, grad_norm_clip_thr = 0.01, 100, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
    MSE, echo_every, num_iter = [], 10, 0
    for epoch in range(num_epoch):
        for W in Ws:
            W.requires_grad = True
        for _, (data, target) in enumerate(train_loader):
            num_iter += 1
            
            x = data.to(device)           
            num_samples = len(data)
            z = conv_coder(x, Us)
            y = conv_decoder(z, Vs)
            mse = torch.sum((x - y)**2)/num_samples    
            MSE.append(mse.item())
            
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
               
        with torch.no_grad():
            # check reconstructed images   
            y = np.transpose(y.cpu().numpy(), [0,2,3,1])
            I = math.floor(num_samples**0.5)
            image = np.zeros((I*64, I*64, 3))
            for i in range(I):
                for j in range(I):
                    image[64*i:64*i+64, 64*j:64*j+64] = y[I*i+j]
                    
            plt.clf()
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(str(epoch+1)+'.png', bbox_inches='tight')
            
        for W in Ws:
            W.requires_grad = False
        pickle.dump([Us, Vs], open('celeba_dr_codec.pkl', 'wb'))
    
    plt.clf()
    plt.semilogy(MSE)
    
if __name__ == '__main__':
    main()