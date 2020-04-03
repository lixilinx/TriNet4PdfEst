import pickle
import math
import torch
from torch.autograd import grad
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 
import utility as U

device = torch.device('cuda')
random_init = True # set to True to start from random initial guess
additive_noise_sigma = 0.1 # the amount of additive Gaussian noise for training 

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=100, shuffle=True)

def logit(x, lbd=1e-6):
    x = lbd + (1-2*lbd)*x
    return torch.log(x/(1 - x))

def mnist_normalization(sigma2=1.0):
    mu = torch.zeros(1, 28**2)
    R = torch.zeros(28**2, 28**2)
    num_samples = 0
    for _, (data, _) in enumerate(train_loader):
        num_samples += len(data)
        x = logit( data.view(-1, 28**2) )
        mu += torch.sum(x, dim=0, keepdim=True)
        R += x.t() @ x

    mu /= num_samples
    R = R/num_samples - mu.t() @ mu + max(additive_noise_sigma**2, sigma2)*torch.eye(28**2)
    P = torch.inverse(torch.cholesky(R, upper=True))
    return mu, P

def mnist_test( ):
    with torch.no_grad():
        total_num_samples, Nll = 0, 0.0
        for _, (data, _) in enumerate(test_loader):
            num_samples = len(data)
            total_num_samples += num_samples
            x = data.reshape([num_samples, -1])
            x = logit(x)
            x = (x - mnist_mu) @ mnist_P
            x = x.to(device)
            _, nll = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)
            nll -= torch.sum(torch.log(torch.abs(torch.diag(mnist_P)))).item()
            Nll += num_samples*nll
            
        return Nll/total_num_samples

mnist_mu, mnist_P = mnist_normalization( )

if random_init:  
    Wis, Mis, Wos, Mos = U.mono_tri_fnn_init(28**2, [16, 16, 16], device)
else:
    Wis, Mis, Wos, Mos = pickle.load(open('mnist_mono_tri_net_nll.pkl', 'rb')) 
    
Ws = Wis + Wos
    
Qs = [[torch.cat([torch.ones((1, W.shape[0]), device=device), torch.zeros((1, W.shape[0]), device=device)]),
       torch.ones((1, W.shape[1]), device=device)] for W in Ws]
step_size, num_epoch, grad_norm_clip_thr = 0.01, 1000, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
NLL, num_iter, echo_every, TestNLL = [], 0, 10, 1e38
for epoch in range(num_epoch):
    for W in Ws:
        W.requires_grad = True
    for _, (data, _) in enumerate(train_loader):
        num_iter += 1
        num_samples = len(data)
        x = data.reshape([num_samples, -1])
        x = logit(x) + additive_noise_sigma*torch.randn(x.shape)
        x = (x - mnist_mu) @ mnist_P
        x = x.to(device)
        _, nll = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)
        nll -= torch.sum(torch.log(torch.abs(torch.diag(mnist_P)))).item()
        NLL.append(nll.item())
        
        Q_update_gap = max(math.floor(math.log10(num_iter)), 1)
        if num_iter % Q_update_gap == 0:
            grads = grad(nll, Ws, create_graph=True)     
            v = [torch.randn(W.shape, device=device) for W in Ws]
            Hv = grad(grads, Ws, v)      
            with torch.no_grad():
                Qs = [psgd.update_precond_scan(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        else:
            grads = grad(nll, Ws)

        with torch.no_grad():
            pre_grads = [psgd.precond_grad_scan(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= step_adjust*step_size*pre_grads[i]
                
            if num_iter % echo_every == 0:
                print('Epoch: {}; Train NLL: {}; Test NLL: {}'.format(epoch, nll.item(), TestNLL))
                
    test_nll = mnist_test().item()
    if test_nll < TestNLL:
        TestNLL = test_nll
        for W in Ws:
            W.requires_grad = False 
        pickle.dump([Wis, Mis, Wos, Mos], open('mnist_mono_tri_net_nll.pkl', 'wb'))