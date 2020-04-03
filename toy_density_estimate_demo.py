import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import preconditioned_stochastic_gradient_descent as psgd 
import utility as U

device = torch.device('cpu')

def ring_sampling(num_samples):
    # its relative entropy is log(0.5*pi) = 0.45
    x = []
    while len(x)<num_samples:
        a = 2*torch.rand(2) - 1
        if torch.norm(a)<1 and torch.norm(a)>math.sqrt(0.5):
            x.append(a)    
    return torch.stack(x)
    
Wis, Mis, Wos, Mos = U.mono_tri_fnn_init(2, [50, 50], device)
Ws = Wis + Wos
for W in Ws:
    W.requires_grad = True
num_para = sum([np.prod(W.size()) for W in Ws])
Q = torch.eye(int(num_para)).to(device)
num_samples, step_size, num_iter, grad_norm_clip_thr = 128, 0.01, 50000, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
NLL, echo_every = [], 100
for bi in range(num_iter):
    x = ring_sampling(num_samples).to(device)
    _, nll = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)
    NLL.append(nll.item())
    
    # PSGD optimizer. 
    Q_update_gap = max(math.floor(math.log10(bi + 1)), 1)
    if num_iter % Q_update_gap == 0:
        grads = grad(nll, Ws, create_graph=True)     
        v = [torch.randn(W.shape, device=device) for W in Ws]
        Hv = grad(grads, Ws, v)      
        with torch.no_grad():
            Q = psgd.update_precond_dense(Q, v, Hv)
    else:
        grads = grad(nll, Ws)
        
    with torch.no_grad():
        pre_grads = psgd.precond_grad_dense(Q, grads)
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= step_adjust*step_size*pre_grads[i]
            
        if (bi+1) % echo_every == 0:
            print('NLL: {}'.format(nll.item()))
  
with torch.no_grad():
    im_size = 64
    im = np.zeros((im_size, im_size))
    for i in range(im_size):
        for j in range(im_size):
            x = torch.tensor([[i+0.5, j+0.5]]).to(device)
            x = 2.2*x/im_size - 1.1
            im[i, j] = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)[1].item()
    im = np.exp(-im)
    plt.imshow(im)
    
import pickle    
pickle.dump([Wis, Mis, Wos, Mos], open('toy_density_estimate_demo.pkl', 'wb'))     