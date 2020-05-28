import math
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
    
Ws_encoder, Ms, blk_diag_locs = U.tri_fnn_init(2, [1, 20, 20, 20, 1], device)
Ws_decoder = U.fnn_init([2, 128, 128, 128, 2], device)
Ws = Ws_encoder + Ws_decoder
for W in Ws:
    W.requires_grad = True
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
num_samples, step_size, num_iter, grad_norm_clip_thr = 64, 0.01, 100000, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
thr, eta = 1.2/100, 100
LOSS, NLL, MSE, echo_every = [], [], [], 100
for bi in range(num_iter):
    x = ring_sampling(num_samples).to(device)
    v_gauss, nll = U.encoding_tri_fnn(x, Ws_encoder, Ms, blk_diag_locs)
    y = U.decoding_fnn(v_gauss, Ws_decoder)
    mse = torch.sum((x - y)**2)/num_samples
    loss =  nll + eta*torch.clamp(mse**0.5 - thr, 0)**2
    NLL.append(nll.item())
    MSE.append(mse.item())
    LOSS.append(loss.item())
    
    # PSGD optimizer. No need to tune
    Q_update_gap = max(math.floor(math.log10(bi + 1)), 1)
    if bi % Q_update_gap == 0:
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
            
        if (bi+1) % echo_every == 0:
            print('Total loss: {}; NLL: {}; MSE: {}'.format(loss.item(), nll.item(), mse.item()))
        if (bi+1) % round(0.1*num_iter) == 0:
            # Sampling from the generative model. Should look like a ring
            y = U.decoding_fnn(torch.randn(128, 2, device=device), Ws_decoder).data.to('cpu').numpy()
            plt.clf()
            plt.scatter(y[:,0], y[:,1])
            plt.savefig(str(bi+1) + '.png')
            
plt.clf()
y = U.decoding_fnn(torch.randn(128, 2, device=device), Ws_decoder).data.to('cpu').numpy()
plt.subplot(2,2,1)
plt.scatter(y[:,0], y[:,1])
plt.title('Randomly generated samples')
plt.subplot(2,2,2)
plt.plot(LOSS)
plt.title('Total loss')
plt.subplot(2,2,3)
plt.plot(NLL)
plt.title('Relative entropy estimation')
plt.subplot(2,2,4)
plt.semilogy(MSE)
plt.title('Reconstruction error')

import pickle    
pickle.dump([Ws_encoder, Ms, blk_diag_locs, Ws_decoder], open('toy_data_generation_demo.pkl', 'wb'))  
