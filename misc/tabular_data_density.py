import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import utility as U
import datasets

name = 'power'
if name == 'bsds300':
    step_size = 5e-5
elif name == 'power':
    step_size = 1e-3
elif name == 'gas':
    step_size = 1e-3
elif name == 'hepmass':
    step_size = 2e-4
elif name == 'miniboone':
    step_size = 1e-4
else:
    raise ValueError('Unknown dataset')
    
device = torch.device('cuda')

def load_data(name):
    if name == 'bsds300':
        return datasets.BSDS300()
    elif name == 'power':
        return datasets.POWER()
    elif name == 'gas':
        return datasets.GAS()
    elif name == 'hepmass':
        return datasets.HEPMASS()
    elif name == 'miniboone':
        return datasets.MINIBOONE()
    else:
        raise ValueError('Unknown dataset')
        
data = load_data(name)
batch_size = data.trn.N//1000

def normalization():
    x, N = torch.tensor(data.trn.x), data.trn.N
    mu = torch.sum(x, dim=0, keepdim=True)/N
    R = x.t() @ x / N - mu.t() @ mu
    P = torch.inverse(torch.cholesky(R, upper=True))
    log_det_P = torch.sum(torch.log(torch.diag(P)))
    return mu, P, log_det_P

mu, P, log_det_P = normalization( )

def performance( x ):
    x = torch.tensor(x)
    with torch.no_grad():
        i, N, Nll = 0, len(x), 0.0
        while i < N:
            num_samples = min(batch_size, N - i)
            y = (x[i:i+num_samples] - mu) @ P
            y = y.to(device)
            _, nll = U.MonoTriNet(y, Ws)
            Nll += num_samples*nll
            i += num_samples
            
        Nll = Nll/N - log_det_P
        return Nll.item()

Ws = U.MonoTriNetInit(data.n_dims, [512, 512, 512, 512, 512, 512, 512, 512, 512, 512], device)

for W in Ws:
    W.requires_grad = True
m1 = [torch.zeros_like(W) for W in Ws]    
m2 = [torch.ones_like(W) for W in Ws]
beta, num_epoch = 0.1, 1000
Train_NLL, Val_NLL, epochs_no_progress, best_val_nll = [], [], 0, 1e38
trn_x, trn_N = torch.tensor(data.trn.x), data.trn.N
for epoch in range(num_epoch):
    i, trn_x = 0, trn_x[torch.randperm(trn_N)]
    while i+batch_size <= trn_N:
        y = (trn_x[i:i+batch_size] - mu) @ P
        y = y.to(device)
        _, nll = U.MonoTriNet(y, Ws)
        #print(nll.item())
        Train_NLL.append(nll.item() - log_det_P.item())
        
        grads = grad(nll, Ws)        
        with torch.no_grad():
            m1 = [(1.0 - beta)*a + beta*b for (a, b) in zip(m1, grads)]
            m2 = [(1.0 - 0.1*beta)*a + 0.1*beta*b*b for (a, b) in zip(m2, grads)]
            for j in range(len(Ws)):
                Ws[j] -= step_size*m1[j]*torch.rsqrt(m2[j] + 1e-30)
                
        i += batch_size
    
    val_nll =  performance(data.val.x)   
    Val_NLL.append(val_nll)
    best_val_nll = min(best_val_nll, val_nll)   
    print('Epoch: {}; Train NLL: {}; Validation NLL: {}; step size: {}'.format(epoch+1, Train_NLL[-1], val_nll, step_size))
    if val_nll > best_val_nll:
        epochs_no_progress += 1
        if epochs_no_progress > 20:
            step_size /= 2.0
            if step_size < 1e-6:
                break
            epochs_no_progress = 0
    else:
        epochs_no_progress = 0
        print('          Test NLL: {}'.format(performance(data.tst.x)))
        
plt.plot(Train_NLL, 'k')
print(performance(data.trn.x))