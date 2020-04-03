import pickle
import torch
from torch.autograd import grad
from torchvision import datasets, transforms
import utility as U

device = torch.device('cuda')
random_init = True # set to True to start from random initial guess

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,           
                       transform=transforms.Compose([ 
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(    
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)

def cifar_normalization(sigma2=0.01):
    mu = torch.zeros(1, 3*32**2)
    R = torch.zeros(3*32**2, 3*32**2)
    num_samples = 0
    for _, (data, _) in enumerate(train_loader):
        for flip in range(2):
            num_samples += len(data)
            x = 2*data.view(-1, 3*32**2) - 1
            mu += torch.sum(x, dim=0, keepdim=True)
            R += x.t() @ x
            data = data.flip(3)
    
    mu /= num_samples
    R = R/num_samples - mu.t() @ mu + sigma2*torch.eye(3*32**2)
    P = torch.inverse(torch.cholesky(R, upper=True))
    return mu, P

def cifar_test( ):
    with torch.no_grad():
        total_num_samples, Nll = 0, 0.0
        for _, (data, _) in enumerate(test_loader):
            num_samples = len(data)
            total_num_samples += num_samples
            x = data.reshape([num_samples, -1])
            x = 2*x - 1 
            x = (x - cifar_mu) @ cifar_P
            x = x.to(device)
            _, nll = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)
            nll -= torch.sum(torch.log(torch.abs(torch.diag(cifar_P)))).item()
            Nll += num_samples*nll
            
        return Nll/total_num_samples

cifar_mu, cifar_P = cifar_normalization( )

if random_init:  
    Wis, Mis, Wos, Mos = U.mono_tri_fnn_init(3*32**2, [2, 4, 8], device)
else:
    Wis, Mis, Wos, Mos = pickle.load(open('cifar_mono_tri_net_nll.pkl', 'rb')) 
    
Ws = Wis + Wos
var_grads = [torch.ones_like(W) for W in Ws]
step_size, num_epoch, grad_norm_clip_thr = 0.01, 500, 0.01*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5
NLL, num_iter, echo_every, TestNLL = [], 0, 10, 1e38
for epoch in range(num_epoch):
    for W in Ws:
        W.requires_grad = True
    for _, (data, _) in enumerate(train_loader):
        num_iter += 1
        num_samples = len(data)
        x = data.reshape([num_samples, -1])
        x = x + (torch.rand(x.shape)/128 - 1/256)
        x = 2*x - 1
        x = (x - cifar_mu) @ cifar_P
        x = x.to(device)
        _, nll = U.encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos)
        nll -= torch.sum(torch.log(torch.abs(torch.diag(cifar_P)))).item()
        NLL.append(nll.item())     
        grads = grad(nll, Ws)
        with torch.no_grad():
            var_grads = [0.98*v + 0.02*g*g for (v, g) in zip(var_grads, grads)]
            pre_grads = [g*torch.rsqrt(v + 1e-10) for (v, g) in zip(var_grads, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= step_adjust*step_size*pre_grads[i]
                
            if num_iter % echo_every == 0:
                print('Epoch: {}; Train NLL: {}; Test NLL: {}'.format(epoch, nll.item(), TestNLL))
                
    test_nll = cifar_test().item()
    if test_nll < TestNLL:
        TestNLL = test_nll
        for W in Ws:
            W.requires_grad = False 
        pickle.dump([Wis, Mis, Wos, Mos], open('cifar_mono_tri_net_nll.pkl', 'wb'))