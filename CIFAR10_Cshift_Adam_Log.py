import pickle
import math
import random
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torchvision import datasets, transforms
import utility as U

device = torch.device('cuda')
alpha = 0.05

train_set, validation_set = torch.utils.data.random_split(datasets.CIFAR10('../data', train=True, download=True,           
                       transform=transforms.Compose([ 
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])), [45000, 5000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)

test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=64, shuffle=False)

def logit(x):
    x = alpha + (1 - 2*alpha)*x
    return torch.log(x/(1 - x))

def rand_cshift(im, max_shift=3):
    _, _, I, J = im.shape
    
    i = random.randint(-max_shift, max_shift)
    if i>0:
        im = torch.cat([im[:,:,i:], im[:,:,:i]], dim=2)
    elif i<0:
        im = torch.cat([im[:,:,I+i:], im[:,:,:I+i]], dim=2)
        
    j = random.randint(-max_shift, max_shift)
    if j>0:
        im = torch.cat([im[:,:,:,j:], im[:,:,:,:j]], dim=3)
    elif j<0:
        im = torch.cat([im[:,:,:,J+j:], im[:,:,:,:J+j]], dim=3)
        
    return im

def cifar_normalization():
    mu = torch.zeros(1, 3*32**2)
    R = torch.zeros(3*32**2, 3*32**2)
    num_samples = 0
    for _, (data, _) in enumerate(train_loader):
        for _ in range(20):
            num_samples += len(data)
            data = rand_cshift(data)
            x = 255/256*data.view(-1, 3*32**2)
            x = logit( x + torch.rand(x.shape)/256 )
            mu += torch.sum(x, dim=0, keepdim=True)
            R += x.t() @ x

    mu /= num_samples
    R = R/num_samples - mu.t() @ mu
    P = torch.inverse(torch.cholesky(R, upper=True))
    log_det_P = torch.sum(torch.log(torch.diag(P)))
    return mu, P, log_det_P

cifar_mu, cifar_P, log_det_P = cifar_normalization( )

def cifar_performance( data_loader ):
    with torch.no_grad():
        total_num_samples, Nll, bitspdim = 0, 0.0, 0.0
        for _, (data, _) in enumerate(data_loader):
            num_samples = len(data)
            total_num_samples += num_samples
            x = logit( (255*data.reshape([num_samples, -1]) + 0.5)/256 )
            bitspdim = bitspdim - torch.sum(torch.abs(x))/math.log(2) - 2*torch.sum(torch.log2(1 + torch.exp(-torch.abs(x))))
            x = (x - cifar_mu) @ cifar_P
            x = x.to(device)
            _, nll = U.MonoTriNet(x, Ws, nonlinearity='log')
            Nll += num_samples*nll
            
        Nll = Nll/total_num_samples - log_det_P
        bitspdim = Nll/3/32/32/math.log(2) - math.log2(1 - 2*alpha) + 8 + bitspdim/total_num_samples/3/32/32
        return Nll.item(), bitspdim.item()

Ws = U.MonoTriNetInit(3*32**2, [8, 8, 8, 8], device)
 
for W in Ws:
    W.requires_grad = True
m1 = [torch.zeros_like(W) for W in Ws]    
m2 = [torch.ones_like(W) for W in Ws]
step_size, num_epoch = 1e-4, 1000
Train_NLL, Val_NLL, epochs_no_progress, best_val_nll = [], [], 0, 1e38
for epoch in range(num_epoch):
    for _, (data, _) in enumerate(train_loader):
        num_samples = len(data)
        data = rand_cshift(data)
        x = 255/256*data.reshape([num_samples, -1])
        x = logit( x + torch.rand(x.shape)/256 )
        x = (x - cifar_mu) @ cifar_P
        x = x.to(device)
        _, nll = U.MonoTriNet(x, Ws, nonlinearity='log')
        #print(nll.item())
        Train_NLL.append(nll.item() - log_det_P.item())
        
        grads = grad(nll, Ws)        
        with torch.no_grad():
            m1 = [0.9*a + 0.1*b for (a, b) in zip(m1, grads)]
            m2 = [0.99*a + 0.01*b*b for (a, b) in zip(m2, grads)]
            for i in range(len(Ws)):
                Ws[i] -= step_size*m1[i]*torch.rsqrt(m2[i] + 1e-30)
    
    val_nll, _ =  cifar_performance(validation_loader)   
    Val_NLL.append(val_nll)
    best_val_nll = min(best_val_nll, val_nll)   
    print('Epoch: {}; Train NLL: {}; Validation NLL: {}; step_size: {}'.format(epoch+1, Train_NLL[-1], val_nll, step_size))
    if val_nll > best_val_nll:
        epochs_no_progress += 1
        if epochs_no_progress > 5:
            step_size *= 0.1
            if step_size < 1e-6:
                break
            epochs_no_progress = 0
    else:
        epochs_no_progress = 0
        test_nll, test_bitspdim = cifar_performance(test_loader)
        print('          Test NLL: {}; Test Bits per dim: {}'.format(test_nll, test_bitspdim))
        
plt.plot(Train_NLL, 'k')
print(cifar_performance(validation_loader))
print(cifar_performance(train_loader))
for W in Ws:
    W.requires_grad = False 
cifar_mu_P = [cifar_mu, cifar_P]
pickle.dump([cifar_mu_P, Ws], open('cifar_bij_tri_net_mdl.pkl', 'wb'))