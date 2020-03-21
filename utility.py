import torch
import math

def fnn_init(dims, device):
    """
    feedforward neural network initialization specified by dims on device
    """
    Ws = []
    for i in range(len(dims) - 1):
        W = torch.randn(dims[i] + 1, dims[i+1], device = device)/math.sqrt(dims[i] + 1)
        Ws.append(W)
        
    return Ws


def encoding_fnn(x, Ws, return_nll=False):
    """
    encode x to v with standard feedforward neural network
    assuming Gaussian distribution for NLL (negative log likelihood) calculation
    """
    batch_size, dim_input = x.shape
    
    if return_nll:
        nonl_derivatives = []
    for W in Ws[:-1]:
        x = torch.tanh( x @ W[:-1] + W[-1:] )
        if return_nll:
            nonl_derivatives.append(1.0 - x*x)
        
    v = x @ Ws[-1][:-1] + Ws[-1][-1:]
    
    if return_nll:
#        # no batching if having limited memory, but too slow
#        nll = 0.0
#        for n in range(batch_size):
#            J = Ws[-1][:-1]
#            for i in range(len(nonl_derivatives)-1, -1, -1):
#                J = (nonl_derivatives[i][n:n+1] * Ws[i][:-1]) @ J
#            nll -= torch.slogdet(J)[1] 
#            
#        nll = nll/batch_size + 0.5*torch.sum(v*v)/batch_size + 0.5*dim_input*math.log(2.0*math.pi)
            
        # batching if having enough memory
        J = (Ws[-1][:-1]).repeat(batch_size, 1, 1)
        for i in range(len(nonl_derivatives)-1, -1, -1):
            J = torch.bmm(nonl_derivatives[i].view(batch_size, 1, -1) * (Ws[i][:-1]).repeat(batch_size, 1, 1), J)
            
        nll = -sum(torch.slogdet(J)[1])/batch_size + 0.5*torch.sum(v*v)/batch_size + 0.5*dim_input*math.log(2.0*math.pi)
        return v, nll
    else:
        return v


def decoding_fnn(v, Ws):
    """
    decode v to y with standard feedforward neural network
    """
    for W in Ws[:-1]:
        v = torch.tanh( v @ W[:-1] + W[-1:] )  
        
    return v @ Ws[-1][:-1] + Ws[-1][-1:]


def median_dist(x):
    """
    find the median sample distance in tensor x
    """
    dist = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist.append(torch.norm(x[i] - x[j]))
    return torch.median(torch.stack(dist))