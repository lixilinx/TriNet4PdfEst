import numpy as np
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


def tri_fnn_init(N, Bs, device):
    """
    (upper) triangular feedforward neural network initialization
    N:  dim of input/output
    Bs: for example, list [1,2,3,1] defines a three-layer network with block sizes 1x2, 2x3 and 3x1 at each layer
    device: cpu or cuda
    return matrix coefficients, their masks and block diagonals' locations for advanced indexing
    """
    Ws, Ms, blk_diag_locs = [], [], []
    for i in range(len(Bs) - 1):
        M = np.kron(np.triu(np.ones([N, N])), np.ones([Bs[i], Bs[i+1]]))
        M = np.concatenate([M, np.ones([1, N*Bs[i+1]])])
        M = torch.FloatTensor(M).to(device)
        W = M*torch.randn(N*Bs[i]+1, N*Bs[i+1]).to(device)/torch.sqrt(torch.sum(M, dim=0, keepdim=True))
        Ws.append(W)
        Ms.append(M)
        
        loc0, loc1 = [], []#locations for advance indexing
        for n in range(N):
            for c in range(Bs[i]):
                for r in range(Bs[i+1]):
                    loc0.append(n*Bs[i] + c)
                    loc1.append(n*Bs[i+1] + r)
        loc0, loc1 = torch.tensor(loc0, dtype=torch.int64), torch.tensor(loc1, dtype=torch.int64)
        blk_diag_locs.append([loc0, loc1])
        
    return Ws, Ms, blk_diag_locs


def encoding_tri_fnn(x, Ws, Ms, blk_diag_locs):
    """
    encoding x with (upper) triangular feedforward neural network 
    x: input
    Ws and Ms: neural network coefficients and their masks
    blk_diag_locs: block diagonals' locations
    return encoded results and negative-log-likelihood
    """
    batch_size, N = x.shape
    j_diag = torch.ones(batch_size, N, 1, 1, device = x.device)#diagonals of Jacobian 
    for i in range(len(Ws)):
        Bl, Br = Ws[i].shape
        Bl, Br = (Bl - 1)//N, Br//N#the block size is (Bl, Br)
        W = Ms[i] * Ws[i]
        diag_matrices = Ws[i][blk_diag_locs[i][0], blk_diag_locs[i][1]].view(N, Bl, Br)
        if i < len(Ws) - 1:
            x = torch.tanh( x @ W[:-1] + W[-1:] )
            derivative = 1.0 - x*x#d(tanh) = 1 - tanh^2
            current_j_diag = diag_matrices[None,:,:,:] * derivative.view(batch_size, N, 1, Br)
            j_diag = torch.matmul(j_diag, current_j_diag)
        else:
            x = x @ W[:-1] + W[-1:]#the last layer has no tanh
            j_diag = torch.matmul(j_diag, diag_matrices[None,:,:,:])
            
    nll = -torch.sum(torch.log(torch.abs(j_diag) + 1e-15))/batch_size + 0.5*torch.sum(x*x)/batch_size + 0.5*N*math.log(2.0*math.pi)
    return x, nll

"""
Only the code of monotonic triangular network part is maintained after Apr. 10, 2020
"""

def mono_tri_fnn_unit_init(N, B, device):
    """
    monotonic triangular feedforward neural network unit initialization
    N: dim of input/output
    B: defines block sizes 1xB (input matrix) and Bx1 (output matrix)
    device: cpu or cuda
    """
    # Input matrix and its mask 
    Mi = np.kron(np.triu(np.ones([N, N])), np.ones([1, B]))
    Mi = np.concatenate([Mi, np.ones([1, B*N])])
    Mi = torch.FloatTensor(Mi)
    Wi = Mi*torch.randn(N+1, B*N)/torch.sqrt(torch.sum(Mi, dim=0, keepdim=True))
    
    # Output matrix and its mask
    Mo = np.kron(np.triu(np.ones([N, N])), np.ones([B, 1]))
    Mo = np.concatenate([Mo, np.ones([1, N])])
    Mo = torch.FloatTensor(Mo)
    Wo = Mo*torch.randn(B*N+1, N)/torch.sqrt(torch.sum(Mo, dim=0, keepdim=True))
    
    # will convert diagonals to log(1 + exp(.)). So here reverse them
    arng = torch.arange(B*N, dtype=torch.int64)
    Wi[arng//B, arng] = torch.log(torch.expm1(torch.abs( Wi[arng//B, arng] )) + 1e-15)
    Wo[arng, arng//B] = torch.log(torch.expm1(torch.abs( Wo[arng, arng//B] )) + 1e-15)
        
    return Wi.to(device), (Mi>0).to(device), Wo.to(device), (Mo>0).to(device)


def encoding_mono_tri_fnn_unit(x, Wi, Mi, Wo, Mo, nonlinearity):
    """
    encoding x with a monotonic triangular feedforward neural network unit
    """    
    def phi(x):
        if nonlinearity == 'tanh':#this is the default nonlinearity
            y = torch.tanh(x)
            d = 1 - y*y
        else:                   # example of a self-defined nonlinearity
            d = torch.rsqrt(x*x + 1)
            y = x*d
            d = d*d*d
        
        return y, d
        
    N = Wi.shape[0] - 1#dimension of input
    B = Wi.shape[1]//N#block size
    Wi = Mi.float() * Wi#apply mask
    Wo = Mo.float() * Wo
    
    # extract the block diagonal parts
    arng = torch.arange(B*N, dtype=torch.int64)
    wi_diag = Wi[arng//B, arng]#torch.cat([Wi[n, n*B:n*B+B] for n in range(N)])
    wo_diag = Wo[arng, arng//B]#torch.cat([Wo[n*B:n*B+B, n] for n in range(N)])
    
    # additive modifications on block diagonals to transfer to log(1 + exp(.))
    delta_wi_diag = torch.logsumexp(torch.stack([wi_diag, torch.zeros_like(wi_diag)]), dim=0) - wi_diag
    delta_wo_diag = torch.logsumexp(torch.stack([wo_diag, torch.zeros_like(wo_diag)]), dim=0) - wo_diag
    
    # spread
    x, derivative = phi( x@Wi[:-1] + Wi[-1:] + x.repeat_interleave(B, 1)*delta_wi_diag[None, :] )
    j_diag = (wi_diag + delta_wi_diag)[None, :] * derivative * (wo_diag + delta_wo_diag)[None, :]
    j_diag = torch.sum(j_diag.view(-1, N, B), dim=2)#these are the diagonals of the Jacobian
    
    # collect
    x = x@Wo[:-1] + Wo[-1:] + torch.sum((x*delta_wo_diag[None, :]).view(-1, N, B), dim=2)

    #return the encoded results, and SumLogDet(Jacobian)
    return x, torch.sum(torch.log(j_diag + 1e-15))
    
 
def mono_tri_fnn_init(N, Bs, device):  
    """
    monotonic triangular feedforward neural network initialization
    N: dim of input/output
    Bs: each B in list Bs defines block sizes 1xB (input matrix) and Bx1 (output matrix)
    device: cpu or cuda
    """
    Wis, Mis, Wos, Mos = [], [], [], []
    for B in Bs:
        Wi, Mi, Wo, Mo = mono_tri_fnn_unit_init(N, B, device)
        Wis.append(Wi)
        Mis.append(Mi)
        Wos.append(Wo)
        Mos.append(Mo)
        
    return Wis, Mis, Wos, Mos


def encoding_mono_tri_fnn(x, Wis, Mis, Wos, Mos, nonlinearity='tanh'):
    """
    encoding x with monotonic triangular feedforward neural network
    default nonlinearity is tanh; otherwise, a self-defined one, see the code
    """
    batch_size, dim_input = x.shape
    
    nll = 0.0
    for i in range(len(Wis)):
        x, sumlogdetJ = encoding_mono_tri_fnn_unit(x, Wis[i], Mis[i], Wos[i], Mos[i], nonlinearity)
        nll = nll - sumlogdetJ
        x = torch.flip(x, [1])#reverse the order of input to balance the network
        
    nll = nll/batch_size + 0.5*torch.sum(x*x)/batch_size + 0.5*dim_input*math.log(2.0*math.pi)
    return x, nll









"""economy size monotonic triangular network"""
def MonoTriNetUnitInit(N, B, device):
    """
    monotonic triangular network unit initialization
    N: dimension of input and output
    B: block size
    device: cpu or cuda
    """
    # Input matrix and its mask 
    Mi = np.kron(np.triu(np.ones([N, N])), np.ones([1, B]))
    Mi = np.concatenate([Mi, np.ones([1, B*N])])
    Mi = torch.FloatTensor(Mi)
    Wi = Mi*torch.randn(N+1, B*N)/torch.sqrt(torch.sum(Mi, dim=0, keepdim=True))
    
    # Output matrix and its mask
    Mo = np.kron(np.triu(np.ones([N, N])), np.ones([B, 1]))
    Mo = np.concatenate([Mo, np.ones([1, N])])
    Mo = torch.FloatTensor(Mo)
    Wo = Mo*torch.randn(B*N+1, N)/torch.sqrt(torch.sum(Mo, dim=0, keepdim=True))
    
    # will convert diagonals to log(1 + exp(.)). So here reverse them
    arng = torch.arange(B*N, dtype=torch.int64)
    Wi[arng//B, arng] = torch.log(torch.expm1(torch.abs( Wi[arng//B, arng] )) + 1e-15)
    Wo[arng, arng//B] = torch.log(torch.expm1(torch.abs( Wo[arng, arng//B] )) + 1e-15)
    
    # packed all into one matrix!!!
    whole_matrix = Wi # input matrix
    whole_matrix[:-1] += (1 - Mi[:-1])*( Wo[:-1].t() ) # then the upper part (not including the diagonals) of Wo[:-1]
    whole_matrix = torch.cat([whole_matrix, Wo[arng, arng//B][None, :]]) # then the diagonals of Wo
    whole_matrix = torch.cat([whole_matrix, torch.zeros(1, B*N)])# reserve some space
    whole_matrix[-1, :N] = Wo[-1] # then the bias of Wo. B*N - N floats are not used, that is fine  
        
    return whole_matrix.to(device)


def MonoTriNetUnit(x, whole_matrix, nonlinearity):
    """
    monotonic triangular network unit
    x: input
    whole_matrix: coefficients
    nonlinearity: tanh or some self-defined ones
    """    
    def phi(x):
        if nonlinearity == 'rsqrt':
            d = torch.rsqrt(x*x + 1)
            y = x*d
            d = d*d*d
        elif nonlinearity == 'log':
            d = 1/(1 + torch.abs(x))
            y = -torch.sign(x)*torch.log(d)
        else:
            y = torch.tanh(x)
            d = 1 - y*y
        
        return y, d
    
    # unpack the whole_matrix into input matrix, input bias, output matrix (excluding diagonals), output diagonals, output bias
    N = whole_matrix.shape[0] - 3
    B = whole_matrix.shape[1]//N    
    device = whole_matrix.device
    mask = (torch.triu(torch.ones(N, N, device=device))).repeat_interleave(B, 1)#repeat dim1 B times
    # input_matrix = mask * whole_matrix[:N]
    input_bias = whole_matrix[-3]
    #output_matrix = ( (1 - mask) * whole_matrix[:N] ).t()#excluding the diagonals
    output_diag = whole_matrix[-2]
    output_bias = whole_matrix[-1][:N]
    
    # extract the block diagonal parts
    arng = torch.arange(B*N, dtype=torch.int64, device=device)
    wi_diag = whole_matrix[arng//B, arng] #wo_diag is already there
    
    # additive modifications on block diagonals to transfer to log(1 + exp(.))
    # useful relationships: log(1+exp(x)) = x - log sigmoid(x) = -log sigmoid(-x)
    delta_wi_diag = -torch.log(torch.sigmoid( wi_diag )) #torch.logsumexp(torch.stack([wi_diag, torch.zeros_like(wi_diag)]), dim=0) - wi_diag
    delta_wo_diag = -torch.log(torch.sigmoid(-output_diag )) #torch.logsumexp(torch.stack([output_diag, torch.zeros_like(output_diag)]), dim=0)
    
    # spread
    x, derivative = phi( x @ (mask * whole_matrix[:N]) + input_bias[None,:] + x.repeat_interleave(B, 1)*delta_wi_diag[None, :] )
    j_diag = (wi_diag + delta_wi_diag)[None, :] * derivative * delta_wo_diag[None, :]
    j_diag = torch.sum(j_diag.view(-1, N, B), dim=2)#these are the diagonals of the Jacobian
    
    # collect
    x = x @ (( (1 - mask) * whole_matrix[:N] ).t()) + output_bias[None,:] + torch.sum((x*delta_wo_diag[None, :]).view(-1, N, B), dim=2)

    #return the encoded results, and SumLogDet(Jacobian)
    return x, torch.sum(torch.log(j_diag + 1e-15))
    
 
def MonoTriNetInit(N, Bs, device):  
    """
    monotonic triangular network initialization
    N: dim of input and output
    Bs: each B in list Bs defines block sizes 1xB (input matrix) and Bx1 (output matrix) for each unit
    device: cpu or cuda
    """
    Ws = []
    for B in Bs:
        W = MonoTriNetUnitInit(N, B, device)
        Ws.append(W)
        
    return Ws


def MonoTriNet(x, Ws, nonlinearity='tanh', flip=True):
    """
    monotonic triangular network
    x: input 
    Ws: coefficients 
    nonlinearity: tanh or some self-defined ones
    """
    batch_size, dim_input = x.shape
    
    nll = 0.0
    for W in Ws:
        x, sumlogdetJ = MonoTriNetUnit(x, W, nonlinearity)
        nll = nll - sumlogdetJ
        if flip:
            x = torch.flip(x, [1])#reverse the order of output to balance the network
        
    nll = nll/batch_size + 0.5*torch.sum(x*x)/batch_size + 0.5*dim_input*math.log(2.0*math.pi)
    return x, nll