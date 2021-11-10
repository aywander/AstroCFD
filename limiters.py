
import numpy as np

def minmod(ql, qi, qr):
    '''
    Minmod limiter
    
    ql, qi, qr:    at least one-dimensional array.
    
    '''
    eps = 1.e-30
    r = (qi - ql) / (qr - qi + eps)
    
    return np.minimum(np.maximum(r, 0), 1.)

def superbee(ql, qi, qr):
    '''
    Superbee limiter
    
    ql, qi, qr:    at least one-dimensional array.
    
    '''
    eps = 1.e-30
    r = (qi - ql) / (qr - qi + eps)
    
    return np.maximum(0., np.maximum(np.minimum(2.*r, 1.), np.minimum(r, 2.)))

def van_leer(ql, qi, qr):
    '''
    Van Leer limiter
    
    ql, qi, qr:    at least one-dimensional array.
    
    '''
    eps = 1.e-30
    r = (qi - ql) / (qr - qi + eps)
    
    return (r + np.abs(r)) / (1 + np.abs(r))

def gen_minmod(ql, qi, qr, theta=1.):
    '''
    Minmod limiter
    
    ql, qi, qr:    at least one-dimensional array.
    
    '''
    eps = 1.e-30
    r = (qi - ql) / (qr - qi + eps)
    
    return np.maximum(0, np.minimum(theta * r, (1 + r) / 2., theta))

