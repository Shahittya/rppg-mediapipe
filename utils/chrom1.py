import numpy as np
def chrom_method(r,g,b):
    r=np.array(r)
    g=np.array(g)
    b=np.array(b)
    c_chrom=np.array([
        3*r - 2*g,
        1.5*r + g - 1.5*b
    ])
    
    std1=np.std(c_chrom[0])
    std2=np.std(c_chrom[1])
    if std2==0:
        return np.zeros_like(r)
    alpha=std1/std2
    s=c_chrom[0] - alpha*c_chrom[1]
    return s
