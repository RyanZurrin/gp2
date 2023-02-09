import numpy as np


def dataSplit(num, split = 4):
    n = num
    perm = np.random.permutation(n)
    if(split == 4):
#         splits = [0.002, 0.2, 0.399, 0.399]  # new
        
        splits = [0.01, 0.2, 0.395, 0.395]
    else:
        splits = [0.6, 0.1, 0.3]
    assert(sum(splits)==1.0)
    n_vals = [ int(n * s) for s in splits]
    n_vals = np.cumsum(n_vals)
    
    return n_vals, perm

# def dataSplit(num, split = 4):
    
#     perm = np.random.permutation(num)
#     if(split == 4):
#         splits = [0.02, 0.2, 0.2, 0.58]
#     else:
#         splits = [0.5, 0.5]
#     assert(sum(splits)==1.0)
#     n_vals = [ int(num * s) for s in splits]
# #     n_vals = np.cumsum(n_vals)
    
#     return np.split(perm, n_vals[:-1])
    