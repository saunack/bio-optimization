import numpy as np


## Functions for digit classification objective
def mnist_initial_pop(size, kwargs):
    w1 = np.random.rand((size, 784, 12))
    w2 = np.random.rand((size, 12, 1))
    return [w1,w2]

def mnist_f(w, images, labels):
    # weights are size pop_sizex784x12 and pop_sizex12x1
    # image sizes are batch_sizex784
    # we need scores for each individual, i.e., pop_size number of scores
    # averaging across batch will be done via sum
    images = np.asarray(images).expand_dims(axis=1) # image size will be batchx1x784 (already flattened)
    labels = np.asarray(labels)
    
    # output shape will be batchx1x1
    outputs = np.matmul(np.matmul(w[0],images),w[1])
    outputs = outputs[:,0,0]
    rmse = (outputs-labels)**2
    return rmse

## Functions for sorting objective
def sort_initial_pop(size, kwargs):
    rng = np.random.default_rng()
    if 'list_size' not in kwargs.keys():
        raise ValueError("Need to specify list_size for sorting objective")

    K = kwargs.pop('list_size')
    original = [np.expand_dims(np.arange(K),0) for _ in range(size)]
    permuted = [rng.permutation(x,axis=1) for x in original]
    return np.concatenate(permuted,axis=0)

def sort_f(x, kwargs):
    k = x.shape[-1]
    return np.sum(np.abs(x[:,:-1]-x[:,1:]),axis=1) - (k-1)
    # return (x-1)*(x+2)*x*(x+3) + (x-4)*(x-6)*(x+1)

def sort_mutation(x, p):
    # batch processing for mutation
    N, k = x.shape
    # get random numbers for each of N individuals
    rand = np.tile(np.expand_dims(np.random.rand(N) < p, 1), (1,k))
    
    # generate switching positions for mutation
    idx = np.tile(np.arange(k),(N,1))
    i1 = idx == np.expand_dims(np.random.randint(k, size=N),1)
    i2 = idx == np.expand_dims(np.random.randint(k, size=N),1)
    
    # switch
    temp = x[rand & i1]
    x[rand & i1] = x[rand & i2]
    x[rand & i2] = temp

    return x

def sort_crossover(x1, x2):
    # can only be done pairwise
    N = x1.shape[-1]
    idx = np.random.randint(1,N)
    #print(f"Crossover index {idx}")
    changes = 0
    for i in range(idx):
        i1_in_2 = np.where(x2==x1[i])[0][0]
        i2_in_1 = np.where(x1==x2[i])[0][0]
        temp = x2[i]
        x2[i] = x2[i1_in_2]
        x2[i1_in_2] = temp
        changes += 1 if i1_in_2 != i else 0
        
        temp = x1[i]
        x1[i] = x1[i2_in_1]
        x1[i2_in_1] = temp
        changes += 1 if i2_in_1 != i  else 0
        
        if changes < idx:
            break
        
    return x1, x2
