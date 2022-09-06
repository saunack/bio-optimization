import numpy as np


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
    # swap mutation is suitable here because the resulting list should be a permutation
    N, k = x.shape
    # get random numbers for each of N individuals for checking if mutation will be done
    # generates 2D array of Nxk. For each N, value is [True]*k or [False]*k
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


##### MNIST functions
"""
    Model to be compared against
    model = keras.Sequential([
        layers.Input(input_shape=(784,)),
        layers.Dense(12,use_bias=False),
        layers.Dense(10,activation='softmax',use_bias=False),
    ])
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mnist_keras.utils import SCALE

def mnist_mutation(x, p):
    n_models = x[0].shape[0]
    # get random numbers for each of N individuals for checking if mutation will be done
    rand = np.random.rand(n_models) < p
    # choose whether the first weight matrix will be mutated or the second
    matrix_rand = np.random.rand(n_models) < 0.5
    def tile(m,shape):
    	return np.tile(np.expand_dims(np.expand_dims(m,-1),-1),shape[1:])
    mutation_w1_bool = tile(rand,x[0].shape) * tile(matrix_rand,x[0].shape)
    mutation_w2_bool = tile(rand,x[1].shape) * tile(~matrix_rand,x[1].shape)

    # changing mutation might change results
    x[0] = x[0] + mutation_w1_bool * np.random.uniform(size=x[0].shape)
    x[1] = x[1] + mutation_w2_bool * np.random.uniform(size=x[1].shape)

    return x

def mnist_crossover(x, i1, i2):
	# x1 and x2 shape: w1 x w2
	# different ways of crossing over
	# # 1. switch layers directly
	# layer = 1
	# temp = x1[layer].copy()
	# x1[layer] = x2[layer]
	# x2[layer] = temp
	# return x1, x2
	# 2. switch all connections for a layer
	# choose layer to switch
	x1 = [x[i][i1,:,:] for i in range(len(x))]
	x2 = [x[i][i2,:,:] for i in range(len(x))]
	layer = 0 if np.random.rand()<0.5 else 1
	# choose whether to switch all connections from the input side or the output side
	# e.g.: w1 x w2. if head_tail == 1, k x w2 will be switched. Otherwise, w1 x k
	head_tail = 0 if np.random.rand()<0.5 else 1
	idx = np.random.randint(1,x1[layer].shape[head_tail]+1)
	if head_tail == 1:
		temp = x1[layer][:idx,:].copy()
		x1[layer][:idx,:] = x2[layer][:idx,:]
		x2[layer][:idx,:] = temp
	else:	
		temp = x1[layer][:,:idx].copy()
		x1[layer][:,:idx] = x2[layer][:,:idx]
		x2[layer][:,:idx] = temp
	return x1, x2

