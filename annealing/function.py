import numpy as np

def f(x, base=False):
	def foo(x):
		# minima is at -3.30938
		return 2*np.sin(.5*x)+0.5*np.cos(3*x+1)+np.sin(5*x-4)
		# minima is at -0.802743
		return 2*np.sin(.5*x)+0.5*np.cos(3*x+1)+np.sin(5*x-4) + x*x
	if base:
		return foo(x)
	else:
		return (foo(x)-foo(-3.30938))**2

def sampler(x):
	return np.random.normal(loc=x)

##### MNIST functions

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to import mnist_keras.utils
from mnist_keras.utils import SCALE

def sampler_ml(layers):
	new_layers = []
	# randomly changing layers is worse
	# rand = np.random.uniform(0,1)
	# flag = False
	# for i in range(len(layers)):
	# 	if i > rand and not flag:
	# 		new_layers.append(layers[i]+np.random.normal(size=layers[i].shape,scale=SCALE))
	# 		flag = True
	# 	else:
	# 		new_layers.append(layers[i])
	for w in layers:
		new_layers.append(w+np.random.normal(size=w.shape,scale=SCALE))
	return new_layers

