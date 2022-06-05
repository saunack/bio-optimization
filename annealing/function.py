import numpy as np

def f(x):
	return 2*np.sin(.5*x)+0.5*np.cos(3*x+1)+np.sin(5*	x-4)
    # return (x-1)*(x+2)*x*(x+3) + (x-4)*(x-6)*(x+1)

def sampler(x):
	return np.random.normal(loc=x)

