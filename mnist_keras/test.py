import numpy as np
def loss(output, labels):
	# output and labels shape: batch_size * categories (10)
	clipped = np.clip(output,1e-7, 1-1e-7)
	cross_entropy_1 = labels * np.log(clipped+1e-7)
	cross_entropy_2 = (1-labels) * np.log((1-clipped)+1e-7)
	#print(cross_entropy_1, cross_entropy_2)
	cross_entropy = cross_entropy_2 + cross_entropy_1
	# cross_entropy: batch_size x categories
	mean_loss = -np.mean(cross_entropy, axis=-1)
	return mean_loss
 
y = np.array([[1.,1.,1.],[1.,1.,1.]])
x = np.array([[1.,1.,0.],[1.,1.,0.]])
print(loss(y,x))
#print(loss(x,y))
