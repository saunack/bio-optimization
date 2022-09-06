import os
import numpy as np

## CONSTANTS
SCALE = 1
K = 12

## FUNCTIONS
def one_hot(y):
    out = np.zeros((y.size, 10))
    out[np.arange(y.size), y] = 1
    return out


def initializer_mnist(load_pretrained=False,initializer='glorot',n_models=1):
	if initializer == 'glorot':
		# glorot initializer
		w1 = np.random.uniform(-np.sqrt(6/(28**2+K)),np.sqrt(6/(28**2+K)),size=(n_models,28*28,K))
		w2 = np.random.normal(-np.sqrt(6/(10+K)),np.sqrt(6/(10+K)),size=(n_models,K,10))
	else: #elif initializer == 'gaussian':
		w1 = np.random.normal(size=(n_models,28*28,K),scale=SCALE)
		w2 = np.random.normal(size=(n_models,K,10),scale=SCALE)

	if load_pretrained:
		# saved with K=12
		# add some initial peturbation
		w1 = w1 + np.load(os.path.join('..','mnist_keras','weights','layer_0.npy'))
		w2 = w2 + np.load(os.path.join('..','mnist_keras','weights','layer_1.npy'))
	return [w1,w2]

## x is the weights matrix
def output_mnist(x, images):
	output = images
	# apply 2 dense layers
	for layer in x:
		output = np.matmul(output,layer)

	# output shape: batch_size * categories (10)
	# apply softmax (with upper and lower bound clipping)
	sigmoid = np.clip(np.exp(output),np.exp(-708),np.exp(709))
	sigmoid_sum = np.reshape(np.repeat(np.sum(sigmoid,axis=-1), output.shape[-1]), output.shape)
	softmax = sigmoid/(sigmoid_sum)

	return softmax

def loss_mnist(x, images, labels):
	output = output_mnist(x, images)
	clipped = np.clip(output,1e-7, 1-1e-7)
	# x shape: n_models * [weight matrix 1, weight matrix 2]
	# output shape: n_models * batch_size * categories (10)
	# labels shape: batch_size * categories
	print(output.shape, labels.shape, x[0].shape)
	# add dimension to labels and replicate across new dimension
	labels = np.dstack([labels]*output.shape[0]).reshape(output.shape[0],*labels.shape)
	cross_entropy_1 = labels * np.log(clipped+1e-7)
	cross_entropy_2 = (1-labels) * np.log((1-clipped)+1e-7)
	cross_entropy = cross_entropy_2 + cross_entropy_1
	mean_loss = -np.mean(cross_entropy, axis=-1)
	# take the mean. Sum will give the same results since batch sizes are constant and last batch does not matter
	return np.mean(mean_loss)


def accuracy_mnist(x, images, labels):
	output = output_mnist(x, images)
	predicted_class = np.argmax(output,axis=-1)
	true_class = np.argmax(labels,axis=-1)
	return np.sum(predicted_class==true_class,axis=-1)/predicted_class.shape[-1] 
