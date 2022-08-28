import numpy as np

## CONSTANTS
SCALE = 1
K = 12

## FUNCTIONS
def one_hot(y):
    out = np.zeros((y.size, 10))
    out[np.arange(y.size), y] = 1
    return out


def initializer_mnist(load_pretrained=False,initializer='gaussian'):
	if initializer == 'glorot':
		# glorot initializer
		w1 = np.random.uniform(-np.sqrt(6/(28**2+K)),np.sqrt(6/(28**2+K)),size=(28*28,K))
		w2 = np.random.normal(-np.sqrt(6/(10+K)),np.sqrt(6/(10+K)),size=(K,10))
	else: #elif initializer == 'gaussian':
		w1 = np.random.normal(size=(28*28,K),scale=SCALE)
		w2 = np.random.normal(size=(K,10),scale=SCALE)

	if load_pretrained:
		# saved with K=12
		# add some initial peturbation
		w1 = w1 + np.load(os.path.join('weights','layer_0.npy'))
		w2 = w2 + np.load(os.path.join('weights','layer_1.npy'))
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
	# output and labels shape: batch_size * categories (10)
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
	return np.sum(predicted_class==true_class)/predicted_class.shape[0]
 
