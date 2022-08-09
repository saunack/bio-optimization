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
"""
	Model to be compared against
	model = keras.Sequential([
	    layers.Input(input_shape=(784,)),
	    layers.Dense(12,use_bias=False),
	    layers.Dense(10,activation='softmax',use_bias=False),
	])
"""

def one_hot(y):
    out = np.zeros((y.size, 10))
    out[np.arange(y.size), y] = 1
    return out

SCALE = 1
def initializer_mnist(load_pretrained=False,initializer='gaussian'):
	K = 12
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
		w1 = w1 + np.load('../mnist_keras/weights/layer_0.npy')
		w2 = w2 + np.load('../mnist_keras/weights/layer_1.npy')
	return [w1,w2]

def output_mnist(x, images):
	# apply 2 dense layers
	output = np.matmul(np.matmul(images,x[0]),x[1])
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

