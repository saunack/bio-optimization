from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

def convert(y):
    out = np.zeros((y.size, 10))
    out[np.arange(y.size), y] = 1
    return out


model = keras.Sequential([
        #layers.Flatten(input_shape=(28,28)),
        layers.Dense(12,use_bias=False),
        layers.Dense(10,activation='softmax',use_bias=False),
    ])

from mnist import MNIST
data = MNIST('/home/rope/Projects/optimization/genetic_evo/data/',return_type='numpy')
train_x,train_y = data.load_training()
train_y = convert(train_y)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate)
    )

test_x,test_y = data.load_testing()
one_hot_y = convert(test_y)

hist = model.fit(train_x,train_y,epochs=10,validation_split=0.15)

model.save('mnist.h5')

### Testing the model
from mnist import MNIST
data = MNIST('/home/rope/Projects/optimization/genetic_evo/data/',return_type='numpy')
test_x,test_y = data.load_testing()
test_y_one_hot = convert(test_y)

out=model.predict(test_x)
predictions = np.argmax(out,axis=1)
correct = np.sum(predictions==test_y)
print("Accuracy: ", correct/out.shape[0])

