from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(12,use_bias=False),
        layers.Dense(10,activation='softmax',use_bias=False),
    ])

load_status = model.load_weights("mnist.h5")
#load_status.assert_consumed()

import matplotlib.pyplot as plt
for i,l in enumerate(model.layers):
    print("Layer size:",len(l.weights))
    layer = l.weights[0].numpy()
    print("Weight shape",layer.shape)
    np.save(f'weights/layer_{i}.npy',layer)
    fig = plt.figure()
    flat = layer.flatten()
    plt.hist(flat,bins=[x for x in np.arange(min(flat),max(flat),0.01)])
    plt.xlabel(f"Value")
    plt.ylabel(f"Frequency")
    plt.title(f"Weight distribution for layer {i}")
    plt.savefig(f"weights/Layer_{i}.png")
    if i==1:
        for k in range(layer.shape[-1]):
            fig = plt.figure()
            flat = layer[:,k]
            plt.hist(flat,bins=[y for y in np.arange(min(flat),max(flat),0.01)])
            plt.xlabel(f"Value")
            plt.ylabel(f"Frequency")
            plt.title(f"Weight distribution for layer {i}, connecting to categorical output {k}")
            plt.savefig(f"weights/Layer_{i}_categorical_{k}.png")
