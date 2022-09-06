import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to import mnist_keras.utils

import numpy as np
from mnist_keras.utils import one_hot

def generate_iter(iterations):
    for i in range(iterations):
        yield i

def generate_T_decay(T, decay, threshold):
    while T >= threshold:
        T = T*decay
        yield T

def ga(fitness, mutation, crossover, pop_generator, elitism=0.2, pop=100,
		 mutation_prob=0.05, iterations=100, threshold=None, **kwargs):
    log = []
    if iterations is not None and threshold is not None:
        raise ValueError("Specify one of iterations and threshold")
    
    # initial population
    x = pop_generator(pop, kwargs)
    to_keep = int(elitism*pop)

    for epoch in generate_iter(iterations):
        y = fitness(x,kwargs)
        # generate new samples
        sort_idx = np.argsort(y)
        
        elite = x[sort_idx[:to_keep],:]
        
        min_y = y[sort_idx[0]]
        max_y = y[sort_idx[-1]]
        log.append({'min_y':min_y, 'max_y':max_y})
        
        new_pop = []
        while len(new_pop) < pop-to_keep:
            i1 = np.random.randint(to_keep)
            i2 = np.random.randint(to_keep)
            if i1 != i2:
                cross = crossover(x[i1].copy(),x[i2].copy())
                new_pop.append(np.expand_dims(cross[0],0))
                new_pop.append(np.expand_dims(cross[1],0))
        x = np.concatenate([elite]+new_pop)
        x = mutation(x, mutation_prob)
    return x, log, (x[0], fitness(x[:1,:])[0])

def ga_ml(fitness, accuracy, mutation, crossover, pop_generator, data, elitism=0.2, pop=100,
		 mutation_prob=0.05, iterations=None, threshold=None,
		 batch_size = 32, load_pretrained=False, kernel_initializer=None):
    log = []
    if iterations is not None and threshold is not None:
        raise ValueError("Specify one of iterations and threshold")
    
    temp_iterator =  generate_iter(iterations) if iterations is not None else generate_T_decay(T, decay, threshold)

    # initial population
    x = pop_generator(load_pretrained, kernel_initializer, n_models=pop)
    to_keep = int(elitism*pop)
    generator = data.load_training_in_batches(batch_size)

    for temperature in temp_iterator:
        try:
            images, labels = next(generator)
            labels = one_hot(labels)
        except StopIteration:
            generator = data.load_training_in_batches(batch_size)
            images, labels = next(generator)
            labels = one_hot(labels)

        y = fitness(x, images, labels)
        print(y.shape)
        acc = accuracy(x, images, labels)
        # generate new samples
        sort_idx = np.argsort(y)

        elite = [x[i][sort_idx[:to_keep],:,:] for i in range(len(x))]
        
        min_y, max_y = y[sort_idx[0]], y[sort_idx[-1]]
        min_acc, max_acc = acc[sort_idx[0]], acc[sort_idx[-1]]
        #avg_y = avg(y)
        log.append({'min_y':min_y, 'max_y':max_y,'min_acc':min_acc,'max_acc':max_acc})
        
        new_pop = []
        while len(new_pop) < pop-to_keep:
            i1 = np.random.randint(to_keep)
            i2 = np.random.randint(to_keep)
            if i1 != i2:
                cross = crossover(x,i1,i2)
                new_pop.append(cross[0])
                new_pop.append(cross[1])
        x[0] = np.concatenate([k[0] for k in elite]+[np.expand_dims(k[0],0) for k in new_pop])
        x[1] = np.concatenate([k[1] for k in elite]+[np.expand_dims(k[1],0) for k in new_pop])
        x = mutation(x, mutation_prob)
        print(log[-1])

    test_images, test_labels = data.load_testing()
    test_labels_category = one_hot(test_labels)
    test_accuracy = accuracy(x, test_images, test_labels)
	
    return test_accuracy, log
