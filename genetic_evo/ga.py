import numpy as np

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

def ga_ml(fitness, mutation, crossover, pop_generator, elitism=0.2, pop=100,
		 mutation_prob=0.05, iterations=100, threshold=None, batch_yield=None, batch_size = 32, kwargs=None):
    log = []
    if iterations is not None and threshold is not None:
        raise ValueError("Specify one of iterations and threshold")
    
    # initial population
    x = pop_generator(pop, kwargs)
    to_keep = int(elitism*pop)

    for epoch in generate_iter(iterations):
        for image, label in batch_yield(batch_size):
            y = fitness(x, image, label)
            # generate new samples
            sort_idx = np.argsort(y)
            
            elite = x[sort_idx[:to_keep],:]
            
            min_y = y[sort_idx[0]]
            max_y = y[sort_idx[-1]]
            #avg_y = avg(y)
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
