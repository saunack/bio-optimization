import numpy as np

def generate_T_iter(T, decay, iterations):
	for _ in range(iterations):
		T = T*decay
		yield T

def generate_T_decay(T, decay, threshold):
	while T >= threshold:
		T = T*decay
		yield T

def annealing(objective, sampler, x, T=5, decay=0.8, iterations=None, threshold=None, comp=None, base_plot=False):
	log = []
	if iterations is not None and threshold is not None:
		raise ValueError("Specify one of iterations and threshold")
	
	temp_iterator =  generate_T_iter(T, decay, iterations) if iterations is not None else generate_T_decay(T, decay, threshold)
	min_x = x
	max_x = x
	for temperature in temp_iterator:
		y = objective(x)
		# generate new samples
		x_new = sampler(x)
		y_new = objective(x_new)
	
		# acceptance probability = e^((y-y_new)/T). If y>=y_new, acceptance probability >= 1. No need to compute the probability to prevent
		# overflow errors
		# print("bug",y,y_new,temperature,(y-y_new)/temperature)
		if (comp is None and y_new < y) or (comp is not None and comp(y_new,y)):
			metropolis_criterion = 1
			x = x_new
		else:
			metropolis_criterion = np.exp(-(y_new-y)/temperature)
			uniform_sample = np.random.uniform(0,1)
			if uniform_sample > metropolis_criterion:
				pass
			else:
				x = x_new
		min_x = min(x,min_x)
		max_x = max(x,max_x)
		log.append({'x':x, 'y':objective(x), 'T':temperature, 'alpha':metropolis_criterion})
	if base_plot:
		scale = 1.5
		x_range = np.linspace(min_x*scale if min_x < 0 else min_x/scale,max_x*scale,500)
		y_range = np.asarray([objective(z) for z in x_range])
		return x, log, (x_range,y_range)
	else:
		return x, log
