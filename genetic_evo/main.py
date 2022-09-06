from ga import ga, ga_ml
from utils import plot
import argparse

from function import sort_f, sort_mutation, sort_crossover, sort_initial_pop
from function import mnist_mutation, mnist_crossover

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to import mnist_keras.utils

from mnist_keras.utils import loss_mnist, accuracy_mnist, initializer_mnist
from mnist import MNIST


parser = argparse.ArgumentParser()
# options for output naming/location
parser.add_argument('--prefix', default="ga_",
                    help='add file prefix names')
parser.add_argument('--output_dir', default="output/",
                    help='output_directory')
# options for genetic evolution algorithm
parser.add_argument('--pop', default=100, type=int,
                    help='population size')
parser.add_argument('--mutation_prob', '-p', default=0.05, type=float,
                    help='probability for mutation')
parser.add_argument('--elitism', '-e', default=0.2, type=float,
                    help='percentage of population to keep after an iteration')
parser.add_argument('--iterations', '-i', default=100, type=int,
                    help='Number of iterations to run the evolution for (threshold is ignored if iterations are provided)')
parser.add_argument('--thresh', default=None, type=float,
                    help='threshold for function')
# options for objectives
parser.add_argument('--objective', default="ml",
                    help='Objective to run the code for (ml/poly)')
# options for polynomial objective
parser.add_argument('-K', default=10, type=int,
                    help='length of list to be sorted for sorting')
# options for ml objective
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch_size for mnist')
parser.add_argument('--load_pretrained', dest='load_pretrained', action='store_true',
                    help='Load pretrained weights for classification for MNIST')
parser.add_argument('--initializer', default='gaussian',
                    help='Kernel initializer. glorot/gaussian supported. for pretrained, peturbations are done using initializer')

args = parser.parse_args()

def run_ga_sort(initial_pop, mutation_prob, elitism, list_size, threshold, iterations):
	if iterations is not None:
		run_logs = ga(sort_f, sort_mutation, sort_crossover, sort_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, iterations=iterations, list_size=list_size)
	else:
		run_logs = ga(sort_f, sort_mutation, sort_crossover, sort_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, threshold=threshold, list_size=list_size)
	return run_logs

def run_ga_mnist(initial_pop, mutation_prob, elitism, threshold, iterations, batch_size, load_pretrained, kernel_initializer):
	mndata = MNIST(os.path.join('..','data'),return_type='numpy')
	if iterations is not None:
		run_logs = ga_ml(loss_mnist, accuracy_mnist, mnist_mutation, mnist_crossover, initializer_mnist, mndata,
			elitism=elitism, pop=initial_pop, mutation_prob=mutation_prob, iterations=iterations, 
			load_pretrained=load_pretrained, kernel_initializer=kernel_initializer, batch_size=batch_size)
	else:
		run_logs = ga_ml(loss_mnist, accuracy_mnist, mnist_mutation, mnist_crossover, initializer_mnist, mndata,
            elitism=elitism, pop=initial_pop, mutation_prob=mutation_prob, threshold=threshold, 
            load_pretrained=load_pretrained, kernel_initializer=kernel_initializer, batch_size=batch_size)
	return run_logs

if __name__ == "__main__":
	if args.objective == 'sort':
		x_opt, run_logs, (x, y) = run_ga_sort(args.pop,args.mutation_prob,args.elitism,list_size=args.K,
											 threshold=args.thresh, iterations=args.iterations)
		print("Optimal solution: ",x, y)
		print("Final iteration: ",x_opt[:5,:])
	elif args.objective == 'ml':
		x_opt, run_logs, (x, y) = run_ga_mnist(args.pop,args.mutation_prob,args.elitism,
											 threshold=args.thresh, iterations=args.iterations, 
											 load_pretrained=args.load_pretrained, kernel_initializer=args.initializer, batch_size=args.batch_size)
	else:
		x_opt, run_logs, (x, y) = None, None, (None, None)
	plot(run_logs,args.output_dir,args.prefix)
