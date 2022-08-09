from ga import ga, ga_ml
from utils import plot
import argparse

from function import sort_f, sort_mutation, sort_crossover, sort_initial_pop

from mnist import MNIST
from function import mnist_f, mnist_mutation, mnist_crossover, mnist_initial_pop

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default="ga_",
                    help='add file prefix names')
parser.add_argument('--output_dir', default="output/",
                    help='output_directory')
parser.add_argument('--pop', default=100, type=int,
                    help='population size')
parser.add_argument('--mutation_prob', '-p', default=0.05, type=float,
                    help='probability for mutation')
parser.add_argument('--elitism', '-e', default=0.2, type=float,
                    help='percentage of population to keep after an iteration')
parser.add_argument('--thresh', default=None, type=float,
                    help='threshold for function')
parser.add_argument('--function','-f', default='mnist', 
                    help='objective function to use (mnist/sort)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch_size for mnist')
parser.add_argument('-K', default=10, type=int,
                    help='length of list to be sorted for sorting')
parser.add_argument('--iterations', '-i', default=100, type=int,
                    help='Number of iterations to run the evolution for (threshold is ignored if iterations are provided)')

args = parser.parse_args()

def run_ga_sort(initial_pop, mutation_prob, elitism, list_size, threshold, iterations):
	if iterations is not None:
		run_logs = ga(sort_f, sort_mutation, sort_crossover, sort_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, iterations=iterations, list_size=list_size)
	else:
		run_logs = ga(sort_f, sort_mutation, sort_crossover, sort_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, threshold=threshold, list_size=list_size)
	return run_logs

def run_ga_mnist(initial_pop, mutation_prob, elitism, list_size, threshold, iterations, batch_size):
    mndata = MNIST('../data',return_type='numpy')
	if iterations is not None:
		run_logs = ga_ml(mnist_f, mnist_mutation, mnist_crossover, mnist_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, iterations=iterations, batch_yield=mndata.load_training_in_batches, batch_size=batch_size)
	else:
		run_logs = ga_ml(mnist_f, mnist_mutation, mnist_crossover, mnist_initial_pop, elitism=elitism,
                pop=initial_pop, mutation_prob=mutation_prob, threshold=threshold, batch_yield=mndata.load_training_in_batches, batch_size=batch_size)
	return run_logs

if __name__ == "__main__":
	if args.function == 'sort':
		x_opt, run_logs, (x, y) = run_ga_sort(args.pop,args.mutation_prob,args.elitism,list_size=args.K,
											 threshold=args.thresh, iterations=args.iterations)
		print("Optimal solution: ",x, y)
		print("Final iteration: ",x_opt[:5,:])
	elif args.function == 'mnist':
		x_opt, run_logs, (x, y) = run_ga_mnist(args.pop,args.mutation_prob,args.elitism,
											 threshold=args.thresh, iterations=args.iterations, batch_size=args.batch_size)
	else:
		x_opt, run_logs, (x, y) = None, None, (None, None)
	plot(run_logs,args.output_dir,args.prefix)
