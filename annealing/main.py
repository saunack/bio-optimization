from annealing import annealing, annealing_ml
from function import f, sampler
from utils import plot, batch_plot
import argparse

from mnist import MNIST
from function import sampler_ml

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to import mnist_keras.utils

from mnist_keras.utils import loss_mnist, accuracy_mnist, initializer_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--disable_gif', dest='base_plot', action='store_false',
						help='flag for disabling gif creation')
parser.add_argument('--prefix', default="sa_",
                    help='add file prefix names')
parser.add_argument('--output_dir', default="output/",
                    help='output_directory')
parser.add_argument('--x0', default=0, type=float,
                    help='initial solution')
parser.add_argument('--t0', default=20, type=float,
                    help='initial temperature')
parser.add_argument('--decay', default=0.92, type=float,
                    help='decay factor for temperature')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='threshold for temperature')
parser.add_argument('--iterations', default=None, type=int,
                    help='Number of iterations to run the code for (threshold is ignored if iterations are provided)')
parser.add_argument('--objective', default="ml",
                    help='Objective to run the code for (ml/poly)')
parser.add_argument('--load_pretrained', dest='load_pretrained', action='store_true',
                    help='Load pretrained weights for classification for MNIST')
parser.add_argument('--initializer', default='gaussian',
                    help='Kernel initializer. glorot/gaussian supported. for pretrained, peturbations are done using initializer')

args = parser.parse_args()

def run_annealing_quadratic(x, T, decay, threshold, iterations, base_plot):
	if iterations is not None:
		run_logs = annealing(f, sampler, x, T, decay=decay, iterations=iterations, base_plot=base_plot)
	else:
		run_logs = annealing(f, sampler, x, T, decay=decay, threshold=threshold, base_plot=base_plot)
	return run_logs

def run_annealing_ml(T, decay, threshold, iterations, load_pretrained, kernel_initializer):
	mndata = MNIST(os.path.join('..','data'),return_type='numpy')
	if iterations is not None:
		y, run_logs = annealing_ml(loss_mnist, accuracy_mnist, sampler_ml, initializer_mnist, mndata, T, decay=decay, iterations=iterations, load_pretrained=load_pretrained, kernel_initializer=kernel_initializer)
	else:
		y, run_logs = annealing_ml(loss_mnist, accuracy_mnist, sampler_ml, initializer_mnist, mndata, T, decay=decay, threshold=threshold, load_pretrained=load_pretrained, kernel_initializer=kernel_initializer)
	return y, run_logs

if __name__ == "__main__":
	if args.objective == 'ml':
		opt, run_logs = run_annealing_ml(args.t0,decay=args.decay,
						 threshold=args.thresh, iterations=args.iterations,
						 load_pretrained=args.load_pretrained, kernel_initializer=args.initializer)
		print("Test accuracy: ",opt)
		plot(run_logs,args.output_dir,args.prefix)
	elif args.objective == 'poly':
		opt, run_logs, (x,y) = run_annealing_quadratic(args.x0,args.t0,decay=args.decay,
											 threshold=args.thresh, iterations=args.iterations, base_plot=args.base_plot)
		print("Optimal solution: ",opt)
		plot(run_logs,args.output_dir,args.prefix,base_plot=(x,y))
	elif args.objective == 'poly_100':
		opts = []
		logs = []
		for i in range(100):
			opt, run_logs, (x,y) = run_annealing_quadratic(args.x0,args.t0,decay=args.decay,
											 threshold=args.thresh, iterations=args.iterations, base_plot=args.base_plot)
			opts.append(f(opt))
			logs.append(run_logs)
			if i%10 == 0:
				print(f"At run {i}. Plotting")
				batch_plot(opts,logs,args.output_dir,args.prefix,opts_name='loss')
		batch_plot(opts,logs,args.output_dir,args.prefix,opts_name='loss')
	elif args.objective == 'ml_100':
		opts = []
		logs = []
		for i in range(100):
			opt, run_logs = run_annealing_ml(args.t0,decay=args.decay,
						 threshold=args.thresh, iterations=args.iterations,
						 load_pretrained=args.load_pretrained, kernel_initializer=args.initializer)
			opts.append(opt)
			logs.append(run_logs)
			if i%10 == 0:
				print(f"At run {i}. Plotting")
				batch_plot(opts,logs,args.output_dir,args.prefix)
		batch_plot(opts,logs,args.output_dir,args.prefix)
	else:
		opt, run_logs = None, None
