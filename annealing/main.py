from annealing import annealing
from function import f, sampler
from utils import plot
import argparse

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

args = parser.parse_args()

def run_annealing_quadratic(x, T, decay, threshold, iterations, base_plot):
	if iterations is not None:
		run_logs = annealing(f, sampler, x, T, decay=decay, iterations=iterations, base_plot=base_plot)
	else:
		run_logs = annealing(f, sampler, x, T, decay=decay, threshold=threshold, base_plot=base_plot)
	return run_logs

if __name__ == "__main__":
	x_opt, run_logs, (x,y) = run_annealing_quadratic(args.x0,args.t0,decay=args.decay,
										 threshold=args.thresh, iterations=args.iterations, base_plot=args.base_plot)
	print("Optimal solution: ",x_opt)
	plot(run_logs,args.output_dir,args.prefix,base_plot=(x,y))
