import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def plot(run_logs, output_dir, prefix, base_plot=None):
	min_y = np.asarray([z['min_y'] for z in run_logs])
	max_y = np.asarray([z['max_y'] for z in run_logs])

	# plot min_Y vs iterations
	plt.figure()
	plt.plot(min_y,label='min')
	plt.plot(max_y,label='max')
	plt.xlabel('Iterations')
	plt.ylabel('Fitness/Objective function')
	plt.legend()
	plt.savefig(os.path.join(output_dir,f'{prefix}_y.jpg'))

	if 'min_acc' in run_logs[0].keys():
		min_acc = np.asarray([z['min_acc'] for z in run_logs])
		max_acc = np.asarray([z['max_acc'] for z in run_logs])
		# plot min_Y vs iterations
		plt.figure()
		plt.plot(min_acc,label='min')
		plt.plot(max_acc,label='max')
		plt.xlabel('Iterations')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(os.path.join(output_dir,f'{prefix}_accuracy.jpg'))
