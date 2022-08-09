import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def batch_plot(opts, run_logs, output_dir, prefix, opts_name='test_accuracy'):
	alpha = 0.6
	linewidth = 0.6

	plt.figure()
	plt.hist(opts, bins=100)
	plt.xlabel('Test Accuracy')
	plt.ylabel('Frequency')
	plt.savefig(os.path.join(output_dir,f'{prefix}_{opts_name}_hist.jpg'))
	plt.close()

	plt.figure()
	for i in range(len(run_logs)):
		y = np.asarray([z['loss'] for z in run_logs[i]])
		# plot Y vs iterations
		plt.plot(y, alpha=alpha, linewidth=linewidth)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.savefig(os.path.join(output_dir,f'{prefix}_loss.jpg'))
	plt.close()

	plt.figure()
	for i in range(len(run_logs)):
		# plot acceptance probability vs iterations
		plt.plot([z['alpha'] for z in run_logs[i]], alpha=alpha, linewidth=linewidth)
	plt.xlabel('Iterations')
	plt.ylabel('Metropolis criterion')
	plt.savefig(os.path.join(output_dir,f'{prefix}_alpha.jpg'))
	plt.close()

	if 'accuracy' in run_logs[0][0].keys():
		plt.figure()
		for i in range(len(run_logs)):
			# plot accuracy vs iterations
			plt.plot([z['accuracy']*100 for z in run_logs[i]], alpha=alpha, linewidth=linewidth)
		plt.xlabel('Iterations')
		plt.ylabel('Accuracy')
		plt.savefig(os.path.join(output_dir,f'{prefix}_accuracy.jpg'))
		plt.close()

def plot(run_logs, output_dir, prefix, base_plot=None):
	loss = np.asarray([z['loss'] for z in run_logs])
	t = np.asarray([z['T'] for z in run_logs])
	# plot T vs iterations
	plt.figure()
	plt.plot(t)
	plt.xlabel('Iterations')
	plt.ylabel('Temperature')
	plt.savefig(os.path.join(output_dir,f'{prefix}_T.jpg'))
	plt.close()

	# plot Y vs iterations
	plt.figure()
	plt.plot(loss)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.savefig(os.path.join(output_dir,f'{prefix}_loss.jpg'))
	plt.close()

	# plot acceptance probability vs iterations
	plt.figure()
	plt.plot([z['alpha'] for z in run_logs])
	plt.xlabel('Iterations')
	plt.ylabel('Metropolis criterion')
	plt.savefig(os.path.join(output_dir,f'{prefix}_alpha.jpg'))
	plt.close()

	if 'accuracy' in run_logs[0].keys():
		# plot accuracy vs iterations
		plt.figure()
		plt.plot([z['accuracy']*100 for z in run_logs])
		plt.xlabel('Iterations')
		plt.ylabel('Accuracy')
		plt.savefig(os.path.join(output_dir,f'{prefix}_accuracy.jpg'))
		plt.close()

	if base_plot is None:
		return
	# creating a gif of iterations
	Figure = plt.figure()
	plt.plot(*base_plot)
	plt.savefig(os.path.join(output_dir,f'{prefix}_base.jpg'))
	# plt.savefig('base.jpg')
	# return
	x = np.asarray([z['x'] for z in run_logs])  
	y = np.asarray([z['y'] for z in run_logs])
	def get_vline_params(i):
		return {'x':x[i], 'ymin':min(y[i],0), 'ymax':max(y[i],0)}

	vline = plt.vlines(**get_vline_params(0),colors='red')
	plt.hlines(y=0,xmin=min(base_plot[0]),xmax=max(base_plot[0]),colors='black')
	# create frame every time the function is called
	def AnimationFunction(frame):
	    # lines[0].set_data((x, y))
	    params = get_vline_params(frame)
	    # change the vertical line
	    vline.set_segments([np.array([[params['x'], params['ymin']],
	                         [params['x'], params['ymax']]])])
	    plt.title(f'T: {t[frame]}',loc='right')


	anim = FuncAnimation(Figure, AnimationFunction, frames=len(run_logs), interval=25)
	f = os.path.join(output_dir,f'{prefix}_iter.gif')
	writergif = PillowWriter(fps=30) 
	anim.save(f, writer=writergif)
	plt.close()