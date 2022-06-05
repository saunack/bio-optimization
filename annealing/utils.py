import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def plot(run_logs, output_dir, prefix, base_plot=None):
	x = np.asarray([z['x'] for z in run_logs])  
	y = np.asarray([z['y'] for z in run_logs])
	t = np.asarray([z['T'] for z in run_logs])
	# plot T vs iterations
	plt.figure()
	plt.plot(t)
	plt.xlabel('Iterations')
	plt.ylabel('Temperature')
	plt.savefig(os.path.join(output_dir,f'{prefix}_T.jpg'))

	# plot Y vs iterations
	plt.figure()
	plt.plot(y)
	plt.xlabel('Iterations')
	plt.ylabel('Objective function')
	plt.savefig(os.path.join(output_dir,f'{prefix}_y.jpg'))

	# plot acceptance probability vs iterations
	plt.figure()
	plt.plot([z['alpha'] for z in run_logs])
	plt.xlabel('Iterations')
	plt.ylabel('Metropolis criterion')
	plt.savefig(os.path.join(output_dir,f'{prefix}_alpha.jpg'))

	if base_plot is None:
		return
	# creating a gif of iterations
	Figure = plt.figure()
	plt.plot(*base_plot)
	plt.savefig('base.jpg')
	# plt.savefig('base.jpg')
	# return
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