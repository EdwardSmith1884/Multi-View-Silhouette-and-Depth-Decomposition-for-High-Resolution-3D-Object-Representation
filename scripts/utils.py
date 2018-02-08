import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
import scipy.io as sio
import random
import tensorlayer as tl

# making some subdirectories, checkpoint stores the models,
# savepoint saves some created objects at each epoch
def make_directories(checkpoint,savepoint): 
	if not os.path.exists(checkpoint):
		os.makedirs(checkpoint)

	if not os.path.exists(savepoint):
		os.makedirs(savepoint)


def make_batch(files, h, l, valid = False, side = -1, occupancy = False):
	
	ratio = h//l
	highs, lows, low_ups, sides =  [], [], [], []
	start_time = time.time()
	alter = (side == -1)  
	for i,f in enumerate(files):
		if alter: 
			side = random.randint(0, 5) if not valid else i%6 
		sides.append(np.ones((l,l,1))*side)
		file = f.split('_')[0] + '_' + str(side)  + '_' + f.split('_')[-1]
		face = sio.loadmat(file)

		high = (face['high_odm']).reshape((h,h,1))
		low  = (face['low_odm']).reshape((l,l,1))

		if not occupancy:
			a,b,c = np.where(low > 0) 
			up = np.zeros((h,h,1))
			for x,y,z in zip(a,b,c): 
				up[ratio*x:ratio*(x+1), ratio*y:ratio*(y+1), 0] = (low[x,y,0] -1) *ratio  +1 #  uscale low resolution odm 

			change = np.where(high == 0)
			up[change] = 0
			highs.append(high/float(h))
			low_ups.append(up)

		else: 
			change = np.where( high != 0  )
			high[change] = 1. 
			highs.append(high)

		change = np.where(low == 0)
		low[change] = l*2 
		lows.append(low/float(l*2))
		

	gen = {'high':np.array(highs), 'low': np.array(lows),'low_up':np.array(low_ups), 'side': np.array(sides)}
	return gen, start_time
   


def grab_files(data_dir):
	data_dir+='/'
	files = [f for f in glob(data_dir + '*.mat') if 'face_' in f]
	return files

def save_networks(checkpoint_dir, sess, net, epoch, name = None ):
	print("[*] Saving checkpoints...")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	epoch = '_' + epoch
	if name is not None: epoch = name + epoch
	else: epoch = 'net'	+ epoch
	epoch = epoch + '.npz'
	net_name = os.path.join(checkpoint_dir, epoch)
	tl.files.save_npz(net.all_params, name=net_name, sess=sess)
	print("[*] Saving checkpoints SUCCESS!")


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	import numpy as np
	from math import factorial
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')


def load_networks(checkpoint_dir, sess, net, epoch, name = None): 
	print "[*] Loading checkpoints..." 
	epoch = '_' + epoch
	if name is not None: epoch = name + epoch
	else: epoch = 'net'	+ epoch
	epoch = epoch + '.npz'
	net_name = os.path.join(checkpoint_dir,  epoch)
	
	if not (os.path.exists(net_name)):
		print "[!] Loading checkpoint failed!"
	else:
		net_loaded_params = tl.files.load_npz(name=net_name)
		
		tl.files.assign_params(sess, net_loaded_params, net)
		print "[*] Loading checkpoint SUCCESS!"
		
			
def render_graphs(save_dir,epoch, recon_loss, valid_loss, exact_valid_loss, name = 'depth'): 
	if len(recon_loss)>101: 
		valid_list = []
		for i in range(epoch+1): 
			valid_list.append((i+1)*len(recon_loss)//(epoch+1) )
		smoothed_recon = savitzky_golay(recon_loss, 101, 3)
		plt.plot(recon_loss, color='blue') 
		plt.plot(smoothed_recon, color = 'red')
		plt.plot(valid_list, valid_loss,color='green')
		plt.grid()
		plt.savefig(save_dir+'/' + name + '_' + 'recon.png' )
		plt.clf()
	if len(exact_valid_loss)>5: 
	
		plt.plot( exact_valid_loss[5:])
		plt.grid()
		plt.savefig(save_dir+'/' + name + '_' + 'exact_recon.png' )
		plt.clf()

def evaluate_voxel_prediction(prediction,gt):
  """  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
  intersection = np.sum(np.logical_and(prediction,gt))
  union = np.sum(np.logical_or(prediction,gt))
  IoU = intersection / union
  return IoU
