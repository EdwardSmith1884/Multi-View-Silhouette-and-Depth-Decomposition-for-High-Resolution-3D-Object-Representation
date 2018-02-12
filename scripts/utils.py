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


# loads data for each files 
def make_batch(files, h, l, valid = False, side = -1, occupancy = False):
	ratio = h//l
	highs, lows, low_ups, sides =  [], [], [], []
	start_time = time.time()
	alter = (side == -1)  ### if the odm side should be random or not 
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
			highs.append(high)
			low_ups.append(up)

		else: 
			change = np.where( high != 0  )
			high[change] = 1. 
			highs.append(high)

		# change = np.where(low == 0)
		# low[change] = l*2 
		lows.append(low)
		

	gen = {'high':np.array(highs), 'low': np.array(lows),'low_up':np.array(low_ups), 'side': np.array(sides)}
	return gen, start_time
   
def make_objs(files): 
	high_objs, low_objs = [], []
	for f in files: 
		file = f.split('_')[0][:-4] + 'full_object_'+ f.split('_')[-1]
		face = sio.loadmat(file)
		high_objs.append((face['model']))
		low_objs.append((face['low_model']))
	return np.array(high_objs), np.array(low_objs)

# grabs all files for training 
def grab_files(data_dir):
	data_dir+='/'
	files = [f for f in glob(data_dir + '*.mat') if 'face_' in f]
	return files

# saves netowrks during training 
def save_networks(checkpoint_dir, sess, net, epoch, name = '' ):
	print("[*] Saving checkpoints...")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	epoch = name + '_' + epoch + '.npz'
	best = name + '_best.npz'
	net_name = os.path.join(checkpoint_dir, epoch)
	best_name = os.path.join(checkpoint_dir, best)
	tl.files.save_npz(net.all_params, name=net_name, sess=sess)
	tl.files.save_npz(net.all_params, name=best_name, sess=sess)
	print("[*] Saving checkpoints SUCCESS!")

# averages over array of numbers for smooth graphing
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

# load networks 
def load_networks(checkpoint_dir, sess, net, epoch, name = ''): 
	print "[*] Loading checkpoints..." 
	epoch = name + '_' + epoch + '.npz'
	net_name = os.path.join(checkpoint_dir,  epoch)
	
	if not (os.path.exists(net_name)):
		print "[!] Loading checkpoint failed!"
	else:
		net_loaded_params = tl.files.load_npz(name=net_name)
		
		tl.files.assign_params(sess, net_loaded_params, net)
		print "[*] Loading checkpoint SUCCESS!"
		
# plots and saves graphs 		
def render_graphs(save_dir,epoch, recon_loss, valid_loss, exact_valid_loss, name = 'depth'): 
	if len(recon_loss)>101: 
		valid_list = []
		for i in range(epoch+1): 
			valid_list.append((i+1)*len(recon_loss)//(epoch+1) )

		recon_loss = recon_loss[len(recon_loss)//2:]
		valid_list, valid_loss = valid_list[len(valid_list)//2:],valid_loss[len(valid_loss)//2:]

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



# comptue complete depth map from ranged prediction and upsampled low res image
def recover_depths(preds, ups, high, dis): 
	preds =  np.round_(preds*dis).reshape((-1,high,high)) 
	ups = np.array(ups).reshape((-1,high,high))

	for pred, up, i in zip(preds, ups, range(preds.shape[0])):
		pred = np.array(pred)
		pred  = up + pred # add to upsampled low resolution odm
		off = np.where(pred > high)  # set values which predict to high to be unoccupited -> 0        
		pred[off] = 0. 
		preds[i] = pred
	return preds

def recover_occupancy(preds, high, threshold = .5): 
	preds[np.where( preds >  threshold)] = 1.
	preds[np.where( preds <= threshold)] = 0.
	return preds.reshape((-1,high,high))






# reconvers full odm from occupancy and depth map
def recover_odm(depths, occs, ups, high, low, dis, side, threshold = 20):

	# recover exact occupancy and depths information and combine to creat the odms
	odms = recover_depths(depths, ups, high, dis)
	occs = recover_occupancy(occs, high)
	off = np.where(occs == 0 )
	odms[off] = 0

	# smoothing
	for i,odm in enumerate(odms):
		copy = np.array(odm)
		on = np.where(odm != 0)
		for x,y in zip(*on):
			window = odm[max(0,x-3):min(high-1, x+4),max(0,y-3):min(high-1,y+4)] #window
		
			considered =  np.where( window != 0)
			total = 0.
			count = 0.
			for a,b in zip(*considered): 
				if abs(window[a,b] -odm[x,y]) <threshold: 
					total += window[a,b]
					count +=1.
			
			if count > odm[x,y]:
				copy[x,y] = total/count
		odms[i] = np.round_(copy)


	return odms

# nearest neighbor upsampling of object
def upsample(obj, high, low): 
	ratio = high // low 
	big_obj = np.zeros((high,high,high))
	for i in range(low): 
		for j in range(low): 
			for k in range(low):
				big_obj[i*ratio: (i+1)*ratio, j*ratio:(j+1)*ratio, k*ratio:(k+1)*ratio] = obj[i,j,k]
	return big_obj


def apply_occupancy(obj, odms): 
	prediction = np.zeros(obj.shape)
	unoccupied = np.where(odms==0) 
	for x,y,z in zip(*unoccupied): 
		if x == 0 or x == 1: 
			prediction[y,z,:]-=0.25
		elif x == 2 or x == 3: 
			prediction[y,:,z]-=0.25
		else: 
			prediction[:,y,z]-=0.25
	ones = np.where(prediction>.6)
	zeros = np.where(prediction<.6)
	prediction[ones] = 1 
	prediction[zeros] = 0 
	return prediction 

def apply_depth(obj, high, odms):
	# recovering the origional data collected 
	off = np.where(odms == 0)
	for i in range(6): 
		face = np.array(odms[i])
		on  = np.where(face != 0)
		if i%2 == 0: 
			face[on] = 256 - face[on] + 2  # matches how gt obj is created 
		else:
			face[on] = face[on]  -1 
		print odms.shape, face.shape
		odms[i] = face 
	odms[off] = high * 2 

	prediction = np.array(obj)
	depths = np.where(odms<=high)
	for x,y,z in zip(*depths):
		pos = odms[x,y,z]
		if x == 0: 
				data[y,z,pos:high]-=.25
		if x == 1:
				data[y,z,0:pos]-=.25 
		if x == 2: 
			 	data[y,pos:high,z]-=.25
		elif x == 3: 
				data[y,0:pos,z]-=.25
		elif x == 4: 
				data[pos:high,y,z]-=.25
		elif x == 5: 
				data[0:pos,y, z]-=.25
				5
	ones = np.where(prediction>=1)
	zeros = np.where(prediction<1)
	prediction[ones] = 1 
	prediction[zeros] = 0 
	return prediction 


def produce( prediction, obj, small_obj):
	return
