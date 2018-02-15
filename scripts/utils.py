import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
import scipy.io as sio
import random
import tensorlayer as tl
from PIL import Image

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

def grab_images(image_dir, voxel_dir): 
	files = []
	pattern  = "*.png"
	image_dir+='/'
	voxel_dir+='/'
	for dir,_,_ in os.walk(image_dir):
		files.extend(glob(os.path.join(dir,pattern)))
	voxels = [ v.split('/')[-1].split('.')[0].split('_')[-1] for v in glob(voxel_dir + 'full_object*')]
	temp = []
	for f in files: 
		if f.split('/')[-2] not in voxels: continue
		temp.append(f)

	return temp

def make_batch_images(file_batch, voxel_dir):
	start_time = time.time()
	voxel_dir+='/'
	models = []
	images = []
	for i,fil in enumerate(file_batch):  
		split = fil.split('/')[-1].split('_')[0]
		models.append(sio.loadmat(voxel_dir+ 'full_object_' +split +'.mat')['low_model'])
		img = Image.open(fil)
		images.append(np.asarray(img,dtype='uint8'))
	models = np.array(models)
	images = np.array(images)
	
	return models, images, start_time

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
		valid_list, valid_loss = valid_list[:len(valid_loss)//2],valid_loss[len(valid_loss)//2:][:len(valid_loss)//2]

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



def smoothing(odms, high, threshold): 
	return odms 
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


# reconvers full odm from occupancy and depth map
def recover_odms(depths, occs, ups, high, low, dis, threshold = 20):
	# recover exact occupancy and depths information and combine to creat the odms
	odms = recover_depths(depths, ups, high, dis)
	occs = recover_occupancy(occs, high)
	off = np.where(occs == 0 )
	odms[off] = 0
	odms = smoothing(np.array(odms), high, threshold)
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



def evaluate_voxel_prediction(prediction,gt):
	"""  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
	intersection = np.sum(np.logical_and(prediction,gt))
	
	union = np.sum(np.logical_or(prediction,gt))
	IoU = float(intersection) / float(union)
	return IoU


# extracts odms from an object 
def odm(data): 
	dim = data.shape[0]
	a,b,c = np.where(data == 1)
	large = int(dim *1.5)
	big_list = [[[[-1,large]for j in range(dim)] for i in range(dim)] for k in range(3)]
	# over the whole object extract for each face the first and last occurance of a voxel at each pixel
	# we take highest for convinience
	for i,j,k in zip(a,b,c):
		big_list[0][i][j][0] = (max(k,big_list[0][i][j][0]))
		big_list[0][i][j][1] = (min(k,big_list[0][i][j][1]))
		big_list[1][i][k][0] = (max(j,big_list[1][i][k][0]))
		big_list[1][i][k][1] = (min(j,big_list[1][i][k][1]))
		big_list[2][j][k][0] = (max(i,big_list[2][j][k][0]))
		big_list[2][j][k][1] = (min(i,big_list[2][j][k][1]))
	faces = np.zeros((6,dim,dim)) # will hold odms 
	for i in range(dim): 
		for j in range(dim): 
			faces[0,i,j] =   1 + dim - big_list[0][i][j][0]        	   if    big_list[0][i][j][0]   > -1 else 0
			# we subtract from the dimension as we computed the last occurance for half of the faces 
			# we add 1 as a value of 1 indicates the first voxel is filled, and 0 that no voxle is present along that dimension 
			faces[1,i,j] =   1 + big_list[0][i][j][1]        		   if    big_list[0][i][j][1]   < large else 0 
			faces[2,i,j] =   1 + dim - big_list[1][i][j][0]            if    big_list[1][i][j][0]   > -1 else 0
			faces[3,i,j] =   1 + big_list[1][i][j][1]        		   if    big_list[1][i][j][1]   < large else 0
			faces[4,i,j] =   1 + dim - big_list[2][i][j][0]            if    big_list[2][i][j][0]   > -1 else 0
			faces[5,i,j] =   1 + big_list[2][i][j][1]         		   if    big_list[2][i][j][1]   < large else 0
	return faces

def extract_odms(models, factor, high, low): 
	low_ups = []
	views = np.ones((6, low,low, 1))
	odms = []
	split = models.shape[0]/factor
	for i in range(6): 
		views[i] = i * views[i]
	for model in models: 
		model = model.reshape((low, low ,low))
		faces = odm(model)
		low_ups += [ extract_low_up(face, high, low) for face in faces]
		faces = np.concatenate((faces.reshape((6, low , low , 1)), views), axis = 3 ) 
		odms += list(faces)
	return np.array([odms[i*split:(i+1)*split] for i in range(factor*6)]), np.array([low_ups[i*split:(i+1)*split] for i in range(factor*6)])


def extract_low_up(odm, high, low ): 
	ratio  = high // low 
	a,b = np.where(odm > 0) 
	up = np.zeros((high,high,1))
	for x,y in zip(a,b): 
		up[ratio*x:ratio*(x+1), ratio*y:ratio*(y+1)] = (odm[x,y] -1) *ratio  +1 #  uscale low resolution odm 
	return up.reshape((256,256))
		
def evaluate(instance): 
	return 




def voxel_exist(voxels, x,y,z):
	if x < 0 or y < 0 or z < 0 or x >= voxels.shape[0] or y >= voxels.shape[1] or z >= voxels.shape[2]:
		return False
	else :
		return voxels[x,y,z] > .3 

def max_connected(models):
	distance = 1
	for i,voxels in enumerate(models): 
		max_component = np.zeros(voxels.shape, dtype=bool)
		voxels = np.copy(voxels)
		for startx in xrange(voxels.shape[0]):
			for starty in xrange(voxels.shape[1]):
				for startz in xrange(voxels.shape[2]):
					if not voxels[startx,starty,startz]:
						continue
					# start a new component
					component = np.zeros(voxels.shape, dtype=bool)
					stack = [[startx,starty,startz]]
					component[startx,starty,startz] = True
					voxels[startx,starty,startz] = False
					while len(stack) > 0:
						x,y,z = stack.pop()
						for i in xrange(x-distance, x+distance + 1):
							for j in xrange(y-distance, y+distance + 1):
								for k in xrange(z-distance, z+distance + 1):
									if (i-x)**2+(j-y)**2+(k-z)**2 > distance * distance:
										continue
									if voxel_exist(voxels, i,j,k):
										voxels[i,j,k] = False
										component[i,j,k] = True
										stack.append([i,j,k])
					if component.sum() > max_component.sum():
						max_component = component
		models[i] = max_component
	return models