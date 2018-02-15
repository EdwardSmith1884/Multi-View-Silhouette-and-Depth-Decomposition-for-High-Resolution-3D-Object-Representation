import tensorflow as tf 
import keras.backend as K 
import os
import sys 
sys.path.insert(0, './scripts/')
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import random 
from tqdm import tqdm
from glob import glob
from voxel import * 
from utils import *
from models import *
from scipy import ndimage
from sklearn.metrics import mean_squared_error as mse
from scipy.ndimage.interpolation import rotate
import argparse

parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')
parser.add_argument('-n','--name', default='chair', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/voxels/chair/test', help ='The location for the depth maps.' )
parser.add_argument('-b','--batchsize', default=16, help ='The batch size.', type=int)
parser.add_argument('-depth','--depth', default='best', help ='Epoch from which to load the depth map predictor, if you want the best leave default.' )
parser.add_argument('-occ','--occ', default='best', help ='Epoch from which to load the occupancy map predictor, if you want the best leave default.' )
parser.add_argument('-dis','--distance', default=70, help ='The range in which distances will be predicted.', type=int)
parser.add_argument('-high', default= 256, help='The size of the high dimension objects.', type= int)
parser.add_argument('-low', default= 32, help='The size of the low dimension object.', type= int)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
batchsize      = args.batchsize
high           = args.high 
low            = args.low
distance 	   = args.distance
ratio 		   = high // low


#######inputs##########
scope_depth = 'depth'
scope_occ   = 'occupancy'
images_low  = tf.placeholder(tf.float32, [batchsize, low, low, 1], name='images_low') # low res odm input
side        = tf.placeholder(tf.float32, [batchsize, low, low, 1], name='side') # the side being considered
combined    = tf.concat((images_low, side), axis = 3)

########## network computations #######################
net_depth, depth_pred      = upscale(combined, ratio, scope = scope_depth, is_train=False, reuse = False)
net_occ, occ_pred 		   = upscale(combined, ratio, scope = scope_occ, is_train=False, reuse = False)
net_depth.print_params(False)
net_occ.print_params(False)

##### computing #######
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.999)
sess.run(tf.global_variables_initializer())


load_networks(checkpoint_dir, sess, net_depth, args.depth, name ='depth')
load_networks(checkpoint_dir, sess, net_occ, args.occ, name = 'occ')
files = grab_files(args.data)


for idx in xrange(0, len(files)/args.batchsize):
	odms = []
	cur_files = files[idx*batchsize:(idx+1)*batchsize]
	# loops over all sides
	for k in range(6): 
		batch, _    = make_batch(cur_files, high, low, side = k)
		depths, occs = sess.run([depth_pred,occ_pred], feed_dict={images_low:batch['low'], side: batch['side']})     
		odms.append(recover_odms(depths, occs, batch['low_up'], high, low, distance, threshold = 1.5*high//low)) # combining depths and occupancy maps to recover full odms 
	
	# combining information 
	odms =  zip(odms[0], odms[1], odms[2], odms[3], odms[4], odms[5])
	objs, small_objs = make_objs(cur_files) # loading the ground truth object and input object
	batch_predictions = zip(odms, objs,  small_objs)
	
	print 'finished loading a batch'
	for odm, obj, small_obj  in tqdm(batch_predictions):


		small_obj = upsample(small_obj, high, low)
		prediction = apply_occupancy(np.array(small_obj), np.array(odm))
		prediction = apply_depth(np.array(prediction),np.array(odm),high,)
		evaluate_SR(prediction, obj, small_obj)
		
		

		
		 







			
				
			
