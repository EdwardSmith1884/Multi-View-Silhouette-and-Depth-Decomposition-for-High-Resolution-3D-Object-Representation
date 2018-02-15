import tensorflow as tf 
import os
import sys
sys.path.insert(0, './scripts/') 
import numpy as np
import random 
from glob import glob
from utils import *
from models import *
from sklearn.metrics import mean_squared_error as mse
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Evaluation of 3D object reconstruction at high resolutions. ')
parser.add_argument('-n','--name', default='plane', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/voxels/plane/test', help ='The location for the depth maps.' )
parser.add_argument('-i','--images', default='data/images/plane/test', help ='The location for the images.' )
parser.add_argument('-b','--batchsize', default=8, help ='The batch size.', type=int)
parser.add_argument('-dis','--distance', default=70, help ='The range in which distances will be predicted.', type=int)
parser.add_argument('-high', default= 256, help='The size of the high dimension objects.', type= int)
parser.add_argument('-low', default= 32, help='The size of the low dimension object.', type= int)

args = parser.parse_args()
batchsize = args.batchsize
name = args.name
high = args.high
low = args.low
ratio = high // low 
distance = args.distance
random.seed(0)

print 'We assume here that depth.py, occupancy.py and auto_encoder.py have all be called already, and so trained models exist'
print 'The best model from these trainings will be selected automatically and used to train'

####### inputs  ###################
images  = tf.placeholder(tf.float32, [4*batchsize, 128, 128, 3], name='images')
odms    = tf.placeholder(tf.float32, [batchsize, 32, 32, 2], name='odms')
########## network computations #######################


net_auto, model        = auto_encoder(images, scope = 'recon', is_train=False , reuse = False)
net_depth, depth_pred  = upscale(odms,ratio, scope = 'depth', is_train = True, reuse = False)
net_occ, occ_pred      = upscale(odms,ratio, scope = 'occ', is_train = True, reuse = False)

net_auto.print_params(False)
net_depth.print_params(False)
net_occ.print_params(False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
sess.run(tf.global_variables_initializer())




net_depth_loaded_params = tl.files.load_npz(name='checkpoint/' + name + '/depth_best.npz')
tl.files.assign_params(sess, net_depth_loaded_params, net_depth)
net_occ_loaded_params = tl.files.load_npz(name='checkpoint/' + name + '/occ_best.npz')
tl.files.assign_params(sess, net_occ_loaded_params, net_occ)

print '----------------------------------------------------------'
print 'We assume here that depth.py, occupancy.py and auto_encoder.py have all be called already, and so trained models exist'
print 'The best model from these trainings will be selected automatically and used to train'

ae_checkpoints = glob('checkpoint/' + name + '/reconstruction*_best.npz') # getting all reconstruction experiments 
ae_count = len(ae_checkpoints)
ae_limit = 1+ (ae_count //2) #minimum number models who need to indicate a voxel should be filled to predict a filled voxel during the ensemble 
print str(ae_count) + ' auto_encoder experiment checkpoints were discovered'
print 'If you wish to include more in the ensemble please run: auto_encoder.py -ensemble K, where K is some new experiment name'
print '----------------------------------------------------------'

files= grab_images(args.images, args.data) 
random.shuffle(files)

batch = []
for f in files:
	batch.append(f)
	if len(batch) == batchsize*4:

		######## ensemble to predict low resolution model ########
		batch_models, batch_images,_ = make_batch_images(batch, args.data)
		pred_models = np.zeros((batchsize*4, low, low, low))
		for j in range(ae_count):
			net_auto_loaded_params = tl.files.load_npz(name=ae_checkpoints[j])
			tl.files.assign_params(sess, net_auto_loaded_params, net_auto)
			temp_models = sess.run(model, feed_dict={images: batch_images})
			
			temp_models[np.where(temp_models>.45)] = 1 # threshold to predict voxels 
			temp_models[np.where(temp_models<=.45)] = 0 
			pred_models += temp_models
		      
		pred_models[np.where(pred_models<ae_limit )] = 0 # set voxel positions to off if not enough models in ensemble predict them as on 
		pred_models[np.where(pred_models != 0)] = 1 # set all the rest to on 
		pred_models = max_connected(pred_models)# removing stray voxels, increases attractiveness, but not accuracy usually 

		######## extact high-res odms from low res model #############
		odm_batches, low_up_batches = extract_odms(pred_models, 4, high, low ) # exract low res odms and upsampled low res odms from models 
		pred_odms = []
		for odm_batch, low_up_batches in zip(odm_batches, low_up_batches):
			pred_depths, pred_occs = sess.run([depth_pred, occ_pred], feed_dict= {odms: odm_batch}) # predicting depth and occupancy maps 
			pred_odms += list(recover_odms(pred_depths,pred_occs,low_up_batches, high, low, distance, threshold = (1.5*high//low))) # recover complete odms from depth and occupancy maps 
		
		pred_odms = [ pred_odms[i*6:(i+1)*6] for i in range(len(pred_odms)/6)] # combining odms from the same model 

		####### evaluate the models ############	
		combined_information = zip(pred_models, pred_odms, batch)
		for instance in tqdm(combined_information):			
			evaluate(instance) #evaluate each model -> viewing only
			print 'finished'
			exit()

			

