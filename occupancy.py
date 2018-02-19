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

parser = argparse.ArgumentParser(description='Occupancy map predictor for 3D Super Resolution')
parser.add_argument('-n','--name', default='chair', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='data/voxels/chair/train', help ='The location for the training voxel data.' )
parser.add_argument('-v','--valid', default='data/voxels/chair/valid', help ='The location for the validation voxel data.' )
parser.add_argument('-e','--epochs', default= 250, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=64, help ='The batch size.', type=int)
parser.add_argument('-high', default= 256, help='The size of the high dimension objects.', type= int)
parser.add_argument('-low', default= 32, help='The size of the low dimension object.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= 'best', help='The epoch to number to be loaded from, if you just want the best, leave as default.', type=str)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "plots/" + args.name +'/'
high = args.high 
low = args.low
ratio = high // low 
batchsize =args.batchsize
valid_length = 3 
lr = 1e-4


######### make directories ###########################
make_directories(checkpoint_dir,save_dir)

####### inputs  ###################
scope = 'occ'
images_high = tf.placeholder(tf.float32, [batchsize, high, high, 1], name='images_high') # high res odm ground truth
images_low  = tf.placeholder(tf.float32, [batchsize, low, low, 1], name='images_low') # low res odm input
side        = tf.placeholder(tf.float32, [batchsize, low, low, 1], name='side') # the side being considered
combined    = tf.concat((images_low, side), axis = 3)
########## network computations #######################

net, pred      = upscale(combined, ratio, scope = scope, is_train=True, reuse = False)
_, pred_valid  = upscale(combined, ratio, scope = scope, is_train=False, reuse = True)
loss           = tl.cost.mean_squared_error(images_high, pred, is_mean=True)
real_loss      = tl.cost.mean_squared_error(images_high, pred_valid, is_mean=True)

############ Optimization #############

net.print_params(False)
variables = tl.layers.get_variables_with_name(scope, True, True)
optim = tf.train.AdamOptimizer( learning_rate = lr, beta1=0.5, beta2=0.9).minimize(loss, var_list=variables )

####### Training ################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.999)
sess.run(tf.global_variables_initializer())

####### load checkpoints and files ###############
if args.load: 
	load_networks(checkpoint_dir, sess, net, args.load_epoch, name = (scope))
recon_loss, exact_valid_loss, valid_loss = [],[],[]
files = grab_files(args.data)
valid = grab_files(args.valid)[:valid_length*batchsize]
valid, _  = make_batch(valid, high, low, occupancy = True, valid = True)


######### training ##############
if args.load: 
	try: 
		start = int(args.load_epoch) + 1 
	except: 
		start = 0 
else: 
	start = 0 
min_recon = 100000. 
for epoch in range(start, args.epochs):
	for idx in xrange(len(files)/ batchsize/10):
		batch = random.sample(files, batchsize)
		batch, start_time = make_batch(batch, high, low, args.data, occupancy = True)

		batch_loss, _ = sess.run( [loss, optim], feed_dict={images_high:batch['high'], images_low: batch['low'], side : batch['side']})  
		if epoch > 0:  
			recon_loss.append(batch_loss)
		print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.4f, VALID: %.4f" % (epoch, 
			args.epochs, idx, len(files)/batchsize/10, time.time() - start_time, batch_loss, min_recon))
		sys.stdout.flush()

  
	
   	######### check validation ##########
	reconstruction = np.zeros((0,high,high))
	v_loss = 0. 
	for i in range(valid_length): 
		valid_images_low    = valid['low'][i*batchsize:(i+1)*batchsize]
		valid_side          = valid['side'][i*batchsize:(i+1)*batchsize]
		valid_images_high   = valid['high'][i*batchsize:(i+1)*batchsize]
		temp_loss, temp_recon = sess.run([real_loss, pred], feed_dict={ images_low: valid_images_low ,side: valid_side, images_high: valid_images_high})        
		v_loss += temp_loss/float(valid_length)
		reconstruction = np.concatenate((reconstruction, temp_recon.reshape((-1,high,high))), axis =0)
	
	
	ground_truth = np.array((valid['high']) *1.).reshape((-1, high, high))
	reconstruction = recover_occupancy(reconstruction, high)
	mean_squared_error = np.mean(np.square(reconstruction - ground_truth))

	exact_valid_loss.append(mean_squared_error)
	valid_loss.append(v_loss)
	
	####### save networks #########
	test_valid = min_recon
	min_recon = min(mean_squared_error, min_recon)
	if test_valid != min_recon: 
		save_networks(checkpoint_dir, sess, net, name = scope, epoch =str(epoch))

	####### save graphs #####
	render_graphs(save_dir, epoch, recon_loss, valid_loss, exact_valid_loss, name = scope ) 

