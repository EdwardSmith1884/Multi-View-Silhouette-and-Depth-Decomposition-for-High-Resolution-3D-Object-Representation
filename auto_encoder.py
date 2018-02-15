import os
import sys 
sys.path.insert(0, './scripts/') 
import numpy as np
import tensorflow as tf 
import random 
from glob import glob
from utils import *
from models import *
import argparse

parser = argparse.ArgumentParser(description='Auto Encoder for 3D object reconstruction from images')
parser.add_argument('-n','--name', default='plane', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-ensemble', default='0', help ='The ensemble experiment number being perfomed, you should do up to five')
parser.add_argument('-d','--data', default='data/voxels/plane/train', help ='The location for the object voxel models.' )
parser.add_argument('-dv','--datav', default='data/voxels/plane/valid', help ='The location for the object voxel models.' )
parser.add_argument('-i','--images', default='data/images/plane/train', help ='The location for the images.' )
parser.add_argument('-iv','--images_valid', default='data/images/plane/valid', help ='The location for the valid images.' )
parser.add_argument('-e','--epochs', default=1500, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=32, help ='The batch size.', type=int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= 'best', help='The epoch to number to be loaded from, if you just want the best, leave as default.', type=str)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'

save_dir =  "plots/" + args.name +'/'
random.seed(0)
batchsize = args.batchsize
valid_length = 3 # number of batches to use in validation set 

######### make directories ############################
make_directories(checkpoint_dir,save_dir)

####### inputs  ###################
scope = 'reconstruction'
images = tf.placeholder(tf.float32, [args.batchsize, 128, 128, 3], name='images')
models = tf.placeholder(tf.float32, [args.batchsize, 32, 32, 32] , name='real_models')

########## network computations #######################
net, pred     = auto_encoder(images, scope=scope, is_train=True, reuse = False)
_, pred_valid = auto_encoder(images, scope=scope, is_train=False, reuse = True)

mse       = tf.reduce_mean(tf.square(models-pred))
mae       = tf.reduce_mean(tf.abs(models-pred))
loss      = mse + .001*mae
real_loss = tf.reduce_mean(tf.square(models-pred_valid))

############ Optimization #############
vars = tl.layers.get_variables_with_name(scope, True, True)
net.print_params(False)
optim = tf.train.RMSPropOptimizer(learning_rate = 1e-3).minimize(loss, var_list=vars)

####### Training ################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session()
tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.9980)
sess.run(tf.global_variables_initializer())

##### load checkpoints ####################
if args.load: 
	load_networks(checkpoint_dir, sess, net, args.load_epoch, name = (scope + args.ensemble))
recon_loss, valid_IoU, valid_loss, max_IoU = [],[],[], 0 

######## make files and models ##################33
files= grab_images(args.images, args.data)
valid = grab_images(args.images_valid, args.datav)
random.shuffle(valid)
valid = valid[:3*batchsize]
valid_models, valid_images, _ = make_batch_images(valid, args.datav)

if args.load: 
	try: 
		start = int(args.load_epoch) + 1 
	except: 
		start = 0 
else: 
	start = 0 

########### train #################
for epoch in range(start, args.epochs):
	random.shuffle(files)
	for idx in xrange(0, len(files)/args.batchsize):
		batch = random.sample(files, args.batchsize)
		batch_models, batch_images, start_time = make_batch_images(batch, args.data)
		
		batch_loss,_ = sess.run([mse, optim], feed_dict={images: batch_images, models:batch_models })    
		recon_loss.append(batch_loss)
		print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.4f, VALID: %.4f" % (epoch, 
			args.epochs, idx, len(files)/batchsize, time.time() - start_time, batch_loss, max_IoU))

	########## check validation #############
	valid_losses = 0.
	IoU = 0. 
	for i in range(int(len(valid)/args.batchsize)):
		v_images = valid_images[i*args.batchsize: (i+1)*args.batchsize]
		gt_models = valid_models[i*args.batchsize: (i+1)*args.batchsize]
		v_models, temp_loss = sess.run([pred, real_loss], feed_dict={images:v_images, models: gt_models})
		valid_losses += temp_loss/float(valid_length)
		v_models[np.where(v_models >.45)] = 1 
		v_models[np.where(v_models<.45)] = 0 
		for m, gt in zip(v_models,gt_models):
			IoU += evaluate_voxel_prediction(m,gt)

	valid_loss.append(valid_losses)
	IoU  = IoU / float(valid_length * batchsize)
	valid_IoU.append(IoU)

	test_valid = max_IoU
	max_IoU = max(IoU, max_IoU)
	if test_valid != max_IoU: 
		save_networks(checkpoint_dir, sess, net, name=(scope + args.ensemble ), epoch = str(epoch))

	######### save graphs ###########
	render_graphs(save_dir, epoch, recon_loss, valid_loss, valid_IoU, name = scope ) 

	
