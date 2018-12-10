import tensorflow as tf  
import tensorlayer as tl
from tensorlayer.layers import *
from math import log 

def upscale(faces, ratio, scope = 'up', is_train=True, reuse=False):
	w_init = tf.random_normal_initializer(stddev=0.02)
	g_init = tf.random_normal_initializer(1., 0.02)
   
	with tf.variable_scope(scope, reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)

		face = InputLayer(faces, name='input')
		upscale = Conv2d(face, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='cnn1')
	
		temp = upscale
		# residual blocks
		for i in range(16):
			recall = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,  name='res1/%s' % i)
			recall = BatchNormLayer(recall, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='res2/%s' % i)
			recall = Conv2d(recall, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='res3/%s' % i)
			recall = BatchNormLayer(recall, is_train=is_train, gamma_init=g_init, name='res4/%s' % i)
			recall = ElementwiseLayer([upscale, recall], tf.add, name= 'res5/%s' % i)
			upscale = recall

	

		upscale = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn2')
		upscale = BatchNormLayer(upscale, is_train=is_train, gamma_init=g_init, name='bn2')
		upscale = ElementwiseLayer([upscale, temp], tf.add, name= 'sum2')

		# up-sampling
		for i in range(int(log(ratio,2))): 
			upscale = Conv2d(upscale, 256/pow(2,i) , (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn/%s' %(i+1))
			upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='subpixel/%s' %i)

		upscale = Conv2d(upscale, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, name='cnnout')
		return upscale, upscale.outputs





def auto_encoder(inputs, scope = 'reconstruction', is_train=True, reuse=False):
	
	init = tf.random_normal_initializer(stddev=0.02)
	batchsize = tf.shape(inputs)[0]
	with tf.variable_scope("reconstruction", reuse=reuse):
		tl.layers.set_name_reuse(reuse)
		encode = tl.layers.InputLayer(inputs, name='encode/in')
		########## NETOWRK PARAMETERS ###############
		kernal_depth = [3,64,128,256,512,400]
		kernal_dim = [7,5,5,5,8]
		stride = [2,2,2,2,1] 
		########### ENCODER ###################
		for i in range(5): 
			#  downsample to resolution while increasing kernal depth 
			encode = tl.layers.Conv2dLayer(encode, shape=[kernal_dim[0], kernal_dim[0], kernal_depth[i], kernal_depth[i+1]],
						W_init = init,  strides=[1, stride[i], stride[i], 1], name='encode/conv/%s' % i )
			encode = tl.layers.BatchNormLayer(encode, is_train=is_train, name='encode/batch_norm/%s' % i)
			encode.outputs = tl.activation.leaky_relu(encode.outputs, alpha=0.2, name='encode/lrelu/%s' % i )
		
		encode = FlattenLayer(encode, name='encode/flatten')
		encode = tl.layers.DenseLayer(encode, n_units= 256, act=tf.identity,  W_init = init, name='measns')


		############ NETWORK PARAMETERS ###########
		kernal_depth = [128,128,64,32,8]
		kernal_dim = [4,8,16,32]
		############ DECODER #################
		decode = tl.layers.DenseLayer(encode, n_units = 1024, W_init = init, act = tf.identity, name='decode/dense')
		decode = tl.layers.ReshapeLayer(decode, shape = [batchsize, 2,2,2, 128], name='decode/reshape/0')
		for i in range(4): 
			
			# deconv layer to upsample resolution by a factor of two 
			decode = tl.layers.DeConv3dLayer(decode,
				shape = [4, 4, 4, kernal_depth[i +1], kernal_depth[i]],
				output_shape = [batchsize, kernal_dim[i], kernal_dim[i], kernal_dim[i], kernal_depth[i +1]],
				strides=[1, 2, 2, 2, 1], W_init = init, act=tf.identity, name='decode/deconv/%s' % i)
			decode = tl.layers.BatchNormLayer(decode, is_train=is_train, name='encodeer/batch_norm/%s' % i)

			decode.outputs = tf.nn.relu(decode.outputs, name='decode/relu/%s' % i)

			#convlayer
			# decode =  tl.layers.Conv3dLayer(decode,
			# 	shape=[3, 3, 3, kernal_depth[i+1], kernal_depth[i+1]],
			# 	W_init = init, strides=[1, 1, 1, 1, 1], name= 'decode/conv/%s' % i)
			# decode = tl.layers.BatchNormLayer(decode, is_train=is_train, name='encoder/batch_norm/%s' % i)
			# decode.outputs = tf.nn.relu(decode.outputs, name='decode/relu/%s' %i)

		# conv layer to set kenel depth to 1 
		decode =  tl.layers.Conv3dLayer(decode,
								   shape=[3, 3, 3, kernal_depth[-1], 1],
								   W_init = tf.random_normal_initializer(stddev=0.02),
								   strides=[1, 1, 1, 1, 1], name='decode/conv/4')
		
		decode = tl.layers.ReshapeLayer(decode, shape = [batchsize,32,32,32], name='decode/reshape/1')
		decode.outputs = tf.nn.sigmoid(decode.outputs)
		return decode, decode.outputs



		
