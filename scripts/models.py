import tensorflow as tf  
import tensorlayer as tl
from tensorlayer.layers import *

def upscale(faces, scope_name = 'up', is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
   
    with tf.variable_scope(scope_name, reuse=reuse) as vs:
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
            recall = ElementwiseLayer([upscale, recall], tf.add, 'res5/%s' % i)
            upscale = recall

    

        upscale = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn2')
        upscale = BatchNormLayer(upscale, is_train=is_train, gamma_init=g_init, name='bn2')
        upscale = ElementwiseLayer([upscale, temp], tf.add, 'sum2')

        # up-sampling
        upscale = Conv2d(upscale, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn3')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='subpixel3')

        upscale = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn4')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='subpixel4')

        upscale = Conv2d(upscale, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='cnn5')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='subpixel5')

        upscale = Conv2d(upscale, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, name='cnnout')
        return upscale, upscale.outputs





