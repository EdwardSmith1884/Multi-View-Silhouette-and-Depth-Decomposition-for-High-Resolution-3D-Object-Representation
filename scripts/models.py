import tensorflow as tf  
import tensorlayer as tl
from tensorlayer.layers import *

def upscale(faces, scope_name = 'up', is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
   
    with tf.variable_scope(scope_name, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        face = InputLayer(faces, name='in/face')
        upscale = Conv2d(face, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c_face')
    
        temp = upscale
        # residual blocks
        for i in range(16):
            recall = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,  name='n64s1/c1/%s' % i)
            recall = BatchNormLayer(recall, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            recall = Conv2d(recall, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c2/%s' % i)
            recall = BatchNormLayer(recall, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            recall = ElementwiseLayer([upscale, recall], tf.add, 'b_residual_add/%s' % i)
            upscale = recall

    

        upscale = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c/m')
        upscale = BatchNormLayer(upscale, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        upscale = ElementwiseLayer([upscale, temp], tf.add, 'add3')

        # up-sampling
        upscale = Conv2d(upscale, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='pixelshufflerx2/1')

        upscale = Conv2d(upscale, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='pixelshufflerx2/2')

        upscale = Conv2d(upscale, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2ascasc')
        upscale = SubpixelConv2d(upscale, scale=2, act=tf.nn.relu, name='pixelshufflerxcasc2/2')

        upscale = Conv2d(upscale, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, name='out')
        return upscale, upscale.outputs





