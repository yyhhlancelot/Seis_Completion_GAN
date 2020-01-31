import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random

def batch_multiply(batch, mask):
    shape = batch.get_shape().as_list()
    subsamples = []
    for i in range(shape[0]):
        subsamples.append(tf.multiply(batch[i, :, :, :], mask[i, :, :, :]))
        
    return tf.stack(subsamples, axis = 0) # 因为被采样的面全部在一个List里面，还不是一个张量
    
def generate_mask(batch_size, img_size, sampling_rate): 
    ''' lose random trace 
    generate a tensor of seismic data whose every slice has same sampling rate'''
    
    mask_mat = np.ones([batch_size, img_size, img_size, 1])

    loss_num = np.int(img_size * sampling_rate)
    
    for i in range(batch_size):
        np.set_printoptions(threshold = np.inf)
        loss_list = random.sample(range(1, 32), loss_num)
        
        for j in range(loss_num): 
            
            mask_mat[i, :, loss_list[j], :] = np.zeros([img_size, 1])
            
    mask = mask_mat
    
    return mask
    
    
def batchnorm(bottom, n_reference = 0, epsilon=1e-3, decay=0.999, name=None):
    """ virtual batch normalization (poor man's version)
    the first half is the true batch, the second half is the reference batch.
    When num_reference = 0, it is just typical batch normalization.  
    To use virtual batch normalization in test phase, "update_popmean.py" needed to be executed first 
    (in order to store the mean and variance of the reference batch into pop_mean and pop_variance of batchnorm.)
    """

    batch_size = bottom.get_shape().as_list()[0]
    inst_size = batch_size - n_reference
    # inst_size = batch_size
    instance_weight = np.ones([batch_size])

    if inst_size > 0:
        reference_weight = 1.0 - (1.0 / ( n_reference + 1.0))
        instance_weight[0 : inst_size] = 1.0 - reference_weight
        # instance_weight[0 : batch_size] = 1.0
        instance_weight[inst_size:] = reference_weight
    else:
        decay = 0.0

    return slim.batch_norm(bottom, activation_fn=None, is_training=True, reuse = tf.AUTO_REUSE, decay=decay, scale=True, scope=name, batch_weights=instance_weight)
    

    
def new_conv_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, bias=True, name=None):
    """
    typical convolution layer using stride to down-sample
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            "W",
            shape=filter_shape,
            initializer=tf.truncated_normal_initializer(0., 0.005))
        conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))
            output = activation(tf.nn.bias_add(conv, b))
        else:
            output = activation(conv)

    return output
    
def new_fc_layer(bottom, output_size, name=None, bias=True, reuse = None):
    """
    fully connected layer
    """
    shape = bottom.get_shape().as_list()
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.truncated_normal_initializer(0., 0.005))
        if bias == True:
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)
        else:
            fc = tf.matmul(x, w)

    return fc
def channel_wise_fc_layer(bottom, name, bias=True, reuse = None):
    """
    channel-wise fully connected layer
    """
    _, width, height, n_feat_map = bottom.get_shape().as_list()
    input_reshape = tf.reshape( bottom, [-1, width*height, n_feat_map] )  # order='C'
    input_transpose = tf.transpose( input_reshape, [2,0,1] )  # n_feat_map * batch * d

    with tf.variable_scope(name, reuse = reuse):
        W = tf.get_variable(
                "W",
                shape=[n_feat_map,width*height, width*height], # n_feat_map * d * d_filter
                initializer=tf.truncated_normal_initializer(0., 0.005))
        # output = tf.batch_matmul(input_transpose, W)
        output = tf.matmul(input_transpose, W) 
        # matrix multiply

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=width*height,
                initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)

    output_transpose = tf.transpose(output, [1,2,0])  # batch * d_filter * n_feat_map
    output_reshape = tf.reshape( output_transpose, [-1, width, height, n_feat_map] )
    return output_reshape
    
    
    
def conv_layer_new(input, filter_shape, strides, name = None):

    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_shape, initializer = tf. truncated_normal_initializer(0., 0.005))
        # b = tf.get_variable("b", filter_shape[-1], initializer = tf.constant_initializer(0.))
        b = tf.get_variable("b", filter_shape[-1], initializer = tf.truncated_normal_initializer(0.))
        
        conv = tf.nn.conv2d(input, W, strides, padding = 'SAME')
        
        output = tf.nn.bias_add(conv, b)
        
        output = tf.identity(output)
    
    return output

def dilated_conv_layer_new(input, filter_shape, rate, name = None):
    
    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_shape, initializer = tf.truncated_normal_initializer(0., 0.005))
        b = tf.get_variable("b", filter_shape[-1], initializer = tf.truncated_normal_initializer(0.))
        
        conv = tf.nn.atrous_conv2d(input, W, rate, padding = 'SAME')
        
        output = tf.nn.bias_add(conv, b)
        
        output = tf.identity(output)
        
    return output

def deconv_layer_new(input, filter_shape, output_shape, strides, name = None):
    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_shape, initializer = tf. truncated_normal_initializer(0., 0.005))
        
        # b = tf.get_variable("b", filter_shape[-2], initializer = tf.constant_initializer(0.))
        # b = tf.get_variable("b", filter_shape[-2], initializer = tf.truncated_normal_initializer(0.))
        b = tf.get_variable("b", output_shape[-1], initializer = tf.truncated_normal_initializer(0.))
        
        conv = tf.nn.conv2d_transpose(input, W, output_shape, strides, padding = 'SAME')
        
        output = tf.nn.bias_add(conv, b)
        
        output = tf.identity(output)
    
    return output

def np_combine_batch(inst, ref):
    out = np.concatenate([inst, ref], axis = 0)
    return out