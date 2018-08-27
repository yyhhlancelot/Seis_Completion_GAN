import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy as sp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import timeit
from smooth_stream import SmoothStream
from multiprocessing import Process, Queue
import load_mnist as load_dataset
from noise import add_noise
import argparse

def batchnorm(bottom, epsilon=1e-3, decay=0.999, name=None):
    """ virtual batch normalization (poor man's version)
    the first half is the true batch, the second half is the reference batch.
    When num_reference = 0, it is just typical batch normalization.  
    To use virtual batch normalization in test phase, "update_popmean.py" needed to be executed first 
    (in order to store the mean and variance of the reference batch into pop_mean and pop_variance of batchnorm.)
    """

    batch_size = bottom.get_shape().as_list()[0]
    # inst_size = batch_size - num_reference
    inst_size = batch_size
    instance_weight = np.ones([batch_size])

    if inst_size > 0:
        # reference_weight = 1.0 - (1.0 / ( num_reference + 1.0))
        # instance_weight[0:inst_size] = 1.0 - reference_weight
        instance_weight[0 : batch_size] = 1.0
        # instance_weight[inst_size:] = reference_weight
    else:
        decay = 0.0

    return slim.batch_norm(bottom, activation_fn=None, is_training=True, reuse = tf.AUTO_REUSE, decay=decay, scale=True, scope=name, batch_weights=instance_weight)
    
def bottleneck(input, is_train, n_reference, channel_compress_ratio=4, stride=1, bias=True, name=None):
    """
    building block for creating residual net
    """
    input_shape = input.get_shape().as_list()

    if stride is not 1:
        output_channel = input_shape[3] * 2
    else:
        output_channel = input_shape[3]

    bottleneck_channel = output_channel / channel_compress_ratio

    with tf.variable_scope(name):
        bn1 = tf.nn.elu(batchnorm(input, is_train, n_reference, name='bn1'))
        # shortcut
        if stride is not 1:
            shortcut = new_conv_layer(bn1, [1,1,input_shape[3],output_channel], stride=stride, bias=bias, name="conv_sc" )
        else:
            shortcut = input

        # bottleneck_channel
        conv1 = new_conv_layer(bn1, [1,1,input_shape[3],bottleneck_channel], stride=stride, bias=bias, name="conv1" )
        bn2 = tf.nn.elu(batchnorm(conv1, is_train, n_reference, name='bn2'))
        conv2 = new_conv_layer(bn2, [3,3,bottleneck_channel,bottleneck_channel], stride=1, bias=bias, name="conv2" )
        bn3 = tf.nn.elu(batchnorm(conv2, is_train, n_reference, name='bn3'))
        conv3 = new_conv_layer(bn3, [1,1,bottleneck_channel,output_channel], stride=1, bias=bias, name="conv3" )

    return shortcut+conv3
    
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
    
def add_bottleneck_module(input, is_train, nBlocks, n_reference, channel_compress_ratio=4, bias=True, name=None):

    with tf.variable_scope(name):
        # the first block reduce spatial dimension
        out = bottleneck(input, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=2, bias=bias, name='block0')

        for i in range(nBlocks-1):
            subname = 'block%d' % (i+1)
            out = bottleneck(out, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=1, bias=bias, name=subname)
    return out

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
        # n_feat_map * batch * d_filter

        if bias == True:
            b = tf.get_variable(
                "b",
                shape=width*height,
                initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)

    output_transpose = tf.transpose(output, [1,2,0])  # batch * d_filter * n_feat_map
    output_reshape = tf.reshape( output_transpose, [-1, width, height, n_feat_map] )
    return output_reshape


def bottleneck_half(input, channel_compress_ratio = 4, stride = 2, bias = True, name = None, reuse = None):
    """ block for residual network """
    """ function : reducing dimension """
    input_shape = input.get_shape().as_list()
    
    output_channel = input_shape[3] * 2
    
    bottleneck_channel = output_channel / channel_compress_ratio
    
    with tf.variable_scope(name):
    
        bn0 = tf.nn.elu(batchnorm(input, name = 'bn0'))
        
        shortcut = conv_layer_new(bn0, [1, 1, input_shape[3], output_channel], strides = [1, 2, 2, 1], name = 'shortcut_conv')
        
        ###### bottleneck
        conv1 = conv_layer_new(bn0, [1, 1, input_shape[3], bottleneck_channel], strides = [1, 2, 2, 1], name = 'conv1')
        
        bn1 = tf.nn.elu(batchnorm(conv1, name='bn1'))
        
        conv2 = conv_layer_new(bn1, [1, 1, bottleneck_channel, bottleneck_channel], strides = [1, 1, 1, 1], name = 'conv2')
        
        bn2 = tf.nn.elu(batchnorm(conv2, name='bn2'))
        
        conv3 = conv_layer_new(bn2, [1, 1, bottleneck_channel, output_channel], strides = [1, 1, 1, 1], name = 'conv3')
        
        out = shortcut + conv3
    
    return out
    
    
def bottleneck_quarter(input, output_channel, channel_compress_ratio = 4, stride = 2, bias = True, name = None, reuse = None):
    input_shape = input.get_shape().as_list()
    
    bottleneck_channel = output_channel / channel_compress_ratio
    
    with tf.variable_scope(name):
    
        bn0 = tf.nn.elu(batchnorm(input, name = 'bn0'))
        
        shortcut = conv_layer_new(bn0, [1, 1, input_shape[3], output_channel], strides = [1, 2, 2, 1], name = 'shortcut_conv')
        
        ###### bottleneck
        conv1 = conv_layer_new(bn0, [1, 1, input_shape[3], bottleneck_channel], strides = [1, 2, 2, 1], name = 'conv1')
        
        bn1 = tf.nn.elu(batchnorm(conv1, name='bn1'))
        
        conv2 = conv_layer_new(bn1, [1, 1, bottleneck_channel, bottleneck_channel], strides = [1, 1, 1, 1], name = 'conv2')
        
        bn2 = tf.nn.elu(batchnorm(conv2, name='bn2'))
        
        conv3 = conv_layer_new(bn2, [1, 1, bottleneck_channel, output_channel], strides = [1, 1, 1, 1], name = 'conv3')
    
    
    out = shortcut + conv3
    
    return out

    
def bottleneck_new(input, channel_compress_ratio = 4, stride = 1, bias = True, name = None, reuse = None):
    """ block for residual net
        function : not reducing the dimension"""
        
    input_shape = input.get_shape().as_list()
    
    output_channel = input_shape[3]
    
    bottleneck_channel = output_channel / channel_compress_ratio
    
    shortcut = input
    
    with tf.variable_scope(name):
        ###### bottleneck
        bn0 = tf.nn.elu(batchnorm(input, name = 'bn0'))
        
        conv1 = conv_layer_new(bn0, [1, 1, input_shape[3], bottleneck_channel], strides = [1, 1, 1, 1], name = 'conv1')
        
        bn1 = tf.nn.elu(batchnorm(conv1, name='bn1'))
        
        conv2 = conv_layer_new(bn1, [1, 1, bottleneck_channel, bottleneck_channel], strides = [1, 1, 1, 1], name = 'conv2')
        
        bn2 = tf.nn.elu(batchnorm(conv2, name='bn2'))
        
        conv3 = conv_layer_new(bn2, [1, 1, bottleneck_channel, output_channel], strides = [1, 1, 1, 1], name = 'conv3')
    
    out = shortcut + conv3
    
    return out 

def fc_layer_new(input, output_size, name = None, bias = True, reuse = None):
    """fully connected layer by yyh"""
    shape = input.get_shape().as_list()
    dim = np.prod( shape[1 :] ) #将除了第一维以外所有维度相乘
    x = tf.reshape(input, [-1, dim])#将输入重塑为一个矩阵，第一维不变
    input_size = dim
    
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable("w", shape = [input_size, output_size], initializer = tf.truncated_normal_initializer(0., 0.005))
    
        # b = tf.get_variable("b", shape = [output_size], initializer = tf.constant_initializer(0.))
        b = tf.get_variable("b", shape = [output_size], initializer = tf.truncated_normal_initializer(0.))
    
    fc_out = tf.nn.bias_add(tf.matmul(x, w), b)
    
    return fc_out[:, 0]
    
def conv_layer_new(input, filter_shape, strides, name = None):

    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_shape, initializer = tf. truncated_normal_initializer(0., 0.005))
        # b = tf.get_variable("b", filter_shape[-1], initializer = tf.constant_initializer(0.))
        b = tf.get_variable("b", filter_shape[-1], initializer = tf.truncated_normal_initializer(0.))
        
        conv = tf.nn.conv2d(input, W, strides, padding = 'SAME')
        
        output = tf.nn.bias_add(conv, b)
        
        output = tf.identity(output)
    
    return output

def deconv_layer_new(input, filter_shape, output_shape, strides, name = None):
    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_shape, initializer = tf. truncated_normal_initializer(0., 0.005))
        
        # b = tf.get_variable("b", filter_shape[-2], initializer = tf.constant_initializer(0.))
        b = tf.get_variable("b", filter_shape[-2], initializer = tf.truncated_normal_initializer(0.))
        
        conv = tf.nn.conv2d_transpose(input, W, output_shape, strides, padding = 'SAME')
        
        output = tf.nn.bias_add(conv, b)
        
        output = tf.identity(output)
    
    return output
    
    
    

# if autoencoder == True:

def projector_model_new( input ):

    # input = tf.Variable(tf.random_normal([64, 64, 64, 3]))
    with tf.variable_scope('PROJ', reuse = tf.AUTO_REUSE):
    
        with tf.variable_scope('ENCODE'):
            #conv0
            conv0 = conv_layer_new(input, [4, 4, 1, 64], strides = [1, 1, 1, 1], name = 'conv0')
            
            #bn0
            bn0 = tf.nn.elu(batchnorm(conv0, name='bn0'))
            
            #conv1
            conv1 = conv_layer_new(bn0, [4, 4, 64, 128], strides = [1, 1, 1, 1], name = 'conv1')

            #bn1
            bn1 = tf.nn.elu(batchnorm(conv1, name='bn1'))

            #conv2
            conv2 = conv_layer_new(bn1, [4, 4, 128, 256], strides = [1, 2, 2, 1], name = 'conv2')

            #bn2
            bn2 = tf.nn.elu(batchnorm(conv2, name='bn2'))
            
            #conv3
            
            conv3 = conv_layer_new(bn2, [4, 4, 256, 512], strides = [1, 2, 2, 1], name = 'conv3')

            #bn3
            bn3 = tf.nn.elu(batchnorm(conv3, name='bn3'))
            
            #conv4
            conv4 = conv_layer_new(bn3, [4, 4, 512, 1024], strides = [1, 2, 2, 1], name = 'conv4')

            #bn4
            bn4 = tf.nn.elu(batchnorm(conv4, name='bn4'))
            
            #fc
            fc = channel_wise_fc_layer(bn4, 'fc', bias = False, reuse = tf.AUTO_REUSE)


            #conv5 latent
            conv5 = conv_layer_new(fc, [1, 1, 1024, 1024], strides = [1, 1, 1, 1], name = 'conv5')
            
            #bn5
            latent = tf.nn.elu(batchnorm(conv5, name='latent'))

        #deconv4 append
        de_conv4 = deconv_layer_new(latent, [1, 1, 1024, 1024], conv4.get_shape(), strides = [1, 1, 1, 1], name = 'de_conv4')
        
        #debn4
        de_bn4 = tf.nn.elu(batchnorm(de_conv4, name = 'de_bn4'))

        #deconv3
        de_conv3 = deconv_layer_new(de_bn4, [4, 4, 512, 1024], conv3.get_shape(), strides = [1, 2, 2, 1], name = 'de_conv3')
        
        #debn3
        de_bn3 = tf.nn.elu(batchnorm(de_conv3, name = 'de_bn3'))

        #deconv2
        de_conv2 = deconv_layer_new(de_bn3, [4, 4, 256, 512], conv2.get_shape(), strides = [1, 2, 2, 1], name = 'de_conv2')
        
        #debn2
        de_bn2 = tf.nn.elu(batchnorm(de_conv2, name = 'de_bn2'))
        
        #deconv1
        de_conv1 = deconv_layer_new(de_bn2, [4, 4, 128, 256], conv1.get_shape(), strides = [1, 2, 2, 1], name = 'de_conv1')
        
        #debn1
        de_bn1 = tf.nn.elu(batchnorm(de_conv1, name = 'de_bn1'))

        #deconv0
        de_conv0 = deconv_layer_new(de_bn1, [4, 4, 64, 128], conv0.get_shape(), strides = [1, 1, 1, 1], name = 'de_conv0')

        #debn0
        de_bn0 = tf.nn.elu(batchnorm(de_conv0, name = 'de_bn0'))
        
        #deconv_ori
        de_conv_ori = deconv_layer_new(de_bn0, [4, 4, 1, 64], input.get_shape(), strides = [1, 1, 1, 1], name = 'de_conv_ori')
        
        proj = de_conv_ori
    
    return proj, latent

# if discriminator == True:

def discriminator_model_new( input ):
    with tf.variable_scope('DIS', reuse = tf.AUTO_REUSE):
    
        with tf.variable_scope('IMG'):
            # conv0_dis
            conv0 = conv_layer_new(input, [4, 4, 1, 64], strides = [1, 1, 1, 1], name = 'conv0')
            
            ################ module 1
            # block_half1
            # reduce the spatial dimension
            op_m1_h = bottleneck_half(conv0, channel_compress_ratio = 4, stride = 2, bias = True, name = 'op_m1_h', reuse = tf.AUTO_REUSE)
            
            # blocks_3
            # n_Block3 = 3
            n_Block2 = 2 # test
            
            temp_m1 = op_m1_h
            
            # for i in range(n_Block3 - 1): 
            for i in range(n_Block2 - 1): #test
                subname = 'module1_block%d' % (i + 1)
                temp_m1 = bottleneck_new(temp_m1, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_m1 = temp_m1
            ######################
            
            ###################### module 2
            # block_half2
            # reduce the spatial dimension
            op_m2_h = bottleneck_half(op_m1, channel_compress_ratio = 4, stride = 2, bias = True, name = 'op_m2_h', reuse = tf.AUTO_REUSE)
            
            # blocks_4
            n_Block4 = 4
            
            temp_m2 = op_m2_h
            
            for i in range(n_Block4 - 1):
                subname = 'module2_block%d' % (i + 1)
                temp_m2 = bottleneck_new(temp_m2, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_m2 = temp_m2
            ############################
            
            ########################### module 3
            # block_half3
            # reduce the spatial dimension
            op_m3_h = bottleneck_half(op_m2, channel_compress_ratio = 4, stride = 2, bias = True, name = 'op_m3_h', reuse = tf.AUTO_REUSE)
            
            # blocks_6
            n_Block6 = 6
            
            temp_m3 = op_m3_h
            
            for i in range(n_Block6 - 1):
                subname = 'module3_block%d' % (i + 1)
                temp_m3 = bottleneck_new(temp_m3, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_m3 = temp_m3
            ###########################
            
            ########################### module 4
            # block_half4
            # reduce the spatial dimension
            op_m4_h = bottleneck_half(op_m3, channel_compress_ratio = 4, stride = 2, bias = True, name = 'op_m4_h', reuse = tf.AUTO_REUSE)
            
            # blocks_3
            n_Block3 = 3
            
            temp_m4 = op_m4_h
            
            for i in range(n_Block3 - 1):
                subname = 'module4_block%d' % (i + 1)
                temp_m4 = bottleneck_new(temp_m4, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_m4 = temp_m4
            ##########################
            
            ######################## fully connected layer
            bn_fi = tf.nn.elu(batchnorm(op_m4, name = 'bn_fi'))
            
            dis = fc_layer_new(bn_fi, output_size = 1, name = 'fc_dis', bias = True, reuse = tf.AUTO_REUSE)
            #生成一列，所以是1
            ########################
            
            # dis1 = dis[:, 0]
    
    return dis
    

# if discriminator_latent == True:

def discriminator_latent_new( input ):
    with tf.variable_scope('DIS', reuse = tf.AUTO_REUSE):
        
        with tf.variable_scope('LATENT'):
            
            # blocks_3
            # n_Block3 = 3
            n_Block2 = 2 # test
            
            temp_m1 = input #op5
            
            # for i in range(n_Block3 - 1):
            for i in range(n_Block2 - 1): #test
                subname = 'latent_m1_block%d' % (i + 1)
                temp_m1 = bottleneck_new(temp_m1, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_la_m1 = temp_m1
            
            output_channel = op_la_m1.get_shape().as_list()[-1]
            
            # bottleneck quarter
            
            op_qt = bottleneck_quarter(op_la_m1, output_channel, channel_compress_ratio = 4, stride = 2, bias = True, name = "bottleneck_quarter", reuse = tf.AUTO_REUSE)
            
            #blocks_2
            n_Block2 = 2
            
            temp_m2 = op_qt
            
            for i in range(n_Block2 - 1):
                subname = 'latent_m2_block%d' % (i + 1)
                temp_m2 = bottleneck_new(temp_m2, channel_compress_ratio = 4, stride = 1, bias = True, name = subname, reuse = tf.AUTO_REUSE)
            op_la_m2 = temp_m2

            bn_la_fi = tf.nn.elu(batchnorm(op_la_m2, name = 'bn_la_fi'))
            
            # fully connected layer
            dis_la = fc_layer_new(bn_la_fi, output_size = 1, name = 'fc_dis_la', bias = True, reuse = tf.AUTO_REUSE)
    
    return dis_la
    
def create_train_procs(trainset, train_queue, img_size, n_thread, num_prepare, train_procs, std):
    """
    create threads to read the images from hard drive and perturb them
    """
    for n_read in range(n_thread):
        seed = np.random.randint(1e8)
        instance_size = batch_size# - n_reference
        if instance_size < 1:
            print('ERROR: batch_size < n_reference + 1')
        train_proc = Process(target=read_file_cpu, args=(trainset, train_queue, instance_size, img_size, std, num_prepare, seed,))
        train_proc.daemon = True
        train_proc.start()
        train_procs.append(train_proc)


def read_file_cpu(trainset, queue, batch_size, img_size, std, num_prepare, rseed=None):
        local_random = np.random.RandomState(rseed)

        n_train = len(trainset)
        trainset_index = local_random.permutation(n_train)
        idx = 0
        while True:
            # read in data if the queue is too short
            while queue.full() == False:
                batch = np.zeros([batch_size, img_size, img_size, 1])
                noisy_batch = np.zeros([batch_size, img_size, img_size, 1])
                for i in range(batch_size):
                    image_path = trainset[trainset_index[idx+i]]
                    img = sp.misc.imread(image_path)
                    # <Note> In our original code used to generate the results in the paper, we directly
                    # resize the image directly to the input dimension via (for both ms-celeb-1m and imagenet)
                    
                    
                    
                    img = sp.misc.imresize(img, [img_size, img_size]).astype(float) / 255.0 #弄到0到1的范围
                    
                    
                    
                    # print('img.shape : ', img.shape)
                    # The following code crops random-sized patches (may be useful for imagenet)
                    #img_shape = img.shape
                    #min_edge = min(img_shape[0], img_shape[1])
                    #min_resize_ratio = float(img_size) / float(min_edge)
                    #max_resize_ratio = min_resize_ratio * 2.0
                    #resize_ratio = local_random.rand() * (max_resize_ratio - min_resize_ratio) + min_resize_ratio

                    #img = sp.misc.imresize(img, resize_ratio).astype(float) / 255.0
                    #crop_loc_row = local_random.randint(img.shape[0]-img_size+1)
                    #crop_loc_col = local_random.randint(img.shape[1]-img_size+1)
                    #if len(img.shape) == 3:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size,:]
                    #else:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size]

                    # if np.prod(img.shape) == 0:
                        # img = np.zeros([img_size, img_size, 1])

                    # if len(img.shape) < 3:
                        # img = np.expand_dims(img, axis=2) #(28, 28)->(28, 28, 1)
                        # img = np.tile(img, [1,1,1])#(28, 28, 1)

                    ## random flip
                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[-1:None:-1,:,:]

                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[:,-1:None:-1,:]
                        
                    # 参数补充定义
                    uniform_noise_max = 3.464
                    min_spatially_continuous_noise_factor = 0.01
                    
                    max_spatially_continuous_noise_factor = 0.5
                    
                    continuous_noise = 1
                    
                    use_spatially_varying_uniform_on_top = 1
                    
                    clip_input = 0
                    
                    clip_input_bound = 2.0
                    
                    # add noise to img， noisy_img也在0到1的范围
                    noisy_img = add_noise(img, local_random,
                            std=std,
                            uniform_max=uniform_noise_max,
                            min_spatially_continuous_noise_factor=min_spatially_continuous_noise_factor,
                            max_spatially_continuous_noise_factor=max_spatially_continuous_noise_factor,
                            continuous_noise=continuous_noise,
                            use_spatially_varying_uniform_on_top=use_spatially_varying_uniform_on_top,
                            clip_input=clip_input, clip_input_bound=clip_input_bound
                            )

                    
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, axis=2) #(28, 28)->(28, 28, 1)
                    
                    # test
                    # noisy_img = img
                    
                    batch[i] = img
                    noisy_batch[i] = noisy_img

                batch *= 2.0 #为了弄到0到2的范围
                batch -= 1.0 #为了弄到-1到1的范围
                noisy_batch *= 2.0
                noisy_batch -= 1.0
                
                # 添加，保持在-1到1的范围
                batch = np.clip(batch, a_min = -1, a_max = 1)
                noisy_batch = np.clip(noisy_batch, a_min = -1, a_max = 1)
                
                #恢复到0-255
                batch = batch * 127.5 + 127.5
                noisy_batch = noisy_batch * 127.5 + 127.5
                
                
                if clip_input > 0:
                    batch = np.clip(batch, a_min=-clip_input_bound, a_max=clip_input_bound)
                    noisy_batch = np.clip(noisy_batch, a_min=-clip_input_bound, a_max=clip_input_bound)

                queue.put([batch, noisy_batch]) # block until free slot is available

                idx += batch_size
                # if idx > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                if idx > n_train - batch_size:
                    trainset_index = local_random.permutation(n_train)
                    idx = 0



                    
def terminate_train_procs(train_procs):
        """
        terminate the threads to force garbage collection and free memory
        """
        for procs in train_procs:
            procs.terminate()
            
            

if __name__ == '__main__':
    
    ############### arguements

    batch_size = 32
    img_size = 28
    one_sided_label_smooth = 0.85 # default
    
    lambda_ratio = 1e-2 # weight ratio of true to fake, lambda1 in paper
    lambda_de = 1.0 # denoising autoencoder, lambda2 in paper
    lambda_l2 = 5e-3 # l2 loss, lambda3 in paper
    lambda_la = 1e-4 # lambda of latent loss, lambda4 in paper
    lambda_im = 1e-3 # lambda of img loss, lambda5 in paper
    
    
    weight_decay_rate = 0.00001

    learning_rate_proj = 0.002

    adam_beta1_g = 0.9
    adam_beta2_g = 0.999
    adam_eps_g = 1e-5 #default = 1e-8

    learning_rate_dis = 0.0002
    adam_beta1_d = 0.9
    adam_beta2_d = 0.999
    adam_eps_d = 1e-8

    clamp_lower = -0.01 # for clipping var
    clamp_upper = 0.01
    clamp_weight = 1

    D_period = 1
    G_period = 1

    iters_num = 80000

    output_img = 100 # whether to output images, also act as the number of images to output
    output_img_period = 10 #迭代多少次输出一次图像

    num_epochs = 100 # 训练的批次数
    noise_std = 1.2 # noise std

    ################# load mnist data
    
    trainset = load_dataset.load_trainset_path_list()
    #trainset 是一个字典
    #通过函数、目录读取训练数据
    print('trainset : ', trainset)
    
    total_train = len(trainset)

    print('total_train = %d' % (total_train))

    iters_in_epoch = total_train // batch_size

    print('iters_in_epoch = %d' % (iters_in_epoch))

    ####### process

    print('loading_data...')

    n_thread = 8
    # n_thread = 16

    num_prepare = 10
    # num_prepare = 20

    train_queue = Queue(num_prepare + 1)

    train_procs = []

    create_train_procs(trainset, train_queue, img_size, n_thread, num_prepare, train_procs, std = noise_std)

    ####### building the graph #######

    # input = tf.Variable(tf.random_normal([batch_size, 28, 28, 1]))

    images_tf = tf.placeholder( tf.float32, [batch_size, img_size, img_size, 1], name="images_tf")

    # input_z = tf.Variable(tf.random_normal([batch_size, 28, 28, 1])) # add noisy

    noisy_images_tf = tf.placeholder( tf.float32, [batch_size, img_size, img_size, 1], name="noisy_images_tf")

    ## autoencoder ##
    #input 和 input_z分别要换为images_tf和noisy_images_tf
    proj_x, latent_x = projector_model_new(images_tf)
    proj_z, latent_z = projector_model_new(noisy_images_tf)

    ## discriminator ##
    # image

    dis_true = discriminator_model_new(images_tf)
    dis_proj_x = discriminator_model_new(proj_x)
    dis_proj_z = discriminator_model_new(proj_z) 

    # latent

    dis_la_x = discriminator_latent_new(latent_x)
    dis_la_z = discriminator_latent_new(latent_z)

    # tf.nn.sigmoid_cross_entropy_with_logits 
    # output : shape [batch_size, num_classes]




    ############### folder

    base_folder = 'J:/桌面资料/毕设资料/导师分享/Tensor_GAN/OneNet-master-2to3/projector/base_folder'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        
    model_path = '%s/model' % (base_folder)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    epoch_path = '%s/epoch' % (base_folder)
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)

    init_path = '%s/init' % (base_folder)
    if not os.path.exists(init_path):
        os.makedirs(init_path)

    img_path = '%s/image' % (base_folder)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        
    tf.set_random_seed(0)

    ################# loss for discriminator 

    pos_labels = tf.ones(batch_size, 1) #这里没想通，x针对Pos

    soft_pos_labels = pos_labels * one_sided_label_smooth

    neg_labels = tf.zeros(batch_size, 1)#z针对neg

    # if we are using virtual batch normalization, we do not need to calculate the population mean and variance

    updates = tf.zeros([1])

    ##########latent space


    dis_la_pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = soft_pos_labels, logits = dis_la_x))  #size = batch_size, loss的平均值
    dis_la_neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = neg_labels, logits = dis_la_z)) #和第四部分有相似之处，labels不一样

    dis_la_loss = lambda_ratio * dis_la_pos_loss + (1 - lambda_ratio) * dis_la_neg_loss #和第四部分有相似之处，labels不一样

    ### est_la
    est_labels_la1 = tf.greater_equal(dis_la_x, 0.0)
    # dis_la_x大于0的越多会造成影响，使accracy越高

    est_labels_la2 = tf.less_equal(dis_la_z, 0.0)
    # dis_la_z小于0的越多会造成影响，使accracy越高

    est_labels_la = tf.concat([tf.to_float(est_labels_la1), tf.to_float(est_labels_la2)], 0)

    accuracy_la = tf.reduce_mean(est_labels_la)

    ##########image space

    dis_im_pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = soft_pos_labels, logits = dis_true))

    dis_im_neg_proj_x_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = neg_labels, logits = dis_proj_x))

    dis_im_neg_proj_z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = neg_labels, logits = dis_proj_z))

    dis_im_loss = (dis_im_pos_loss + lambda_ratio * dis_im_neg_proj_x_loss + (1 - lambda_ratio) * dis_im_neg_proj_z_loss) * 0.5 #weight mean 

    ### est_im
    est_labels_im1 = tf.greater_equal(dis_true, 0.0)

    est_labels_im2 = tf.less_equal(dis_proj_x, 0.0)

    est_labels_im = tf.concat([tf.to_float(est_labels_im1), tf.to_float(est_labels_im2)], 0)

    accuracy_im = tf.reduce_mean(est_labels_im)

    ################### loss for autoencoder #################

    recon_loss = tf.reduce_mean(tf.square(proj_x - images_tf)) #objective function第一部分

    recon_z_loss = tf.reduce_mean(tf.square(proj_z - images_tf)) #objective function第二部分

    proj_loss = -tf.reduce_mean(tf.square(proj_z - noisy_images_tf)) #objective function第三部分,论文这个地方应该有负号才符合逻辑
    
    ######### G???????????
    ##latent
    labels_G = pos_labels

    G_la_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_G, logits = dis_la_z)) # objective function里面的第四部分

    ##imagespace
    G_proj_x_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_G, logits = dis_proj_x)) 

    G_proj_z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_G, logits = dis_proj_z)) # objective function第五部分
    
    G_im_loss = lambda_ratio * G_proj_x_loss + (1 - lambda_ratio) * G_proj_z_loss # 由于0.01，所以相当于就是第五部分

    ################## loss of G and D ##############

    # G_loss = lambda_la * G_la_loss + lambda_im * G_im_loss # 四部分加五部分，为什么没有负号
    G_loss = -(lambda_la * G_la_loss + lambda_im * G_im_loss) #改过后
    

    D_loss = lambda_la * dis_la_loss + lambda_im * dis_im_loss #有点像四+五，只不过labels都变成了neg_labels,dis_im_loss多了一个dis_true   反正这东西论文里没提

    # real_G_loss = G_loss + lambda_l2 * (lambda_ratio * recon_loss + (1 - lambda_ratio) * proj_loss) #G_loss应该有负号；括号内相当于是第三部分；整体为三+四+五
    real_G_loss = G_loss + lambda_l2 * proj_loss 

    # train with a denosing autoencoder weight

    real_G_loss = real_G_loss + lambda_de * recon_z_loss #二+三+四+五
    
    # 发现少了第一部分，加上
    real_G_loss = real_G_loss + lambda_ratio * recon_loss
    
    #test
    # real_G_loss = recon_z_loss
    
    var_D = [x for x in tf.trainable_variables() if x.name.startswith('DIS')]

    W_D = [x for x in var_D if x.name.endswith('W:0')]

    W_D_clip = [tf.assign(w_d, tf.clip_by_value(w_d, clamp_lower, clamp_upper)) for w_d in W_D] #tf.assign 将原始值更新为新值

    var_G = [x for x in tf.trainable_variables() if x.name.startswith('PROJ')]

    W_G = [x for x in var_G if x.name.endswith('W:0')]

    W_G_clip = [tf.assign(w_g, tf.clip_by_value(w_g, clamp_lower, clamp_upper)) for w_g in W_G]

    var_E = [x for x in tf.trainable_variables() if 'ENCODE' in x.name]

    W_E = [x for x in var_E if x.name.endswith('W:0')]

    if weight_decay_rate > 0:
        real_G_loss = real_G_loss + weight_decay_rate * tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in W_G]))
        #pack 类似于 concat, pack在新版本报错找不到，改为stack
        D_loss = D_loss + weight_decay_rate * tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in W_D]))
        
        
    ################## optimizer

    ########## optimizer_G

    optimizer_G = tf.train.AdamOptimizer(learning_rate = learning_rate_proj, beta1 = adam_beta1_g, beta2 = adam_beta2_g, epsilon = adam_eps_g)

    # compute_gradients函数返回一个梯度，变量各自相对应的列表,minimize函数的第一个部分

    grads_vars_list_G = optimizer_G.compute_gradients(real_G_loss, var_list = var_G)

    # 这里的clip把梯度的值限制在-10到10的范围，gv[0]代表梯度，gv[1]代表变量

    grads_clipped_vars_list_G = [[tf.clip_by_value(gv_group[0], -10., 10.), gv_group[1]] for gv_group in grads_vars_list_G]

    # train G: apply_gradients把梯度应用到变量上，按照梯度下降的方式加到上面去，这是minimize的第二个步骤。返回一个应用的操作？

    train_G_op = optimizer_G.apply_gradients(grads_clipped_vars_list_G)
    
    # train_G_op = optimizer_G.apply_gradients(grads_vars_list_G) #试一试不clip

    ########### optimizer_D

    optimizer_D = tf.train.AdamOptimizer(learning_rate = learning_rate_dis, beta1 = adam_beta1_d, beta2 = adam_beta2_d, epsilon = adam_eps_g)
        
    #compute gradients for variables

    grads_vars_list_D = optimizer_D.compute_gradients(D_loss, var_list = var_D)

    #clip

    grads_clipped_vars_list_D = [[tf.clip_by_value(gv_group[0], -10., 10.), gv_group[1]] for gv_group in grads_vars_list_D]

    D_var_clip_ops = [tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)) for v in W_D]

    E_var_clip_ops = [tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)) for v in W_E]

    # train D

    train_D_op = optimizer_D.apply_gradients(grads_clipped_vars_list_D)
    # train_D_op = optimizer_D.apply_gradients(grads_vars_list_D) #test not clip

    ########## setup the saver,  max_to_keep是最多保存多少checkpoint文件，多出的话就覆盖掉，默认是5

    saver = tf.train.Saver(max_to_keep = 20) # iters的ckpt最多存20个

    saver_epoch = tf.train.Saver(max_to_keep = 50) # epoch的ckpt最多存50个

    #### setup the image saver

    if output_img > 0:
        num_output_img = min(5, batch_size)
        
        # output_ori_imgs_op = (images_tf[0 : num_output_img] * 127.5) + 127.5 # 前面除了255
        output_ori_imgs_op = images_tf[0 : num_output_img]
        
        # output_noisy_imgs_op = (noisy_images_tf[0 : num_output_img] * 127.5) + 127.5 # 原始图像加噪声图像
        output_noisy_imgs_op = noisy_images_tf[0 : num_output_img]
        
        # tf.clip_by_value(proj_z, clip_value_min = -1, clip_value_max = 1)
        # output_proj_imgs_op = (proj_z[0 : num_output_img] * 127.5) + 127.5 # 加噪声图像进入projection
        output_proj_imgs_op = proj_z[0 : num_output_img]
        
        # tf.clip_by_value(proj_x, clip_value_min = -1, clip_value_max = 1)
        # output_recon_imgs_op = (proj_x[0 : num_output_img] * 127.5) + 127.5
        # 原始图像进入projection
        output_recon_imgs_op = proj_x[0 : num_output_img]
        
    ######### initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ######### start training

    print('start training')

    start_time = timeit.default_timer()

    epoch = 0 # initialize epoch

    print("epoch = %d" % (epoch)) #不能打逗号



    #?
    loss_dis_avg = SmoothStream(window_size=100)

    acc_latent_avg = SmoothStream(window_size=100)

    acc_img_avg = SmoothStream(window_size=100)

    loss_recon_avg = SmoothStream(window_size=100)

    loss_recon_z_avg = SmoothStream(window_size=100)

    update_D_left = D_period
    update_G_left = G_period

    loss_G_val = 0
    loss_proj_val = 0
    loss_recon_val = 0
    loss_recon_z_val = 0
    loss_adv_G_val = 0
    loss_D_val = 0
    acc_latent_val = 0
    acc_img_val = 0
    iters = 0 #initialize


    #####alternative training starts

    print('alternative training starts...')

    while True:
        batch, noisy_batch = train_queue.get()
        
        #update G
        if update_G_left > 0:
            

            _, loss_G_val, loss_proj_val, loss_recon_val, loss_recon_z_val, loss_adv_G_val, _ = sess.run([train_G_op, real_G_loss, proj_loss, recon_loss, recon_z_loss, G_loss, updates],
            feed_dict = {images_tf : batch,
            noisy_images_tf : noisy_batch})
            
            update_G_left = update_G_left - 1
            
            loss_recon_avg.insert(loss_recon_val)
            loss_recon_z_avg.insert(loss_recon_z_val)

        #update D
        if update_G_left <= 0 and update_D_left > 0:
            
            _, loss_D_val, acc_latent_val, acc_img_val, _ = sess.run([train_D_op, D_loss, accuracy_la, accuracy_im, updates], feed_dict = {images_tf : batch, noisy_images_tf : noisy_batch})
            
            loss_dis_avg.insert(loss_D_val)

            acc_latent_avg.insert(acc_latent_val)

            acc_img_avg.insert(acc_img_val)
            
            
            if clamp_weight > 0:
                
                _, _ = sess.run([D_var_clip_ops, E_var_clip_ops])
            
            update_D_left = update_D_left - 1

        #reset if zeros
        if update_G_left <= 0 and update_D_left <= 0:
            update_G_left = G_period
            update_D_left = D_period
        
        #save model once every 100 times
        if iters % 1000 == 0:
            saver.save(sess, model_path + '/model_iter', global_step = iters)
            # filename : 'model_path + /model_iter-iters'
            print('model saved once...')
        if iters % 100 == 0:
            print("Iter %d (%.2f minutes): real_G_loss=%.3e  proj_loss=%.3e recon_loss=%.3e (%.3e) recon_z_loss=%.3e (%.3e) loss_adv_gen=%.3e loss_dis=%.3e (%.3e) accuracy_img=%.3e (%.3e) accuracy_latent=%.3e (%.3e) learning_rate_proj=%.3e learning_rate_dis=%.3e qsize=%d" % (
                    iters, (timeit.default_timer()-start_time)/60., loss_G_val, loss_proj_val, loss_recon_val,
                    loss_recon_avg.get_moving_avg(), loss_recon_z_val, loss_recon_z_avg.get_moving_avg(), loss_adv_G_val, loss_D_val, loss_dis_avg.get_moving_avg(),
                    acc_img_val, acc_img_avg.get_moving_avg(),
                    acc_latent_val, acc_latent_avg.get_moving_avg(),
                    learning_rate_proj, learning_rate_dis, train_queue.qsize()))
        
        #save image every 10 iters
        if output_img > 0 and iters % output_img_period == 0:
        # if output_img > 0:
            output_ori_imgs_val, output_noisy_imgs_val, output_proj_imgs_val, output_recon_imgs_val = sess.run([output_ori_imgs_op, output_noisy_imgs_op, output_proj_imgs_op, output_recon_imgs_op], 
            feed_dict = {images_tf : batch,
            noisy_images_tf : noisy_batch})#feed_dict
            
            output_folder = '%s/iter_%d' % (img_path, iters)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for i in range(num_output_img):
                #output_ori_imgs_val 有五副
                filename = '%s/ori_%d.jpg' % (output_folder, i)
                
                output_ori_imgs_val_resp = output_ori_imgs_val[i].reshape(img_size, img_size) # 针对mnist
                    
                
                sp.misc.imsave(filename, output_ori_imgs_val_resp.astype('uint8'))
                
                filename = '%s/noisy_%d.jpg' % (output_folder, i)
                
                output_noisy_imgs_val_resp = output_noisy_imgs_val[i].reshape(img_size, img_size) 
                
                
                sp.misc.imsave(filename, output_noisy_imgs_val_resp.astype('uint8'))
                
                filename = '%s/(proj_z)proj_%d.jpg' % (output_folder, i)
                
                output_proj_imgs_val_resp = output_proj_imgs_val[i].reshape(img_size, img_size)
                
                
                sp.misc.imsave(filename, output_proj_imgs_val_resp.astype('uint8'))
                
                filename = '%s/(proj_x)recon_%d.jpg' % (output_folder, i)
                
                output_recon_imgs_val_resp = output_recon_imgs_val[i].reshape(img_size, img_size)
                
                
                sp.misc.imsave(filename, output_recon_imgs_val_resp.astype('uint8'))
        
        iters = iters + 1
        
        if iters % 10 == 0:
            print('the time of iteration is %d %.2f minutes' % (iters, (timeit.default_timer()-start_time)/60.))
        
        if iters % iters_in_epoch == 0:
        #一个60000已经跑过一次，存一次模型
            epoch = epoch + 1
            
            saver_epoch.save(sess, epoch_path + '/model_epoch', global_step = epoch)
            
            learning_rate_dis = learning_rate_dis * 0.95
            
            learning_rate_proj = learning_rate_proj * 0.95
            
            if epoch > num_epochs:
                break
            
        if iters % 20000 == 0:
            terminate_train_procs(train_procs)
            
            del train_procs
            
            del train_queue
            
            train_queue = Queue(num_prepare + 1)
            
            train_procs = []
            
    sess.close()

#=======================================

'''
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

if use_pretrain == True:
    print('')

op0, op1, op2, op3, op4, fc, op5, op_de4, op_de3, op_de2, op_de1, op_de0, output, = sess.run([op0, op1, op2, op3, op4, fc, op5, op_de4, op_de3, op_de2, op_de1, op_de0, output, ])
input, input_z, proj_x, proj_z, latent_x, latent_z, dis_true, dis_proj_x, dis_proj_z, dis_la_x, dis_la_z, dis_la_pos_loss, dis_la_neg_loss, dis_la_loss, est_labels_la, accracy_la, D_loss, G_loss, real_G_loss, proj_loss, proj_z, input_z = sess.run([input, input_z, proj_x, proj_z, latent_x, latent_z, dis_true, dis_proj_x, dis_proj_z, dis_la_x, dis_la_z, dis_la_pos_loss, dis_la_neg_loss, dis_la_loss, est_labels_la, accracy_la, D_loss, G_loss, real_G_loss, proj_loss, proj_z, input_z])

print('grad_vars_G_clipped', grad_vars_G_clipped)
# print('grads_vars_list_G : ', grads_vars_list_G)
# print("var_G : ", var_G);
# print("W_G : ", W_G);
# print("input : ", input.shape)
# print("input_z : ", input_z.shape)
# print("proj_x : ", proj_x.shape)
# print("proj_z : ", proj_z.shape)
# print("latent_x : ", latent_x.shape)
# print("latent_z : ", latent_z.shape)

# print("dis_true : ", dis_true.shape)
# print("dis_proj_x : ", dis_proj_x.shape)
# print("dis_proj_z : ", dis_proj_z.shape)

print("dis_la_x : ", sess.run(dis_la_x))
print("dis_la_z : ", dis_la_z.shape)
print("batch_loss_x_la : ", batch_loss_x_la.shape)
print("batch_loss_x_la : ", sess.run(batch_loss_x_la))
print('dis_la_pos_loss : ', sess.run(dis_la_pos_loss))
print("batch_loss_z_la : ", batch_loss_z_la.shape)
print("batch_loss_z_la : ", sess.run(batch_loss_z_la))

print('dis_la_neg_loss : ', sess.run(dis_la_neg_loss))
print('dis_la_loss : ', sess.run(dis_la_loss))

# print('est_labels_la : ', est_labels_la)
# print('accracy_la : ', accracy_la)
# print('D_loss : ', D_loss)
# print('G_loss : ', G_loss)
# print('real_G_loss : ', real_G_loss)
# print('proj_loss : ', proj_loss)
# print('mean of proj_z : ', sess.run(tf.reduce_mean(proj_z)))
# print('mean of input_z : ', sess.run(tf.reduce_mean(input_z)))

print("=============projector============")
print('input : ', input.shape)
print('op0 : ', op0.shape)
print('op1 : ', op1.shape)
print('op2 : ', op2.shape)
print('op3 : ', op3.shape)
print('op4 : ', op4.shape)
print('fc : ', fc.shape)
print('op5 : ', op5.shape)

print('op_de4 : ', op_de4.shape)
print('op_de3 : ', op_de3.shape)
print('op_de2 : ', op_de2.shape)
print('op_de1 : ', op_de1.shape)
print('op_de0 : ', op_de0.shape)
print('output : ', output.shape)

print("============discriminator===========")
print('op_d_c : ', op_d_c.shape)
print('op_m1_h : ', op_m1_h.shape)
print('op_m1 : ', op_m1.shape)
print('op_m2_h : ', op_m2_h.shape)
print('op_m2 : ', op_m2.shape)
print('op_m3_h : ', op_m3_h.shape)
print('op_m3 : ', op_m3.shape)
print('op_m4_h : ', op_m4_h.shape)
print('op_m4 : ', op_m4.shape)
print('dis : ', dis.shape)

print("============discriminator_latent=========")
print("op_la_m1 : ", op_la_m1.shape)
print("op_qt : ", op_qt.shape)
print("op_la_m2 : ", op_la_m2.shape)
print("dis_la : ", dis_la.shape)
'''