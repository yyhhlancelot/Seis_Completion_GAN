import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy as sp
import scipy.io as sio
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import timeit
from smooth_stream import SmoothStream
from multiprocessing import Process, Queue
import load_seis_1channel as load_dataset
from noise import add_noise
import argparse
import random
from layers import *

def completion_model_new( input ):
    with tf.variable_scope('CON', reuse = tf.AUTO_REUSE):
        #conv0
        # print('input', input.shape)
        
        conv0 = conv_layer_new(input, [5, 5, 1, 64], strides = [1, 1, 1, 1], name = 'conv0')
        
        # print('conv0', conv0.shape)
        
        #relu0 
        relu0 = tf.nn.relu(batchnorm(conv0, name = 'bn0'), name = 'relu0')
        
        # print('relu0', relu0.shape)
        
        #conv1
        conv1 = conv_layer_new(relu0, [3, 3, 64, 128], strides = [1, 2, 2, 1], name = 'conv1')
        
        # print('conv1', conv1.shape)
        
        #relu1
        relu1 = tf.nn.relu(batchnorm(conv1, name = 'bn1'), name = 'relu1')
        
        # print('relu1', relu1.shape)
        
        #conv2
        conv2 = conv_layer_new(relu1, [3, 3, 128, 128], strides = [1, 1, 1, 1], name = 'conv2')
        
        # print('conv2', conv2.shape)
        
        #relu2 
        relu2 = tf.nn.relu(batchnorm(conv2, name = 'bn2'), name = 'relu2')
        
        # print('relu2', relu2.shape) 
        
        #conv3
        conv3 = conv_layer_new(relu2, [3, 3, 128, 256], strides = [1, 2, 2, 1], name = 'conv3') ###
        
        # print('conv3', conv3.shape)
        
        #relu3 
        relu3 = tf.nn.relu(batchnorm(conv3, name = 'bn3'), name = 'relu3')
        
        # print('relu3', relu3.shape)
        
        #conv4
        conv4 = conv_layer_new(relu3, [3, 3, 256, 256], strides = [1, 1, 1, 1], name = 'conv4')
        
        # print('conv4', conv4.shape)
        
        #relu4 
        relu4 = tf.nn.relu(batchnorm(conv4, name = 'bn4'), name = 'relu4')
        
        # print('relu4', relu4.shape)
        
        #conv5
        conv5 = conv_layer_new(relu4, [3, 3, 256, 256], strides = [1, 1, 1, 1], name = 'conv5')
        
        # print('conv5', conv5.shape)
        
        #relu5 
        relu5 = tf.nn.relu(batchnorm(conv5, name = 'bn5'), name = 'relu5')
        
        # print('relu5', relu5.shape)
        
        #dilated conv6
        di_conv6 = dilated_conv_layer_new(relu5, [3, 3, 256, 256], rate = 2, name = 'diconv6')
        
        # print('di_conv6', di_conv6.shape)
        
        #relu6
        relu6 = tf.nn.relu(batchnorm(di_conv6, name = 'bn6'), name = 'relu6')
        
        # print('relu6', relu6.shape)
        
        #dilated conv7
        di_conv7 = dilated_conv_layer_new(relu6, [3, 3, 256, 256], rate = 4, name = 'diconv7')
        
        # print('di_conv7', di_conv7.shape)
        
        #relu7
        relu7 = tf.nn.relu(batchnorm(di_conv7, name = 'bn7'), name = 'relu7')
        
        # print('relu7', relu7.shape)
        
        #deconv8
        de_conv8 = deconv_layer_new(relu7, [4, 4, 128, 256], conv2.get_shape(), strides = [1, 2, 2, 1], name = 'de_conv8')
        
        # print('de_conv8', de_conv8.shape)
        
        #relu8
        relu8 = tf.nn.relu(batchnorm(de_conv8, name = 'bn8'), name = 'relu8')
        
        # print('relu8', relu8.shape)
        
        #conv9
        conv9 = conv_layer_new(relu8, [3, 3, 128, 256],  strides = [1, 1, 1, 1], name = 'conv9')
        
        # print('conv9', conv9.shape)
        
        relu9 = tf.nn.relu(batchnorm(conv9, name = 'bn9'), name = 'relu9')
        
        # print('relu9', relu9.shape)
        
        #deconv10
        # de_conv10 = deconv_layer_new(relu9, [4, 4, 64, 256], conv0.get_shape(), strides = [1, 1/2, 1/2, 1], name = 'de_conv10')
        de_conv10 = deconv_layer_new(relu9, [4, 4, 64, 256], conv0.get_shape(), strides = [1, 2, 2, 1], name = 'de_conv10')
        
        # print('deconv10', de_conv10.shape)
        
        relu10 = tf.nn.relu(batchnorm(de_conv10, name = 'bn10'), name = 'relu10')
        
        # print('relu10', relu10.shape)
        
        #conv11
        conv11 = conv_layer_new(relu10, [3, 3, 64, 32], strides = [1, 1, 1, 1], name = 'conv11')
        
        # print('conv11', conv11.shape)
        
        relu11 = tf.nn.relu(batchnorm(conv11, name = 'bn11'), name = 'relu11')
        
        # print('relu11', relu11.shape)
        
        #conv12
        conv12 = conv_layer_new(relu11, [3, 3, 32, 1], strides = [1, 1, 1, 1], name = 'conv12')
        
        # print('conv12', conv12.shape)
        
        G_logit = conv12
        
        G_prob = tf.nn.sigmoid(conv12)
        
        # print('G_prob', G_prob.shape)
        
        return G_prob, G_logit

def local_discriminator_new( input ):
    with tf.variable_scope('L_DIS', reuse = tf.AUTO_REUSE):
    
        with tf.variable_scope('IMG'):
            #conv0_dis
            conv0 = conv_layer_new(input, [5, 5, 1, 64], strides = [1, 2, 2, 1], name = 'conv0')
            
            #relu0
            relu0 = tf.nn.relu(batchnorm(conv0, name = 'bn0'), name = 'relu0')
            
            #conv1_dis
            conv1 = conv_layer_new(relu0, [5, 5, 64, 128], strides = [1, 2, 2, 1], name = 'conv1')
            
            #relu1
            relu1 = tf.nn.relu(batchnorm(conv1, name = 'bn1'), name = 'relu1')
            
            #conv2_dis
            conv2 = conv_layer_new(relu1, [5, 5, 128, 256], strides = [1, 2, 2, 1], name = 'conv2')
            
            #relu2
            relu2 = tf.nn.relu(batchnorm(conv2, name = 'bn2'), name = 'relu2')
            
            #conv3_dis
            conv3 = conv_layer_new(relu2, [5, 5, 256, 512], strides = [1, 2, 2, 1], name = 'conv3')
            
            #relu3
            relu3 = tf.nn.relu(batchnorm(conv3, name = 'bn3'), name = 'relu3')
            
            #conv4_dis
            conv4 = conv_layer_new(relu3, [5, 5, 512, 1024], strides = [1, 2, 2, 1], name = 'conv4')
            
            #relu4
            relu4 = tf.nn.relu(batchnorm(conv4, name = 'bn4'), name = 'relu4')
            
            #FC
            fc = new_fc_layer(relu4, output_size = 1, 'fc', bias = False, reuse = tf.AUTO_REUSE)
            
            D_logit = fc
            
            D_prob = tf.nn.sigmoid(D_logit)
            
            return D_prob, D_logit
            
    
def create_train_procs(trainset, num_reference, train_queue, img_size, n_thread, num_prepare, sampling_rate, train_procs, std):
    """
    create threads to read the images from hard drive and perturb them
    """
    for n_read in range(n_thread):
        seed = np.random.randint(1e8)
        instance_size = batch_size - num_reference
        if instance_size < 1:
            print('ERROR: batch_size < n_reference + 1')
        train_proc = Process(target=read_file_cpu, args=(trainset, train_queue, batch_size, num_reference, img_size, num_prepare, sampling_rate, seed))
        train_proc.daemon = True
        train_proc.start()
        train_procs.append(train_proc)


def read_file_cpu(trainset, queue, batch_size, num_reference, img_size, num_prepare, sampling_rate, rseed=None):
        
        local_random = np.random.RandomState(rseed)
        
        n_train = len(trainset)
        trainset_index = local_random.permutation(n_train)
        idx = 0
        while True:
            # read in data if the queue is too short
            
            # 参数补充定义
            std = 1.2 # noise_std
            
            uniform_noise_max = 3.464
            
            min_spatially_continuous_noise_factor = 0.01
            
            max_spatially_continuous_noise_factor = 0.5
            
            continuous_noise = 1
            
            use_spatially_varying_uniform_on_top = 1
            
            clip_input = 0
            
            clip_input_bound = 2.0
            
            while queue.full() == False:
                
                instance_size = batch_size - num_reference
                
                inst_batch = np.zeros([instance_size, img_size, img_size, 1])
                
                inst_incomplete_batch = np.zeros([instance_size, img_size, img_size, 1])
                
                
                
                for i in range(instance_size):
                    image_path = trainset[trainset_index[idx+i]]
                    
                    img_dict = sio.loadmat(image_path)
                    
                    # 归一化 [-1, 1]
                    img = img_dict['b'] / np.max(np.abs(img_dict['b']))
                    
                    # print("img' : ", img)
                    # os.system("pause")
                    
                    # incomplete img
                    # incomplete_img = batch_multiply(batch, mask_c) 
                    # mask_c似乎还没有定义
                    
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
                    
                    inst_batch[i] = img
                    # noisy_batch[i] = noisy_img


                inst_mask_c = generate_mask(instance_size, img_size, sampling_rate)
                
                inst_incomplete_batch = np.multiply(inst_batch, inst_mask_c)
                
                # batch *= 2.0 #为了弄到0到2的范围
                # batch -= 1.0 #为了弄到-1到1的范围
                # noisy_batch *= 2.0
                # noisy_batch -= 1.0
                
                # 添加，保持在-1到1的范围
                inst_batch = np.clip(inst_batch, a_min = -1, a_max = 1)
                # noisy_batch = np.clip(noisy_batch, a_min = -1, a_max = 1)
                inst_incomplete_batch = np.clip(inst_incomplete_batch, a_min = -1, a_max = 1)
                
                #恢复到0-255, 这里还是不恢复到0-255,先归一化处理
                # batch = batch * 127.5 + 127.5
                # noisy_batch = noisy_batch * 127.5 + 127.5
                
                
                
                if clip_input > 0:
                    inst_batch = np.clip(inst_batch, a_min=-clip_input_bound, a_max=clip_input_bound)
                    inst_noisy_batch = np.clip(inst_noisy_batch, a_min=-clip_input_bound, a_max=clip_input_bound)
                    

                # queue.put([batch, noisy_batch]) # block until free slot is available
                queue.put([inst_batch, inst_incomplete_batch, inst_mask_c])
                
                
                idx += batch_size
                # if idx > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                if idx > n_train - batch_size:
                    trainset_index = local_random.permutation(n_train)
                    idx = 0


def get_inst(batch, inst_size):
    return batch[0 : inst_size]

def terminate_train_procs(train_procs):
        """
        terminate the threads to force garbage collection and free memory
        """
        for procs in train_procs:
            procs.terminate()
            
            

if __name__ == '__main__':
    
    ############### arguements
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default = 96, help = 'input batch size')#32
    # batch_size = 96 #32
    parser.add_argument('--num_reference', default = 48, help = 'size of reference size')
    
    
    
    # batch_size = 1
    parser.add_argument('--img_size', default = 32, help = 'input image size')
    
    one_sided_label_smooth = 0.85 # default
    
    parser.add_argument('--lambda_ratio', default = 1e-2, help = 'weight ratio of true to fake, lambda1 in paper')
    
    parser.add_argument('--lambda_de', default = 1.0, help = 'denoising autoencoder, lambda2 in paper')
    
    parser.add_argument('--lambda_l2', default = 5e-3, help = 'l2 loss, lambda3 in paper') 
    
    parser.add_argument('--lambda_la', default = 1e-4, help = 'lambda of latent loss, lambda4 in paper')
    
    parser.add_argument('--lambda_im', default = 1e-3, help = 'lambda of img loss, lambda5 in paper')
    
    parser.add_argument('--sampling_rate', default = 4e-1, help = 'missing rate of the seis data')
    
    parser.add_argument('--alpha', default = 4e-4, help = 'weighint hyper parameter')
    
    args = parser.parse_args()
    
    batch_size = int(args.batch_size)
    
    num_reference = int(args.num_reference)
    
    instance_size = batch_size - num_reference
    
    img_size = int(args.img_size)
    
    sampling_rate = float(args.sampling_rate)
    
    alpha = float(args.alpha)
    
    weight_decay_rate = 0.00001

    learning_rate_proj = 2e-3

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
    
    ############### folder

    base_folder = 'J:/桌面资料/毕设资料/导师分享/Tensor_GAN/GLCGAN-master/projector_seis/base_folder'
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
        
    ################# load seis data
    
    trainset = load_dataset.load_trainset_path_list()
    #trainset 是一个字典
    #通过函数、目录读取训练数据
    print('trainset : ', trainset)
    
    total_train = len(trainset)

    print('total_train = %d' % (total_train))

    iters_in_epoch = total_train // batch_size

    print('iters_in_epoch = %d' % (iters_in_epoch))

    print('create reference batch...')
    
    n_thread = 1
    
    num_prepare = 1
    
    reference_queue = Queue(num_prepare)
    
    ref_seed = 1085
    
    instance_size = batch_size - num_reference
    
    ref_proc = Process(target = read_file_cpu, args = (trainset, reference_queue, batch_size, num_reference, img_size, num_prepare, sampling_rate, ref_seed))
    
    ref_proc.daemon = True
    
    ref_proc.start()
    
    ref_batch, ref_incomplete_batch, ref_mask_c = reference_queue.get() # 这里留到后面与batch和noisy_batch合并
    
    ref_proc.terminate()
    del ref_proc
    del reference_queue
    
    ####### process

    print('loading_data...')

    n_thread = 8

    num_prepare = 10

    train_queue = Queue(num_prepare + 1)

    train_procs = []

    create_train_procs(trainset, num_reference, train_queue, img_size, n_thread, num_prepare, sampling_rate, train_procs, std = noise_std)

    ####### building the graph #######

    images_tf = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 1], name = "images_tf")
    
    mask_c = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 1], name = 'mask')
    
    incomplete_images_tf = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 1], name = "incomplete_images_tf")
    
    ## autoencoder ##

    G_prob, images_recon = completion_model_new(incomplete_images_tf)
    
    D_real, D_logit_real = local_discriminator_new(images_tf)
    
    D_fake, D_logit_fake = local_discriminator_new(images_recon)
    

    ################# loss for discriminator 
    seis_recon_loss = tf.reduce_sum(tf.square(tf.multiply(mask_c, tf.abs(images_recon - images_tf)))) # 2 因为只需要计算不一样的那部分的损失，这个地方好像有问题
    print('seis_recon_loss_shape:', seis_recon_loss.shape)
    
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
    print('D_loss_real_shape:', D_loss_real.shape)
    
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))
    print('D_loss_fake_shape:', D_loss_fake.shape)
    # os.system("pause")
    
    D_loss = D_loss_real + D_loss_fake # 3
    
    # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discrimination_F, labels = tf.ones_like(discrimination_F)))
    
    # g_loss = tf.reduce_sum(tf.abs(images_tf - images_com))
    

    D_loss_joint = tf.reduce_mean(seis_recon_loss + alpha * D_loss_fake) # 5
    ################## vars of G and D ##############
    
    var_D = [x for x in tf.trainable_variables() if x.name.startswith('L_DIS')]

    W_D = [x for x in var_D if x.name.endswith('W:0')]

    W_D_clip = [tf.assign(w_d, tf.clip_by_value(w_d, clamp_lower, clamp_upper)) for w_d in W_D] #tf.assign 将原始值更新为新值

    var_G = [x for x in tf.trainable_variables() if x.name.startswith('CON')]
    
    # print('var_G : ', var_G)
    # os.system("pause")

    W_G = [x for x in var_G if x.name.endswith('W:0')]
    
    # print('W_G : ', W_G)
    # os.system("pause")

    W_G_clip = [tf.assign(w_g, tf.clip_by_value(w_g, clamp_lower, clamp_upper)) for w_g in W_G]
        
        
    ################## optimizer

    ########## optimizer_G

    optimizer_G = tf.train.AdamOptimizer(learning_rate = learning_rate_proj, beta1 = adam_beta1_g, beta2 = adam_beta2_g, epsilon = adam_eps_g)
    
    grads_vars_list_G = optimizer_G.compute_gradients(seis_recon_loss, var_list = var_G)
    
    grads_clipped_vars_list_G = [[tf.clip_by_value(gv_group[0], -10., 10.), gv_group[1]] for gv_group in grads_vars_list_G]
    
    train_G_op = optimizer_G.apply_gradients(grads_clipped_vars_list_G)
    
    ########## optimizer_D
    
    optimizer_D = tf.train.AdamOptimizer(learning_rate = learning_rate_dis, beta1 = adam_beta1_d, beta2 = adam_beta2_d, epsilon = adam_eps_g)

    grads_vars_list_D = optimizer_D.compute_gradients(D_loss, var_list = var_D) 
    
    grads_clipped_vars_list_D = [[tf.clip_by_value(gv_group[0], -10., 10.), gv_group[1]] for gv_group in grads_vars_list_D]
    
    train_D_op = optimizer_D.apply_gradients(grads_clipped_vars_list_D)
    
    ########## optimizer_joint
    # optimizer_joint = tf.train.AdamOptimizer(learning_rate = learning_rate_dis, beta1 = adam_beta1_d, beta_2 = adam_beta2_d, epsilon = adam_eps_g)
    
    optimizer_joint = tf.train.AdamOptimizer(learning_rate = learning_rate_dis, beta1 = adam_beta1_d, beta2 = adam_beta2_d, epsilon = adam_eps_g)
    
    grads_vars_list_joint = optimizer_joint.compute_gradients(D_loss_joint, var_list = [var_G, var_D]) 
    
    grads_clipped_vars_list_joint = [[tf.clip_by_value(gv_group[0], -10., 10.), gv_group[1]] for gv_group in grads_vars_list_joint]
    
    train_joint_op = optimizer_D.apply_gradients(grads_clipped_vars_list_joint)

    ########## setup the saver,  max_to_keep是最多保存多少checkpoint文件，多出的话就覆盖掉，默认是5

    saver = tf.train.Saver(max_to_keep = 5) # iters的ckpt最多存20个

    saver_epoch = tf.train.Saver(max_to_keep = 3) # epoch的ckpt最多存50个

    #### setup the image saver

    if output_img > 0:
        num_output_img = min(5, batch_size)
        
        output_ori_imgs_op = (images_tf[0 : num_output_img] + 1) * 127.5
        
        output_incom_imgs_op = (incomplete_images_tf[0 : num_output_img] + 1) * 127.5
        
        output_recon_imgs_op = (images_recon[0 : num_output_img] + 1) * 127.5
        
    ######### initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ######### start training

    print('start training')

    start_time = timeit.default_timer()

    epoch = 0 # initialize epoch
    
    print("epoch = %d" % (epoch)) #不能打逗号
    
    seis_recon_loss_val = 0
    D_loss_real_val = 0
    D_loss_fake_val = 0
    D_loss_val = 0
    D_loss_joint_val = 0
    iters = 0 #initialize
    T_c = 9000
    T_d = 1000
    T_train = 50000
    
    #####alternative training starts
    
    print('alternative training starts...')

    while True:
        ###
        inst_batch, inst_incomplete_batch, inst_mask_c = train_queue.get()
        
        batch = np_combine_batch(inst_batch, ref_batch)
        incomplete_batch = np_combine_batch(inst_incomplete_batch, ref_incomplete_batch)
        mask_c_final = np_combine_batch(inst_mask_c, ref_mask_c)
        
        # update_G
        if iters < T_c :
            
            _, seis_recon_loss_val = sess.run([train_G_op, seis_recon_loss], feed_dict = {images_tf : batch, incomplete_images_tf : incomplete_batch, mask_c : mask_c_final})
        
        # update_D
        elif T_c <= iters <= T_c + T_d:
        
            _, D_loss_real_val, D_loss_fake_val, D_loss_val = sess.run([train_D_op, D_loss_real, D_loss_fake, D_loss], feed_dict = {images_tf : batch, incomplete_images_tf : incomplete_batch, mask_c : mask_c_final})
        
        elif iters > T_c + T_d:
            
            _, D_loss_joint_val = sess.run([train_joint_op, D_loss_joint], feed_dict = {images_tf : batch, incomplete_images_tf : incomplete_batch, mask_c : mask_c_final})
                
        
        #save model once every 100 times
        if iters % 2000 == 0:
            saver.save(sess, model_path + '/model_iter', global_step = iters)
            # filename : 'model_path + /model_iter-iters'
            print('model saved once...')
        if iters % 100 == 0:
            print("Iter %d (%.2f minutes): seis_recon_loss=%.3e D_loss_real=%.3e D_loss_fake=%.3e D_loss=%.3e D_loss_joint=%.3e learning_rate_proj=%.3e learning_rate_dis=%.3e qsize=%d" % 
                    (iters, ((timeit.default_timer()-start_time)/60.), seis_recon_loss_val, D_loss_real_val, D_loss_fake_val, D_loss_val, D_loss_joint_val, learning_rate_proj, learning_rate_dis, train_queue.qsize()))
            # print("Iter %d" % iters)
            # print("%.2f minutes" % ((timeit.default_timer() - start_time)/60.))
            # print("D_loss_real = %.3e" % seis_recon_loss_val)
            # print("D_loss_fake = %.3e" % D_loss_real_val)
            # print("learning_rate_proj = %.3e" % learning_rate_proj)
            # print("learning_rate_dis = %.3e" % learning_rate_dis)
            # print("qsize = %d" % train_queue.qsize())
            
        #save image every 10 iters
        if output_img > 0 and iters % output_img_period == 0:
        # if output_img > 0:
            output_ori_imgs_val, output_incom_imgs_val, output_recon_imgs_val = sess.run([output_ori_imgs_op, output_incom_imgs_op, output_recon_imgs_op], feed_dict = {images_tf : batch, incomplete_images_tf : incomplete_batch, mask_c : mask_c_final})
            
            
            output_folder = '%s/iter_%d' % (img_path, iters)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for i in range(num_output_img):
                #output_ori_imgs_val 有五副
                filename = '%s/ori_%d.jpg' % (output_folder, i)
                
                # output_ori_imgs_val_resp = output_ori_imgs_val[i].reshape(img_size, img_size, 1) # 针对seisdata
                output_ori_imgs_val_resp = output_ori_imgs_val[i].reshape(img_size, img_size)
                
                sp.misc.imsave(filename, output_ori_imgs_val_resp.astype('uint8'))
                
                filename = '%s/incomplete_%d.jpg' % (output_folder, i)
                
                # output_incom_imgs_val_resp = output_incom_imgs_val[i].reshape(img_size, img_size, 1) 
                output_incom_imgs_val_resp = output_incom_imgs_val[i].reshape(img_size, img_size)
                
                
                sp.misc.imsave(filename, output_incom_imgs_val_resp.astype('uint8'))
                
                filename = '%s/recon_%d.jpg' % (output_folder, i)
                
                # output_recon_imgs_val_resp = output_recon_imgs_val[i].reshape(img_size, img_size, 1)
                output_recon_imgs_val_resp = output_recon_imgs_val[i].reshape(img_size, img_size)
                
                
                sp.misc.imsave(filename, output_recon_imgs_val_resp.astype('uint8'))
                
        
        iters = iters + 1
        
        if iters % 10 == 0:
            print('the time of iteration is %d %.2f minutes' % (iters, (timeit.default_timer()-start_time)/60.))
        
        if iters % iters_in_epoch == 0:
        # 一个60000已经跑过一次，存一次模型
            epoch = epoch + 1
            
            saver_epoch.save(sess, epoch_path + '/model_epoch', global_step = epoch)
            
            if iters < T_c :
                learning_rate_proj = learning_rate_proj * 0.95
            
            if T_c <= iters <= T_c + T_d:
                learning_rate_dis = learning_rate_dis * 0.95
            
            if iters > T_c + T_d:
                learning_rate_dis = learning_rate_dis * 0.95
            
            if epoch > num_epochs:
                break
            
        if iters % T_train == 0:
            terminate_train_procs(train_procs)
            
            del train_procs
            
            del train_queue
            
            train_queue = Queue(num_prepare + 1)
            
            train_procs = []
            
    sess.close()
