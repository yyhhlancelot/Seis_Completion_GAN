import sys
import os
import numpy as np
import glob
import os
import timeit
import scipy as sp
import pickle


# the dataset is put at "dataset_base_path".
# In the folder, there should be three subfolders: "train", "valid", "test", each containing
# the training images, validation images, and test images, respectively.
# Note that these images can also be contained in subfolders.
dataset_base_path = os.path.expanduser("J:/桌面资料/毕设资料/导师分享/Tensor_GAN/OneNet-master-2to3/dataset/mnist")


trainset_path = dataset_base_path + '/' + 'train'
if not os.path.exists(trainset_path):
    os.makedirs(trainset_path)
validset_path = dataset_base_path + '/' + 'valid'
if not os.path.exists(validset_path):
    os.makedirs(validset_path)
testset_path = dataset_base_path + '/' + 'test'
if not os.path.exists(testset_path):
    os.makedirs(testset_path)

trainset_pickle_path = dataset_base_path + '/' + 'train_filename.pickle'
validset_pickle_path = dataset_base_path + '/' + 'valid_filename.pickle'
testset_pickle_path = dataset_base_path + '/' + 'test_filename.pickle'


def load_pickle(pickle_path, dataset_path):
    # if not os.path.exists(pickle_path):

        image_files = []
        for (root, dirs, files) in os.walk(dataset_path):
            # 返回的dirs和files均为List
            # files只是文件名，不是文件地址
            # print('files :', files)
            print('root : ', root)
            
            for idx in range(60000):
            
                data_path = os.path.join(root, files[idx])
                image_files.append(data_path)
            
            # print('image_files : ', image_files)
            # glob.glob可以返回文件地址
            # filenames = glob.glob( os.path.join(files, 'mnist_train_*.png'))
            # filenames = glob.glob(files)
            # may be JPEG, depending on your image files
            

            ## use magic to perform a simple check of the images
            # import magic
            # for filename in filenames:
            #	if magic.from_file(filename, mime=True) == 'image/jpeg':
            #		image_files.append(filename)
            #	else:
            #		print '%s is not a jpeg!' % filename
            #		print magic.from_file(filename)
        # print('filenames : ', filenames)
        if len(image_files) > 0:
            image_files = np.hstack(image_files)

        dataset_filenames = {'image_path':image_files}
        pickle.dump( dataset_filenames, open( pickle_path, "wb" ) )
    # else:
        # dataset_filenames = pickle.load( open( pickle_path, "rb" ) )
        
        return dataset_filenames #输出每个文件的地址


# return a pd object
def load_trainset_path(trainset_path):
    trainset_pickle_path = dataset_base_path + '/' + 'train_filename.pickle'
    return load_pickle(trainset_pickle_path, trainset_path)

def load_validset_path():
    return load_pickle(validset_pickle_path, validset_path)

def load_testset_path():
    return load_pickle(testset_pickle_path, testset_path)


# return a list containing all the filenames
def load_trainset_path_list():

    dataset_base_path = os.path.expanduser("J:/桌面资料/毕设资料/导师分享/Tensor_GAN/OneNet-master-2to3/dataset/mnist")

    trainset_path = dataset_base_path + '/' + 'train'
    return load_trainset_path(trainset_path)['image_path']

def load_validset_path_list():
    return load_validset_path()['image_path']

def load_testset_path_list():
    return load_testset_path()['image_path']

# output image range: [-1,1)!!
def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

    import skimage.io
    import skimage.transform

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    resized_img -= 0.5
    resized_img /= 2.0

    return resized_img #(resized_img - 127.5)/127.5


def cache_batch(trainset, queue, batch_size, num_prepare, rseed=None, identifier=None):

    np.random.seed(rseed)

    current_idx = 0
    n_train = len(trainset)
    trainset.index = list(range(n_train))
    trainset = trainset.ix[np.random.permutation(n_train)]
    idx = 0
    while True:

        # read in data if the queue is too short
        while queue.qsize() < num_prepare:
            start = timeit.default_timer()
            image_paths = trainset[idx:idx+batch_size]['image_path'].values
            images_ori = [load_image( x ) for x in image_paths]
            X = np.asarray(images_ori)
            # put in queue
            queue.put(X) # block until free slot is available
            idx += batch_size
            if idx + batch_size > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                trainset = trainset.ix[np.random.permutation(n_train)]
                idx = 0



def cache_train_batch_cube(queue, batch_size, num_prepare, identifier=None):
    trainset = load_pickle(trainset_path, dataset_path)
    cache_batch(trainset, queue, batch_size, num_prepare)

def cache_test_batch_cube(queue, batch_size, num_prepare, identifier=None):
    testset = load_pickle(testset_path, dataset_path)
    cache_batch(testset, queue, batch_size, num_prepare)


def cache_batch_list_style(trainset, Xlist, batch_size, num_prepare, identifier=None):

    current_idx = 0
    n_train = len(trainset)
    trainset.index = list(range(n_train))
    trainset = trainset.ix[np.random.permutation(n_train)]
    idx = 0
    while True:

        # read in data if the queue is too short
        while len(Xlist) < num_prepare:
            image_paths = trainset[idx:idx+batch_size]['image_path'].values
            images_ori = [load_image( x ) for x in image_paths]
            X = np.asarray(images_ori, dtype=float)
            Xlist.append(X)
            idx += batch_size
            if idx + batch_size > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                trainset = trainset.ix[np.random.permutation(n_train)]
                idx = 0

if __name__ == "__main__":
    trainset = load_trainset_path_list() #
    validset = load_validset_path_list()
    testset = load_testset_path_list()   #
