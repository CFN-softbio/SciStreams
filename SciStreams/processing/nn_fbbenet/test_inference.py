import numpy as np
import tensorflow as tf
import model



CHKPT_DIR = "/home/lhermitte/research/projects/sidl/checkpoint-files"
#CHKPT_FILENAME = CHKPT_DIR + "/image/model.ckpt-93000.index"
#CHKPT_FILENAME = CHKPT_DIR + "/fbb/model.ckpt-179000.index"
#CHKPT_FILENAME = CHKPT_DIR + "/joint/model.ckpt-82000.index"
CHKPT_FILENAME = CHKPT_DIR + "/fbbenet/model.ckpt-50000.index"#/model.ckpt-82000.index"
#checkpoint
#CHKPT_FILENAME = CHKPT_DIR + "/fbbenet/checkpoint"
#model.ckpt-50000.data-00000-of-00001
#CHKPT_FILENAME = CHKPT_DIR + "/fbbenet/model.ckpt-50000.meta"
#events.out.tfevents.1505871423.sn-nvda7
#model.ckpt-50000.index

CHKPT_FILENAME = CHKPT_DIR + "/fbbenet/model.ckpt-50000"

from setup_model import images, sigmoid_linear, w_coefs, w_images, local4,\
        IMG_SHAPE

def inference_function(img):
    # add extra dimension to data
    img = img[np.newaxis, :, :, np.newaxis]

    # must be done after graph construction
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # need to initialize variables (checkpoint file?)
        #saver.restore(sess, CHKPT_FILENAME)
        sess.run(init)

        feed_dict = {images : img}
        res = sess.run([sigmoid_linear, w_coefs, w_images, local4], feed_dict=feed_dict)

    # res is a tuple of sigmoid_linear, w_coefs, w_images, local4
    return dict(sigmoid_linear=res[0],
                w_coefs=res[1],
                w_images=res[2],
                local4=res[3])

def reduce_img(**kwargs):
    ''' reduce the image to 256x256 for the machine learning inference.'''
    from skimage.transform import downscale_local_mean
    img = kwargs['image']
    desired_dims = IMG_SHAPE
    #desired_dims = 48, 48
    cts_img = np.ones_like(img)

    # just clip the edges
    # later, could keep edge information by averaging
    edgey, edgex = img.shape[0] - img.shape[0]%desired_dims[0],\
            img.shape[1] - img.shape[1]%desired_dims[1]
    facy, facx = img.shape[0]//desired_dims[0],\
            img.shape[1]//desired_dims[1]

    # this subimg should downscale to 256
    subimg = img[:edgey, :edgex]
    cts_img = cts_img[:edgey, :edgex]
    down_img = downscale_local_mean(subimg, (facy, facx), cval=0)
    cts = downscale_local_mean(cts_img, (facy, facx), cval=0)

    w = np.where(cts != 0)
    down_img[w] /= cts[w]
    w = np.where(cts == 0)
    down_img[w] = 0

    return dict(image=down_img)


def normalize_img(img):
    stdimg = np.std(img)
    avgimg = np.average(img)
    img = (img-avgimg)/stdimg
    return img


# get the data somewhere (bluesky etc)
from SciStreams.interfaces.databroker.databases import databases
cmsdb = databases['cms:data']
hdr = cmsdb['3e502037-9625-49cb-84c9-549840745119']

detector_key = 'pilatus2M_image'
img = cmsdb.get_images(hdr, detector_key)[0]

# reduce to 256x256
img = reduce_img(image=img)['image']
img = normalize_img(img)

#img = np.ones((227, 227))
#img = np.ones((1, 227, 227, 1))

res = inference_function(img)
#tres = test_inference_function(img)

from pylab import *
ion()

figure(2);clf();
imshow(img);clim(0,100)
