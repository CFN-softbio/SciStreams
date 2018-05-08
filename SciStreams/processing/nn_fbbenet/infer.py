# infer from one instance
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from . import model
from .tags import tag_names

from .nn_eval import eval_graph
from .setup_model import images, sigmoid_linear, w_coefs, w_images, local4,\
        IMG_SHAPE, eval_graph

def inference_function(img, checkpoint_filename=None):
    ''' This is the inference function. It will take an image and
        output a few results concerning the results of the CNN.

        Returns
        -------
        A dictionary with the following entries:
            logits : the logits of the sigmoid linear
            w_coefs :
            w_images :
            local4 :
    '''
    if checkpoint_filename is None:
        raise ValueError("Error, need a checkpoint filename for inference")
    # add extra dimension to data
    img = img[np.newaxis, :, :, np.newaxis]
    res = [None, None, None, None]

    # must be done after graph construction
    with eval_graph.as_default():
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

    #try:
    with tf.Session(graph=eval_graph) as sess:
        # need to initialize variables (checkpoint file?)
        saver.restore(sess, checkpoint_filename)
        #sess.run(init)

        feed_dict = {images : img}
        res = sess.run([sigmoid_linear, w_coefs, w_images, local4], feed_dict=feed_dict)
    #except Exception:
    #res = [None, None, None, None]



    # res is a tuple of sigmoid_linear, w_coefs, w_images, local4
    return dict(logits=res[0],
                w_coefs=res[1],
                w_images=res[2],
                local4=res[3])

def reduce_img(**kwargs):
    ''' reduce the image to 227 x 227 for the machine learning inference.'''
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


def normalize_img(image):
    ''' Normalize the image.
    '''
    img = image
    stdimg = np.std(img)
    avgimg = np.average(img)
    img = (img-avgimg)/stdimg
    return dict(image=img)

# these are the steps. it's recommeded calling each separately in your 
# pipeline though to help better understand and optimize
def infer(image, checkpoint_filename=None):
    ''' Run inference function on an image img.
        This algorithm will do three things:
            1. Down sample image to 227 x 227 pixels
            2. normalize image (img - avg(img))/std(img)
            3. run inference function on image
            4. Find the most fitting tag for the image

        NOTE : Must be at least 227x227 in size.

        Returns
        -------
        A dictionary with the following entries:
            logits : the logits outputted by the final sigmoid function
            best_tag_name : the tag name for the most probable tag
            tag_names : the names for all 10 tags used here
                (this is always the same, and useful for reference)
    '''
    img = image
    # reduce image to the size we need which is 227 x 227
    #img = reduce_img(image=img)['image']
    img = resize(img, IMG_SHAPE)
    if tuple(img.shape) != tuple(IMG_SHAPE):
        errormsg = "Image cannot be reshaped to {}".format(IMG_SHAPE)
        errormsg += "\nImg shape : {}".format(img.shape)
        raise ValueError(errormsg)
    # image must also be normalized
    img = normalize_img(img)['image']
    # finally infer
    result = inference_function(img, checkpoint_filename=checkpoint_filename)
    logits = result['logits']
    best_tag = np.argmax(logits)
    best_tag_name = tag_names[best_tag]
    return dict(
            logits=logits,
            best_tag_name=best_tag_name,
            tag_names=tag_names
    )
