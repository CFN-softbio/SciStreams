# do this only once
import tensorflow as tf
from . import model

IMG_SHAPE = 227, 227
eval_graph = tf.Graph()
with eval_graph.as_default():
    images = tf.placeholder(tf.float32, (1, IMG_SHAPE[0], IMG_SHAPE[1], 1))
    sigmoid_linear, w_coefs, w_images, local4 = \
        model.inference(images=images,architecture='cnn')
