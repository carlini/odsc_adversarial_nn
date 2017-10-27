## Copyright (C) 2017, Nicholas Carlini and Nicolas Papernot.
## All rights reserved.


# Start off by just disbaling tensorflow's obnoxious warnings.
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    # it's okay if it fails
    pass

# First, we import tensorflow (the library we are going to use)
import tensorflow as tf

# Next we'll import numpy and scipy, two common libraries
import numpy as np
import scipy.misc

# Then we import the Inception file, for classifying our images
import setup_inception
# And a mapping of the numeric IDs to human readable names
from imagenet_labels import id_to_name

# Tensorflow works by maintaining a "session" of the current
# environment. This is where we instantiate it.
sess = tf.Session()

# First, we set up the Inception model. If this is the first
# time this file is being run, this will also download and
# extract the Inception weights into a temporary folder.
setup_inception.setup()

# Now we create the inception model.
model = setup_inception.InceptionModel(sess)

# Now we're going to attack it.
# Begin by importing the cleverhans library, which we use to
# generate adversarial examples
import cleverhans.attacks

# Cleverhans has it's own model interface, so we're going to
# need to wrap the model we have with the callable wrapper.
# This lets us tell cleverhans that we are passing it the
# logits layer with Inception (as opposed to the probs).
from cleverhans.model import CallableModelWrapper

cleverhans_model = CallableModelWrapper(model, 'logits')

# We are going to use a very simple attack to start off.
# Begin by constructing an instance of the attack object.
fgsm = cleverhans.attacks.FastGradientMethod(cleverhans_model, 'tf', sess)

# We're now done defining the graph. It's time to actually
# load up some images and generate adversarial examples.

# The first image, of a panda, is already in the images folder.
# We load it with scipy
image = scipy.misc.imread("images/panda.jpg")

# Inception wants the images to be 299x299, so resize it.
image = np.array(scipy.misc.imresize(image, (299, 299)),
                 dtype=np.float32)

# And finally, convert each pixel to the range [0,1].
image = (image/255.0)

# The fast gradient (sign) method constructs adversarial
# examples by slightly moving the image in the direction
# of the gradient, to maximize the loss of the network.
target = np.zeros((1,1008))
target[0,260] = 1
adversarial_example = fgsm.generate_np(np.array([image]),
                                       y_target=target,
                                       eps=.1)

scipy.misc.imsave("images/adversarial_panda.png", adversarial_example[0])
