## Copyright (C) 2017, Nicholas Carlini and Nicolas Papernot.
## All rights reserved.
from __future__ import print_function

# Start off by just disabling tensorflow's warnings.
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    # it's okay if it fails
    pass

# First, we import tensorflow (the ML library we are going to use)
import tensorflow as tf

# Next we'll import numpy and scipy, two common libraries
import numpy as np  # useful to manipulate matrices
import scipy.misc  # useful to load and save images in our case

# Then we import the Inception file, for classifying our images
import inception

# Tensorflow works by maintaining a "session" of the current
# environment. This is where we instantiate it.
# The session contains all of the elements of the graph,
# which is how TensorFlow defines a neural network.
sess = tf.Session()

# First, we set up the Inception model. If this is the first
# time this file is being run, this will also download and
# extract the Inception weights into a temporary folder.
model = inception.setup(sess)

# TensorFlow is a lazy library, where the first step is to
# define a "graph" of how the processing will take place,
# and then only afterwards actually perform the execution.

# Here, we create a "placeholder" for where the image will
# go when we know which image we want to classify. We will
# represent the image as a 32-bit floating point number,
# with width 299 and height 299, with 3 color channels.
image_placeholder = tf.placeholder(tf.float32, (1, 299, 299, 3))

# We take this image and feed it into the model, obtaining
# a set of "logits" -- numbers that correspond to how likely
# any given class may be, although not directly.
logits_tensor = model(image_placeholder)

# We turn these logits into probabilities through a softmax
# activation function. At this point, we have a tensor that
# will hold the probability that the input tensor represents
# any given target label.
probs_tensor = tf.nn.softmax(logits_tensor)

# We're now done defining the graph. It's time to actually
# load up some images and classify them.

# The first image, of a panda, is already in the images folder.
# We load it with scipy
image = scipy.misc.imread("images/adversarial_panda.png")

# Inception wants the images to be 299x299, so resize it.
if image.shape != (299, 299, 3):
    image = np.array(scipy.misc.imresize(image, (299, 299)),
                     dtype=np.float32)

# And finally, convert each pixel to the range [0,1].
image = (image/255.0)

# Now that we have an image ready, let's classify it. To do
# this, we actually run the graph, requesting the probs_tensor
# as output and passing in the image we have to the
# placeholder we defined earlier.
probs = sess.run(probs_tensor, {image_placeholder: [image]})[0]

# Let's look at the 5 most likely classes and report what they
# are, and how confident it is for each.
for index in np.argsort(-probs)[:5]:
    print(str(int(probs[index]*100))+"% confident it is a",
          model.id_to_name[index])
