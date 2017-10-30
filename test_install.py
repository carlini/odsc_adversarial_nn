## Copyright (C) 2017, Nicholas Carlini and Nicolas Papernot.
## All rights reserved.

from __future__ import print_function

print("Verifying that the system setup is correct...")

try:
    import numpy as np
    print("numpy: OK")
except:
    print("Unable to import numpy.")
    print("You should install it with pip install numpy")
    exit(1)

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    # This just disables tensorflow obnoxious messages
    # it's okay if it fails
    pass

try:
    import tensorflow as tf
    print("tensorflow: OK")
except:
    print("Unable to import tensorflow.")
    print("You should install it with pip install tensorflow")
    exit(1)

if tf.__version__[0] != '1':
    print("You have tensorflow installed corretly, but an old version")
    print("The tutorial may or may not work.")
    print("You may want to uninstall and reinstall tensorflow.")
    
try:
    sess = tf.Session()
    if sess.run(tf.constant(3)+tf.constant(5)) != 8:
        print("Tensorflow is not able to do math.")
        print("This should never happen.")
        print("Try reinstalling tensorflow, or using a different version.")
        exit(1)
except:
    print("Unable instantiate a session object.")
    print("Try reinstalling tensorflow, or using a different version.")
    exit(1)

try:
    import cleverhans
    print("cleverhans: OK")
except:
    print("Unable to import cleverhans.")
    print("You should install it with pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans")
    exit(1)

try:
    if cleverhans.__version__[0] != '2':
        print("Upgrade Cleverhans with pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans -upgrade")
        exit(1)
except:
    print("Upgrade Cleverhans with pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans -upgrade")
    exit(1)

try:
    import scipy.misc
    print("scipy: OK")
except:
    print("Unable to import scipy.misc.")
    print("You should install it with pip install scipy")
    exit(1)

try:
    import keras
    print("keras: OK")
    has_keras = True
except:
    print("Unable to import keras.")
    print("You should install it with pip install keras")
    print("This is not strictly necessary, but will greatly improve performance.")
    has_keras = False

import inception
model = inception.setup(sess)
image = scipy.misc.imread("images/panda.png")
if image.shape != (299, 299, 3):
    image = np.array(scipy.misc.imresize(image, (299, 299)),
                     dtype=np.float32)/255.0
preds = model(tf.constant(np.array([image])))
preds = sess.run(preds)

if has_keras:
    if np.argmax(preds) == 388:
        print("Everything is properly installed and set up, with keras.")
        print("You are good to go.")
    else:
        print("Inception with Keras did not properly set up; try uninstalling keras")
else:
    if np.argmax(preds) == 169:
        print("Everything is properly installed and set up, but without keras.")
        print("You are good to go.")
    else:
        print("Inception did not properly set up; try installing keras")
