## Copyright (C) 2017, Nicholas Carlini and Nicolas Papernot.
## All rights reserved.
id_to_name = None

def setup(sess):
    global id_to_name
    try:
        import keras
        import imagenet_labels_keras
        id_to_name = imagenet_labels_keras.id_to_name
        return setup_keras(sess)
    except:
        import imagenet_labels
        id_to_name = imagenet_labels.id_to_name
        return setup_pure_tf(sess)

def setup_keras(sess):
    import keras
    print("Load model")
    keras.backend.set_learning_phase(False)
    keras.backend.set_session(sess)
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet')
    print("Done loading model")
    def wrap(xs):
        return tf.log(model(xs*2-1))
    return model

def setup_pure_tf(sess):
    import setup_inception
    setup_inception.setup(sess)
    return setup_inception.InceptionModel(sess)
