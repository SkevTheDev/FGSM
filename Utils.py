import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100

class Utils(object):
   def read_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Cifar100
        # labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        #                 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        #                 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        #                 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        #                 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        #                 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        #                 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        #                 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        #                 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        #                 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        #                 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        #                 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        #                 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        #                 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        #                 'worm']

        # Cifar10
        # labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # MNIST
        labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        # Pre-process data
        img_rows, img_cols, channels = 28, 28, 1 # 32, 32, 3
        num_classes = 10

        x_train = x_train / 255
        x_test = x_test / 255

        x_train = x_train.reshape((-1, img_rows, img_cols, channels))
        x_test = x_test.reshape((-1, img_rows, img_cols, channels))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        print("Data shapes", x_test.shape, y_test.shape, x_train.shape, y_train.shape)
        return x_train, y_train, x_test, y_test



