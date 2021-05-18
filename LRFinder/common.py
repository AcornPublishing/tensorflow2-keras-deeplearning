import numpy as np
import tensorflow as tf


def load_cifar10_data(batch_size):
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

    # scale data using MaxScaling
    xtrain = xtrain.astype(np.float32) / 255
    xtest = xtest.astype(np.float32) / 255

    # convert labels to categorical    
    ytrain = tf.keras.utils.to_categorical(ytrain)
    ytest = tf.keras.utils.to_categorical(ytest)

    # print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
    # print(xtrain.dtype, ytrain.dtype, xtest.dtype, ytest.dtype)

    train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    test_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))

    val_size = xtrain.shape[0] // 10
    train_dataset = train_dataset.shuffle(10000)
    val_dataset = train_dataset.take(val_size).batch(
        batch_size, drop_remainder=True)
    train_dataset = train_dataset.skip(val_size).batch(
        batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(10000).batch(
        batch_size, drop_remainder=True)
    
    return train_dataset, val_dataset, test_dataset


# def build_model():
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Conv2D(
#         name="conv1",
#         filters=6,
#         kernel_size=(5, 5),
#         padding="valid",
#         activation="relu",
#         kernel_initializer="he_normal",
#         input_shape=(32, 32, 3)))
#     model.add(tf.keras.layers.MaxPooling2D(
#         name="pool1",
#         pool_size=(2, 2)))
#     model.add(tf.keras.layers.Conv2D(
#         name="conv2",
#         filters=16,
#         kernel_size=(5, 5),
#         padding="valid",
#         activation="relu",
#         kernel_initializer="he_normal"))
#     model.add(tf.keras.layers.MaxPooling2D(
#         name="pool2",
#         pool_size=(2, 2),
#         strides=(2, 2)))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(
#         name="dense1",
#         units=120,
#         activation="relu",
#         kernel_initializer="he_normal"))
#     model.add(tf.keras.layers.Dense(
#         name="dense2",
#         units=84,
#         activation="relu", 
#         kernel_initializer="he_normal"))
#     model.add(tf.keras.layers.Dense(
#         name="dense3",
#         units=10,
#         activation="softmax",
#         kernel_initializer="he_normal"))
#     return model

class LeNetModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(LeNetModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding="valid",
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding="valid",
            activation="relu",
            kernel_initializer="he_normal")
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2))
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=120,
            activation="relu",
            kernel_initializer="he_normal")
        self.dense2 = tf.keras.layers.Dense(
            units=84,
            activation="relu", 
            kernel_initializer="he_normal")
        self.dense3 = tf.keras.layers.Dense(
            units=10,
            activation="softmax",
            kernel_initializer="he_normal")


    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x        
