import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import common

class LRFinder(object):
    def __init__(self, model, optimizer, loss_fn, dataset):
        super(LRFinder, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = dataset
        # placeholders
        self.lrs = None
        self.loss_values = None
        self.min_lr = None
        self.max_lr = None
        self.num_iters = None


    @tf.function
    def train_step(self, x, y, curr_lr):
        tf.keras.backend.set_value(self.optimizer.lr, curr_lr)
        with tf.GradientTape() as tape:
            # forward pass
            y_ = self.model(x)
            # external loss value for this batch
            loss = self.loss_fn(y, y_)
            # add any losses created during forward pass
            loss += sum(self.model.losses)
            # get gradients of weights wrt loss
            grads = tape.gradient(loss, self.model.trainable_weights)
        # update weights
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


    def range_test(self, min_lr, max_lr, num_iters, debug):
        # create learning rate schedule
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iters = num_iters
        self.lrs =  np.linspace(
            self.min_lr, self.max_lr, num=self.num_iters)
        # initialize loss_values
        self.loss_values = []
        curr_lr = self.min_lr
        for step, (x, y) in enumerate(self.dataset):
            if step >= self.num_iters:
                break
            loss = self.train_step(x, y, curr_lr)
            self.loss_values.append(loss.numpy())
            if debug:
                print("[DEBUG] Step {:d}, Loss {:.5f}, LR {:.5f}".format(
                    step, loss.numpy(), self.optimizer.learning_rate.numpy()))
            curr_lr = self.lrs[step]


    def plot(self):
        plt.plot(self.lrs, self.loss_values)
        plt.xlabel("learning rate")
        plt.ylabel("loss")
        plt.title("Learning Rate vs Loss ({:.2e}, {:.2e}, {:d})"
            .format(self.min_lr, self.max_lr, self.num_iters))
        plt.xscale("log")
        plt.grid()
        plt.show()


if __name__ == "__main__":

    tf.random.set_seed(42)

    # model
    model = common.LeNetModel()
    model.build(input_shape=(None, 32, 32, 3))
    # optimizer
     optimizer = tf.keras.optimizers.SGD()
    # loss_fn
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # dataset
    batch_size = 128
    dataset, _, _ = common.load_cifar10_data(batch_size)
    # min_lr, max_lr
    min_lr = 1e-6
    max_lr = 3
    # min_lr = 1e-2
    # max_lr = 1
    # compute num_iters (Keras fit-one-cycle used 1 epoch by default)
    dataset_len = 45000
    batch_size = 128
    num_iters = dataset_len // batch_size
    # declare LR Finder
    lr_finder = LRFinder(model, optimizer, loss_fn, dataset)
    lr_finder.range_test(min_lr, max_lr, num_iters, debug=True)
    lr_finder.plot()
