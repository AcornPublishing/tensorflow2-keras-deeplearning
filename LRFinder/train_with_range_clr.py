import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import common

def clr_schedule(epoch):
    num_epochs = 10
    mid_pt = int(num_epochs * 0.5)
    last_pt = int(num_epochs * 0.9)
    # min_lr, max_lr = 1e-6, 3e-1
    min_lr, max_lr = 1e-6, 1e-1
    if epoch <= mid_pt:
        return min_lr + epoch * (max_lr - min_lr) / mid_pt
    elif epoch > mid_pt and epoch <= last_pt:
        return max_lr - ((epoch - mid_pt) * (max_lr - min_lr)) / mid_pt
    else:
        return min_lr / 2

# plot the points
# epochs = [x+1 for x in np.arange(10)]
# clrs = [clr_schedule(x) for x in epochs]
# plt.plot(epochs, clrs)
# plt.show()

batch_size = 128

train_dataset, val_dataset, test_dataset = common.load_cifar10_data(batch_size)

model = common.LeNetModel()

min_lr, max_lr = 1e-6, 3e-1
optimizer = tf.keras.optimizers.SGD(learning_rate=min_lr)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(clr_schedule)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=val_dataset,
    callbacks=[lr_scheduler])
model.evaluate(test_dataset)
