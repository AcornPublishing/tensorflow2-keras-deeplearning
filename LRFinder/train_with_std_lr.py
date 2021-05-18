import tensorflow as tf

import common

learning_rate = 0.001
batch_size = 128

train_dataset, val_dataset, test_dataset = common.load_cifar10_data(batch_size)

model = common.LeNetModel()
model.build(input_shape=(None, 32, 32, 3))
model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
model.evaluate(test_dataset)
