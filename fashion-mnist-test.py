# test Tensorflow keras
from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Explore the data
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(test_images.shape)
# print(len(test_labels))

# Print one image - doesnt work in terminal mode -ssh  connections
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
# 1. Feed training data do the model
model.fit(train_images, train_labels, epochs=10)

# 2. Evaluate accuracy
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
print('\nTrain accuracy:', train_acc)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)



