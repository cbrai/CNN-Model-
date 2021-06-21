#importing libraries 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
import os
import numpy as np
 
# Setting the input image pixel value to max
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# rescaling the datasets
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

# Loading the datasets
train_dataset=train.flow_from_directory('train',
                                        target_size=(128,128),
                                        batch_size = 64,
                                        class_mode = 'binary')
validation_dataset = validation.flow_from_directory('test',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'binary')

# Showing the values of the classes .i.e birds:0,drone:1
train_dataset.class_indices

# Adding layers in Sequential model
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #Conv2d and Max-pooling layer
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #Conv2d and Max-pooling layer
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  ##Conv2d and Max-pooling layer
                                  tf.keras.layers.Flatten(),
                                  # Flatten layer
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  # Dense Layer
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  # Sigmoid Activation function
                                  ])
model.summary()
# Compiling the model using loss function, optimizer as RMSprop and metrics as accuracy
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

#Training the model
model_fit=model.fit(train_dataset,
                    epochs=250,
                    steps_per_epoch=15,
                    validation_data=validation_dataset)

# plot the loss
plt.plot(model_fit.history['loss'], label='train loss')
plt.plot(model_fit.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(model_fit.history['accuracy'], label='train acc')
plt.plot(model_fit.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
