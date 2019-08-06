""" Cifar10 classification model """
# coding: utf-8
import sys
import os
import shutil
import tensorflow as tf

from cifar10_data_prep import Cifar10Data

class Cifar10Classification(object):
  """ Lower API based prediction
  """
  def __init__(self, batch_size):
    """ constructor
    Args:
      batch_size (int): batch size
    """
    self.batch_size = batch_size
    self.data = Cifar10Data(batch_size=self.batch_size)

  def build_model(self, learning_rate, dropout_rate, model_dir):
    """ build sequential
    Args:
      learning_rate (float): learning rate
      dropout_rate (float): dropout
      model_dir (str): path to store model
    """
    self.model = tf.keras.models.Sequential()
    input_shape = \
        (self.data.image_height, self.data.image_width, \
        self.data.image_num_channels)

    self.model.add(tf.keras.layers.Conv2D(filters=32, 
      kernel_size=[3, 3], padding='same', 
      activation='relu', input_shape=input_shape))

    self.model.add(tf.keras.layers.Conv2D(filters=32, 
      kernel_size=[3, 3], padding='same', 
      activation='relu'))

    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))
    self.model.add(tf.keras.layers.Dropout(0.25))

    self.model.add(tf.keras.layers.Conv2D(filters=64, 
      kernel_size=[3, 3], padding='same', 
      activation='relu'))

    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))
    self.model.add(tf.keras.layers.Dropout(0.25))

    # flatten
    self.model.add(tf.keras.layers.Flatten())

    self.model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    self.model.add(tf.keras.layers.Dropout(0.5))
    self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    self.model.summary()

    self.model.compile(loss='categorical_crossentropy', 
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      metrics=['accuracy'])


  def train(self, num_epochs, steps_per_epoch):
    """ train """
    self.model.fit(self.data.train_dataset, epochs=num_epochs, 
      steps_per_epoch=steps_per_epoch)
    

  def check_test_accuracy(self):
    """ accuracy check either for test
    Description:
      Uses predictions and calculate RMS error
    """
    print('\n# Evaluate on test data')
    results = self.model.evaluate(self.data.test_dataset)
    print('\ntest loss, test acc:', results)

def main(argv):
  """ main routine """
  if argv is not None:
    print('argv: {}'.format(argv))

  batch_size = 256
  num_epochs = 5
  learning_rate = 0.002
  steps_per_epoch = 30

  cifar_classification = Cifar10Classification(batch_size)
  cifar_classification.build_model(model_dir='models/Cifar10ConvModel',
      learning_rate=learning_rate, dropout_rate=0.8)

  cifar_classification.train(num_epochs, steps_per_epoch)
  cifar_classification.check_test_accuracy()

if __name__ == '__main__':
  main(sys.argv)
