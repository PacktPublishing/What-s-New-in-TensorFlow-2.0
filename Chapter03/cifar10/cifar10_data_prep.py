""" Class for house data preparation, feature engineering etc."""
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

class Cifar10Data(object):
  """ Cifar10Data class
    Description: 
      A lot of code is taken from 
      models/tutorials/image/cifar10_estimator/
  """
  def __init__(self, batch_size, tfrecords_path='./tfrecords'):
    self.batch_size = batch_size
    self.tfrecords_path = tfrecords_path
    self.image_height = 32
    self.image_width = 32
    self.image_num_channels = 3
    self.train_data_size = 40000
    self.eval_data_size = 10000
    self.test_data_size = 10000

    # Create datasets
    self.train_dataset = self.create_datasets(is_train_or_eval_or_test='train')
    self.eval_dataset = self.create_datasets(is_train_or_eval_or_test='eval')
    self.test_dataset = self.create_datasets(is_train_or_eval_or_test='test')

  def create_datasets(self, is_train_or_eval_or_test='train'):
    """ function to create trai
    """
    if is_train_or_eval_or_test is 'train':
      tfrecords_files = glob.glob(self.tfrecords_path + '/train/*')
    elif is_train_or_eval_or_test is 'eval':
      tfrecords_files = glob.glob(self.tfrecords_path + '/eval/*')
    elif is_train_or_eval_or_test is 'test':
      tfrecords_files = glob.glob(self.tfrecords_path + '/test/*')

    # Read dataset from tfrecords
    dataset = tf.data.TFRecordDataset(tfrecords_files)

    # Decode/parse 
    dataset = dataset.map(self._parse_and_decode, num_parallel_calls=self.batch_size)

    if is_train_or_eval_or_test is 'train':
      # For training dataset, do a shuffle and repeat
      dataset = dataset.shuffle(10000).repeat().batch(self.batch_size)
    else:
      # Just create batches for eval and test
      dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(self.batch_size)
    return dataset

  def _parse_and_decode(self, serialized_example):
    features = tf.io.parse_single_example(
        serialized_example, 
        features={
          'image': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.int64),
          }
        )
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    # image = image / 255.
    image.set_shape(self.image_num_channels * self.image_height * self.image_width)

    # reshape to channels, height, width and 
    # transpose to have width, height, and channels
    image = tf.cast(
        tf.transpose(tf.reshape(
          image, [self.image_num_channels, self.image_height, self.image_width]), 
          [1, 2, 0]), tf.float32)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 10)

    return image, label

def main(argv):
  """ main routing """
  if argv is not None:
    print('argv: {}'.format(argv))

  batch_size = 128
  tfrecords_path = './tfrecords'
  cifar_data = Cifar10Data(batch_size, tfrecords_path)
  tds = cifar_data.train_dataset
  print(tds)
  for images, classes in tds.take(4):
    print('shape %', images.shape)
    print('shape %', classes.shape)
    #import ipdb; ipdb.set_trace()
    #plt.imshow(np.resize(images[0], [32, 32, 3]), interpolation='nearest')
    #plt.imshow(images[0])
    #plt.show()

if __name__ == '__main__':
  main(sys.argv)
