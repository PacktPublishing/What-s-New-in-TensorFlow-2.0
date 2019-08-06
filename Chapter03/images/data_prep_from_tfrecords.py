""" Load 2012 iccad dataset from tfrecords"""
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf

class DataPrep(object):
  """ Code to read tfrecords
  """
  def __init__(self, batch_size, tfrecords_path='./tfrecords'):
    """ constructor
    Args:
      batch_size (int): batch size
      tfrecords_path (str): input path for tfrecords files
    """
    self.batch_size = batch_size
    self.tfrecords_path = tfrecords_path
    self.image_height = 1200
    self.image_width = 1200
    self.image_num_channels = 1

    # Create datasets
    self.train_dataset = self.create_datasets(is_train_or_eval_or_test='train')
    self.eval_dataset = self.create_datasets(is_train_or_eval_or_test='eval')
    self.test_dataset = self.create_datasets(is_train_or_eval_or_test='test')

  def create_datasets(self, is_train_or_eval_or_test='train'):
    """ function to create dataset
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
      dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)
    else:
      # Just create batches for eval and test
      dataset = dataset.batch(self.batch_size)
    return dataset

  def _parse_and_decode(self, serialized_example):
    """ parse and decode each image """
    features = tf.io.parse_single_example(
        serialized_example, 
        features={
          'image': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.int64),
          }
        )
    
    image = tf.image.decode_png(features['image'], dtype=tf.uint8)
    image = tf.cast(image, tf.float32)
    # reshape to channels, height, width and 
    # transpose to have width, height, and channels
    image = tf.cast(
        tf.transpose(tf.reshape(
          image, 
          [self.image_num_channels, self.image_height, self.image_width]), 
          [1, 2, 0]), tf.float32)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 2)

    return image, label

def main(argv):
  """ main routing """
  if argv is not None:
    print('argv: {}'.format(argv))

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--tfrecords_outdir', type=str, 
      required=True, help='Input tfrecords folder')
  parser.add_argument('-b', '--batch_size', type=int, 
      required=True, help='Batch size')
  args = parser.parse_args()
  batch_size = args.batch_size
  tfrecords_outdir = args.tfrecords_outdir
  data_prep = DataPrep(batch_size, tfrecords_outdir)
  for images, classes in data_prep.train_dataset:
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  main(sys.argv)
