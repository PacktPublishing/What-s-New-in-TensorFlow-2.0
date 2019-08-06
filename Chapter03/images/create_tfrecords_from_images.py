""" Create tfrecords for image data"""
import sys
import os
import glob
import shutil
import argparse
from tqdm import tqdm

import tensorflow as tf

class ImageTFRecordsCreator(object):
  """ Class
  """
  def __init__(self, num_shards = 1000):
    """ Constructor
      Args:
      num_shards (int): number of sharded tfrecord
    """
    self.num_shards = num_shards

  def _int64_feature(self, value):
    """
    """
    if not isinstance(value, list):
      value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _bytes_feature(self, value):
    """
    """
    if not isinstance(value, list):
      value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  # Read image file
  def _read_image_file(self, filename):
    """
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
      image = f.read()
    d_file = os.path.basename(filename)
    if 'NHS' in d_file:
      label = 0
    elif 'HS' in d_file:
      label = 1
    return image, label

  # Convert sharded data to tfrecord
  def _convert_shard_data_to_tfrecord(self, input_files, output_file):
    """ Convert sharded input files to tfrecords
    """
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
      for input_file in input_files:
        data, label = self._read_image_file(input_file)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': self._bytes_feature(tf.compat.as_bytes(data)),
                'label': self._int64_feature(label)
            }))
        record_writer.write(example.SerializeToString())

  # Convert data to tfrecord
  def _convert_data_to_tfrecord(self, input_files, output_dir):
    """ convert input image files to tfrecords
      Args:
        input_files (list): List of input images to be converted to tfr
        output_dir (str): Output folder where created tfr files will be kept

      Description:
        Shard input_files if they are above threshold and create those many tfr
    """
    num_input_files = len(input_files)
    for i in tqdm(range(0, num_input_files, self.num_shards)):
      sharded_input_files = input_files[i:i+self.num_shards]
      tfr_file_name = os.path.join(output_dir, 'tfr_'+str(i))
      self._convert_shard_data_to_tfrecord(sharded_input_files, tfr_file_name)

  # Get file names for train, validate and test
  def _get_file_names(self, input_folder):
    """ Get file names for train, test and validation
    """
    # Get all the training files
    train_file_names = glob.glob(input_folder+'/train/*.png')
    validation_file_names = glob.glob(input_folder+'/validate/*.png')
    test_file_names = glob.glob(input_folder+'/test/*.png')

    file_names = {}
    file_names['train'] = train_file_names
    file_names['eval'] = validation_file_names
    file_names['test'] = test_file_names
    return file_names

  def create_tfrecords(self, input_folder, tfrecords_outdir):
    """ function to generate tfrecords
    Creates three sub-folders, train, eval, test and put resp 
    tfr files
    """
    raw_files = self._get_file_names(input_folder)
    for dataset_type in ['train', 'eval', 'test']:
      input_files = [fl for fl in raw_files[dataset_type]]
      dataset_resp_dir = os.path.join(tfrecords_outdir, dataset_type)
      shutil.rmtree(dataset_resp_dir, ignore_errors=True)
      os.makedirs(dataset_resp_dir)
      self._convert_data_to_tfrecord(input_files, dataset_resp_dir)

def main(argv):
  """ main routing to create tfrecords for input image data"""
  if argv is not None:
    print('argv: {}'.format(argv))

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_folder', type=str, 
      required=True, help='Input data path')
  parser.add_argument('-o', '--tfrecords_outdir', type=str, 
      required=True, help='Output tfrecords folder')
  parser.add_argument('-s', '--num_shards', type=int,
      default = 1000, help='Num of shards of tfrecords files')
  args = parser.parse_args()

  input_folder = args.input_folder
  tfrecords_outdir = args.tfrecords_outdir
  num_shards = args.num_shards
  tfr_creator = ImageTFRecordsCreator(num_shards)
  tfr_creator.create_tfrecords(input_folder, tfrecords_outdir)

if __name__ == '__main__':
  main(sys.argv)
