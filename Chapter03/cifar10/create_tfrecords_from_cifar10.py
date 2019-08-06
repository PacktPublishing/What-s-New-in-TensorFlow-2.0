import os
import sys
import glob
import pickle
import shutil
import argparse
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_pickle_from_file(filename):
  """ function reads pickle file"""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict

def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())

def _get_file_names(eval_file_idx):
  """ get the file names for eval """
  file_names = {}
  train_files_idx_list = [1, 2, 3, 4, 5]
  train_files_idx_list.remove(eval_file_idx)
  print('Setting idx \'{}\' as validation/eval data.'.format(eval_file_idx))

  file_names['train'] = ['data_batch_%d' % i for i in train_files_idx_list]
  file_names['eval'] = ['data_batch_'+str(eval_file_idx)]
  file_names['test'] = ['test_batch']
  return file_names

def create_tfrecords(cifar10_data_folder, validation_data_idx):
  """ function to generate tfrecords
  Creates three sub-folders, train, eval, test and put resp 
  tfr files
  """
  batch_files = _get_file_names(validation_data_idx)
  tfrecords_outdir = './tfrecords'
  for data_type in ['train', 'eval', 'test']:
    input_files = [os.path.join(cifar10_data_folder, i) \
        for i in batch_files[data_type]]
    resp_dir = tfrecords_outdir + '/' + data_type
    shutil.rmtree(resp_dir, ignore_errors=True)
    os.makedirs(resp_dir)
    for ifile in input_files:
      batch_file_name = os.path.basename(ifile).split('.')[0]
      tfrecords_outfile_name = \
          os.path.join(tfrecords_outdir, data_type, batch_file_name + '.tfr')
      convert_to_tfrecord([ifile], tfrecords_outfile_name)

def main(argv):
  """ main routing to use dump tfrecords for cifar10 data"""
  if argv is not None:
    print('argv: {}'.format(argv))

  parser = argparse.ArgumentParser()
  parser.add_argument('-vidx', '--validation_data_idx', type=int,
      choices=[1, 2, 3, 4, 5], 
      help='Define model to run', default=5)
  parser.add_argument('-cf', '--cifar10_data_folder', type=str, 
      required=True, help='Cifar10 data folder path')
  args = parser.parse_args()

  validation_data_idx = args.validation_data_idx
  cifar10_data_folder = args.cifar10_data_folder
  create_tfrecords(cifar10_data_folder, validation_data_idx)

if __name__ == '__main__':
  main(sys.argv)
