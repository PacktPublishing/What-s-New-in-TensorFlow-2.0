# Creation of tfrecords from cifar10 data set

## Step1
### Download data from https://www.cs.toronto.edu/~kriz/cifar.html  
- The data contains 50,000 images for training the network  
- The data contains 10,000 images for testing the network  

### Once you download and untar, you will see folder cifar-10-batches-py and there would be below files:
batches.meta  data_batch_2  data_batch_4  readme.html
data_batch_1  data_batch_3  data_batch_5  test_batch

data_batch_* files are the ones having training data and test_batch has test data. These files are in Python pickle format.

## Step2
### Create tfrecords
For this book's illustration purpose, we will use one of the data_batch_* files as validation data and rest to be training. E.g. if we choose data_batch_4 as validation, then data_batch_1, data_batch_2, data_batch_3, data_batch_5 will be used as training data.

To create tfrecords, please use below command:
```bash
python3 create_tfrecords_from_cifar10.py -cf ./cifar-10-batches-py -vidx 4 
```

## Step3
### Cifar10 model training and checking accuracy on test data2
```bash
python3 cifar10_classification.py
```

# Using tensorflow-datasets as built-in datasets
```bash
python3 cifar10_classification_from_tfds.py
```

