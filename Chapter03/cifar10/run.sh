python3 create_tfrecords_from_cifar10.py -cf ./cifar-10-batches-py -vidx 4 
python3 run_cifar_classification.py -l 0.002 -e 5 -spe 30 -b 256 -do 1
