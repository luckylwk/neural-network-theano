import gzip
import cPickle
import time
import sys
from math import sqrt


import numpy as np

import theano


from .gpu import fn_theano_gpu_shared_dataset



def fn_T_load_data_MNIST( path_to_file ):

	start_time = time.time()
	print 100 * '-', '\n\t    *** LOADING DATASET: MNIST Handwritten Digits'

	try:
		f = gzip.open(path_to_file, 'rb')
		train_set, valid_set, test_set = cPickle.load(f)
		f.close()
	except:
		print '\t\tFailure: FILE DOES NOT EXIST', sys.exc_info()[0]

	# Reshape dimensions.
	# ...

	# Create Theano shared variables (for GPU processing)
	test_set_x, test_set_y = fn_theano_gpu_shared_dataset(test_set)
	valid_set_x, valid_set_y = fn_theano_gpu_shared_dataset(valid_set)
	train_set_x, train_set_y = fn_theano_gpu_shared_dataset(train_set)

	# Print information.
	print '\t\tTime elapsed:      {:.2} seconds'.format( time.time()-start_time )
	print '\t\tTraining set:      {} {} | {}'.format( train_set_x.get_value(borrow=True).shape, train_set[1].shape, type(train_set_x) )
	print '\t\tValidation set:    {} {}'.format( valid_set[0].shape, valid_set[1].shape )
	print '\t\tTesting set:       {} {}'.format( test_set[0].shape, test_set[1].shape )
	dim = int(sqrt(train_set_x.get_value(borrow=True).shape[1]))
	print '\t\tImage Dimensions:  {}-by-{}-by-1'.format(dim,dim)

	# Output is a list of tuples. Each tuple is filled with an m-by-n matrix and an m-by-1 array.
	return [ (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), (dim,dim,1) ]
	##################

