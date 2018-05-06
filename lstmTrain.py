#!/usr/bin/python
import random
import string
import os
import sys
import math
import pylab
import pickle
import tensorflow as tf
import numpy as np
import argparse

def lstm_cell(lstm_size,keep_prob):
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
	return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

def rnn_cell(lstm_size,keep_prob):
	rnn = tf.contrib.rnn.BasicRNNCell(lstm_size, reuse=tf.get_variable_scope().reuse)
	return tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob=keep_prob)

def get_batches(x, y, batchSize=100):
	nBatches = len(x) #batchSize
	x, y = x[:nBatches*batchSize], y[:nBatches*batchSize]
	for i in range(0, len(x), batchSize):
		yield x[i:i+batchSize], y[i:i+batchSize]

# with open('./data/train.pkl', 'rb') as train:
# 	dataTrain = pickle.load(train)

with open('./data/train.pkl', 'rb') as train:
	dataTrain = pickle.load(train)
	# dataTrain has the reviews and the sentiment
	# Size of dataTrain : 100000
	# dataTrain[0] has the integer input for the review
	# datatrain[1] has the interger ouput for the labels
	# for the time being "0" is marked as positive sentiment and "1" as negative sentiment
	# WARNING: need to verify this


	# No review found with zero length
	##counterForReviewWithZeroLenght = 0
	##for count, element in enumerate(dataTrain[0]):
	##	if(len(element)==0):
	##		counterForReviewWithZeroLenght += 1
	##print (counterForReviewWithZeroLenght)

	# Max size of Review
	maxSizeR = 0
	maxSizeC = 0
	for count, element in enumerate(dataTrain[0]):
		if(len(element)>maxSizeR):
			maxSizeR = len(element)
			maxSizeC = count

	#print(maxSizeR)
	#print(maxSizeC)
	# Max Size of Input Review : 228
	# First index for the max size : 11357

	parser = argparse.ArgumentParser()
	parser.add_argument('-units', type=int, dest='numUnits', default=256)
	parser.add_argument('-layers', type=int, dest='numLayers', default=2)
	parser.add_argument('-batch', type=int, dest='batchSize', default=100)
	parser.add_argument('-lr', type=float, dest='lr', default=0.02)
	parser.add_argument('-ntype', type=str, dest='ntype', default='lstm')
	parser.add_argument('-optimizer', type=str, dest='opt', default='adam')
	parser.add_argument('-epoch', type=int, dest='epoch', default=10)
	args = parser.parse_args()

	useSGD = (args.opt == 'sgd')
	memoryNDtype = args.ntype
	lstm_size = args.numUnits
	lstm_layers = args.numLayers
	batchSize = args.batchSize
	learningRate = args.lr
	epochs = args.epoch
	lrDecayStep = 1
	lrDecayRate = 0.8

	config = memoryNDtype + '_' + str(lstm_size) + 'u_' + str(lstm_layers) + 'l_' + str(batchSize) + 'b_' + args.opt + '_' + str(learningRate) + 'lr'

	logFile = config + '_log.txt'
	fpLog = open(logFile, 'w')

	lossFile = config + '_loss.txt'
	fpLoss = open(lossFile, 'w')

	print (config)
	if not os.path.exists(config):
		os.makedirs(config)

	seq_len = 100
	features = np.zeros((len(dataTrain[0]), seq_len), dtype=int)
	labelsRaw = np.expand_dims(np.array(dataTrain[1]), axis=1)
	for i, row in enumerate(dataTrain[0]):
		features[i, -len(row):] = np.array(row)[:seq_len]
	features[:10,:100]

	# keep a note of this
	splitFraction = 0.8
	splitIndex = int(splitFraction * features.shape[0])
	train_x, val_x = features[:splitIndex], features[splitIndex:]
	train_y, val_y = labelsRaw[:splitIndex], labelsRaw[splitIndex:]


	split_frac = 0.5
	split_index = int(split_frac * len(val_x))

	val_x, test_x = val_x[:split_index], val_x[split_index:]
	val_y, test_y = val_y[:split_index], val_y[split_index:]

	print ('Train Size ', train_x.shape)
	print ('Val Size', val_x.shape)
	print ('Test Size ', test_x.shape)

	#shuffle data
	p = np.random.permutation(len(train_x))
	train_x = train_x[p]
	train_y = train_y[p]

	# Number of Words
	# nWords = {}
	# for i in dataTrain[0]:
	# 	for j in i:
	# 		if j not in nWords.keys():
	# 			nWords[j] = 0
	# 		nWords[j] +=1

	#print(len(nWords))
	#LSTM paramerters

	dictFile = open('./data/dictionary.pkl', 'rb')
	nWords = pickle.load(dictFile)
	n_words = len(nWords) + 2 # add 1 for 0 added to the vocabulary

	os.chdir(config)
	# Create a graph object
	tf.reset_default_graph()
	with tf.name_scope('inputs'):
		inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
		labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
		keep_prob = tf.placeholder(tf.float32, name="keep_prob")

	embed_size = 512
	with tf.name_scope("Embeddings"):
		embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
		embed = tf.nn.embedding_lookup(embedding, inputs_)

	with tf.name_scope("RNN_layers"):
		if memoryNDtype == 'lstm':
			cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size, keep_prob) for _ in range(lstm_layers)])
			initial_state = cell.zero_state(batchSize, tf.float32)

		elif memoryNDtype == 'rnn':
			cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(lstm_size, keep_prob) for _ in range(lstm_layers)])
			initial_state = cell.zero_state(batchSize, tf.float32)

		elif memoryNDtype == 'blstm':
			fwdCell = lstm_cell(lstm_size, keep_prob)
			initialStateFw = fwdCell.zero_state(batchSize, tf.float32)
			bwdCell = lstm_cell(lstm_size, keep_prob)
			initialStateBw = bwdCell.zero_state(batchSize, tf.float32)
			#initial_state = tf.concat([initialStateFw, initialStateBw],2)


	with tf.name_scope("RNN_forward"):
		if memoryNDtype == 'blstm':
			outputBi, final_state = tf.nn.bidirectional_dynamic_rnn(fwdCell, bwdCell, embed, initial_state_fw=initialStateFw, initial_state_bw=initialStateBw)
			outputs = tf.concat(outputBi, 2)
		else:
			outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


	with tf.name_scope('predictions'):
		predictions = tf.contrib.layers.fully_connected(outputs[:,-1], 1, activation_fn=tf.sigmoid)
		tf.summary.histogram('predictions', predictions)

	with tf.name_scope('cost'):
		cost = tf.losses.mean_squared_error(labels_, predictions)
		tf.summary.scalar('cost', cost)

	with tf.name_scope('train'):
		global_step = tf.Variable(0, trainable=False)
		lrDecayStep = lrDecayStep * (train_x.shape[0]/batchSize)
		learning_rate = tf.train.exponential_decay(learningRate, global_step, lrDecayStep, lrDecayRate, staircase=True)
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
			if useSGD:
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=True).minimize(cost, global_step = global_step)
			else:
				optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

	merged = tf.summary.merge_all()

	with tf.name_scope('validation'):
		rPrediction = tf.cast(tf.round(predictions), tf.int32)
		correct_pred = tf.equal(rPrediction, labels_)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		acc, acc_op = tf.metrics.accuracy(labels=labels_, predictions=rPrediction)
		rec, rec_op = tf.metrics.recall(labels=labels_, predictions=rPrediction)
		pre, pre_op = tf.metrics.precision(labels=labels_, predictions=rPrediction)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_writer = tf.summary.FileWriter('./log/tb/train', sess.graph)
		test_writer = tf.summary.FileWriter('./log/tb/test', sess.graph)
		iteration = 1
		for e in range(epochs):
			sess.run(tf.local_variables_initializer())
			if memoryNDtype == 'blstm':
				stateFw = sess.run(initialStateFw)
				stateBw = sess.run(initialStateBw)
			else:
				state = sess.run(initial_state)

			for i, (x,y) in enumerate(get_batches(train_x, train_y, batchSize), 1):

				if memoryNDtype == 'blstm':
					feed = {inputs_: x,
	                        labels_: y[:],
							keep_prob: 0.5,
							initialStateFw: stateFw,
							initialStateBw: stateBw}
					summary, loss, stateFw, stateBw, _ = sess.run([merged, cost, final_state[0], final_state[1], optimizer], feed_dict=feed)
				else:
					feed = {inputs_: x,
	                        labels_: y[:],
							keep_prob: 0.5,
							initial_state: state}

					summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict=feed)

				train_writer.add_summary(summary, iteration)

				if iteration%5==0:
					print("Epoch: {}/{}".format(e, epochs),
						  "Iteration: {}".format(iteration),
						  "Train loss: {:.3f}".format(loss))
					lsstr = "Epoch: " + "{}/{}".format(e, epochs) + "Iteration: "+"{}".format(iteration) + "Train loss: " + "{:.3f}".format(loss) + '\n'
					fpLoss.write(lsstr)

				iteration +=1
			val_acc = []

			if memoryNDtype == 'blstm':
				stateFwVal = sess.run(initialStateFw)
				stateBwVal = sess.run(initialStateBw)
			else:
				val_state = sess.run(cell.zero_state(batchSize, tf.float32))
			# checkpoint
			for x, y in get_batches(val_x, val_y, batchSize):
				if memoryNDtype == 'blstm':
					feed = {inputs_: x,
							labels_: y[:],
							keep_prob: 1,
							initialStateFw: stateFwVal,
							initialStateBw: stateBwVal}
					summary, batch_acc, stateFwVal,stateBwVal, prec, recall, accr  = sess.run([merged, accuracy, final_state[0], final_state[1], pre_op,rec_op, acc_op], feed_dict=feed)
				else:
					feed = {inputs_: x,
							labels_: y[:],
							keep_prob: 1,
							initial_state: val_state}
#                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
					summary, batch_acc, val_state, prec, recall, accr = sess.run([merged, accuracy, final_state, pre_op, rec_op, acc_op], feed_dict=feed)
				val_acc.append(batch_acc)
			print("Val acc: {:.3f}".format(np.mean(val_acc)))
			print ("Acc: ",str(accr) , "Precision: ", str(prec), "Recall: ", str(recall))
			logStr = 'Itr ' + str(e+1) + ' Val acc: ' + str(accr) + " Val Precision: " + str(prec) + "Val Recall: "+ str(recall) +'\n'
			fpLog.write(logStr)
			fpLog.flush()

			test_writer.add_summary(summary, iteration)
			saver.save(sess, "checkpoints/sentiment.ckpt")
		saver.save(sess, "checkpoints/sentiment.ckpt")

		test_acc = []
		with tf.Session() as sess:
			saver.restore(sess, "checkpoints/sentiment.ckpt")
			sess.run(tf.local_variables_initializer())
			if memoryNDtype == 'blstm':
				stateFwVal = sess.run(initialStateFw)
				stateBwVal = sess.run(initialStateBw)
				for ii, (x, y) in enumerate(get_batches(test_x, test_y, batchSize), 1):
					feed = {inputs_: x,
							labels_: y[:],
							keep_prob: 1,
							initialStateFw: stateFwVal,
							initialStateBw: stateBwVal}
					batch_acc, stateFwVal, stateBwVal,prec, recall, accr = sess.run([accuracy, final_state[0], final_state[1],pre_op,rec_op, acc_op], feed_dict=feed)
					test_acc.append(batch_acc)
			else:
				test_state = sess.run(cell.zero_state(batchSize, tf.float32))
				for ii, (x, y) in enumerate(get_batches(test_x, test_y, batchSize), 1):
					feed = {inputs_: x,
					labels_: y[:],
					keep_prob: 1,
					initial_state: test_state}
					batch_acc, test_state, prec, recall, accr = sess.run([accuracy, final_state, pre_op,rec_op, acc_op], feed_dict=feed)
					test_acc.append(batch_acc)
			print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
			print ("Acc: ",str(accr) , "Precision: ", str(prec), "Recall: ", str(recall))
			logStr = 'Test acc: ' + str(accr) + " Test Precision: " + str(prec) + "Test Recall: "+ str(recall) +'\n'
			fpLog.write(logStr)
		fpLog.close()
