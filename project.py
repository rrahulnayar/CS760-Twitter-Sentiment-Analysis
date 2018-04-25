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


def lstm_cell():
	global lstm_size
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
	return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

def getBatches(x, y, batchSize=100):
	nBatches = len(x) #batchSize
	x, y = x[:nBatches*batchSize], y[:nBatches*batchSize]
	for i in range(0, len(x), batchSize):
		yield x[i:i+batchSize], y[i:i+batchSize]

with open('./data/train.pkl', 'rb') as train:
	dataTrain = pickle.load(train)

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

	seq_len = 230
	features = np.zeros((len(dataTrain[0]), seq_len), dtype=int)

	for i, row in enumerate(dataTrain[0]):
		features[i, -len(row):] = np.array(row)[:seq_len]
	features[:10,:100]

	# keep a note of this
	splitFraction = 0.8
	splitIndex = int(splitFraction * len(features))
	train_x, val_x = features[:splitIndex], features[splitIndex:]
	train_y, val_y = dataTrain[1][:splitIndex], dataTrain[1][splitIndex:]


	split_frac = 0.5
	split_index = int(split_frac * len(val_x))

	val_x, test_x = val_x[:split_index], val_x[split_index:]
	val_y, test_y = val_y[:split_index], val_y[split_index:]

	# Number of Words
	nWords = {}
	for i in dataTrain[0]:
		for j in i:
			if j not in nWords.keys():
				nWords[j] = 0
			nWords[j] +=1

	#print(len(nWords))



	#LSTM paramerters
	#LSTM size = 256
	#LSTM layers = 2
	#Batch Size = 1000
	#Learning Rate = 0.02
	lstm_size = 256
	lstm_layers = 2
	batchSize = 1000
	learningRate = 0.02
	n_words = len(nWords) + 1 # add 1 for 0 added to the vocabulary

	# Create a graph object
	tf.reset_default_graph()
	with tf.name_scope('inputs'):
		inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
		labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
		keep_prob = tf.placeholder(tf.float32, name="keep_prob")

	embed_size = 300
	with tf.name_scope("Embeddings"):
		embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
		embed = tf.nn.embedding_lookup(embedding, inputs_)

	with tf.name_scope("RNN_layers"):
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
		initial_state = cell.zero_state(batchSize, tf.float32)


'''
	with tf.name_scope("RNN_forward"):
		outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


	with tf.name_scope('predictions'):
		predictions = tf.contrib.layers.fully_connected(outputs[:,-1], 1, activation_fn=tf.sigmoid)
		tf.summary.histogram('predictions', predictions)

	with tf.name_scope('cost'):
		cost = tf.losses.mean_squared_error(labels_, predictions)
		tf.summary.scalar('cost', cost)

	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

	merged = tf.summary.merge_all()

	with tf.name_scope('validation'):
		correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



	epochs = 10

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter('./log/tb/train', sess.graph)
		test_writer = tf.summary.FileWriter('./log/tb/test', sess.graph)
		iteration = 1
		for e in range(epochs):
			state = sess.run(initial_state)

			for i, (x,y) in enumerate(get_batches(train_x, train_y, batchSize), 1):
				feed = {inputs_: x,
						labels_: y[:, None],
						keep_prob: 0.5,
						initial_state: state}

				summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict=feed)

				train_writer.add_summary(summary, iteration)

				if iteration%5==0:
					print("Epoch: {}/{}".format(e, epochs),
						  "Iteration: {}".format(iteration),
						  "Train loss: {:.3f}".format(loss))

				if iteration%25==0:
					val_acc = []
					val_state = sess.run(cell.zero_state(batch_size, tf.float32))
					# checkpoint
					for x, y in get_batches(val_x, val_y, batch_size):
						feed = {inputs_: x,
								labels_: y[:, None],
								keep_prob: 1,
								initial_state: val_state}
	#                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
						summary, batch_acc, val_state = sess.run([merged, accuracy, final_state], feed_dict=feed)
						val_acc.append(batch_acc)
					print("Val acc: {:.3f}".format(np.mean(val_acc)))
				iteration +=1
				test_writer.add_summary(summary, iteration)
				saver.save(sess, "checkpoints/sentiment_manish.ckpt")
		saver.save(sess, "checkpoints/sentiment_manish.ckpt")






'''