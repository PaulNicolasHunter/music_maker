"""
Implement the LSTM logic in Oops manner,

points to remember:

-> timings of drops
	the sample data of the drops
	we can slice it

	( 3 * 60 + 52 ) * 1000 = 23,20,00 milisecs
	44100 -> 1000 mil

	1 -> 1000 / (44100 => frame rate)

	1 => 0.02267573696145124716553287981859 => X

	0.02267573696145124716553287981859 * 10273180 must be equal to 232000

	but it is equal to 232951.92743764172335600907029474 ~~ 232000


	let's find the audio quality with the original frame rate

	X = 232000 / 10273180
	X = 0.02258307554233450596602025857621
	f = 1000 / X
	f = 44280.94827586206896551724137932
	4109 batches
"""

import numpy as np
from scipy.io.wavfile import read, write
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EdmMaker:
	def __init__(self, filepath):
		self.input_size = 3
		self.output_size = 3
		self.num_steps = 25
		self.filepath = filepath
		self.learning_rate = 0.00252
		self.f_bias = 0.0021
		self.num_iterations = 1
		self.neurons = 400
		self.inpt = tf.placeholder(dtype=tf.float64,
								   shape=[None, self.num_steps,
										  self.input_size])  # how may rows in each cell/n-cell/column

		self.oupt = tf.placeholder(dtype=tf.float64, shape=[None, self.num_steps, self.output_size])

	def read_wav(self):
		music = np.array(list(read(self.filepath)[1]))

		starting = len(music)

		st_pt = len(music) % self.num_steps

		music = music[st_pt:]

		self.full = int(len(music) / self.num_steps)

		# init = 0

		# for _ in music[:]:
		#
		# 	if _[0] == 0 and _[1] == 0:
		# 		music.remove(_)
		# 		init = init + 1
		#
		# 	else:
		# 		break

		frame_len = 0.0226  # pretty close to the real one

		time_tags = np.array([float(round(_ * frame_len, 2)) for _ in range(
			st_pt, starting)]).reshape([-1, 1])

		self.mus_inp = np.append(music[:-1], time_tags[:-1], 1)
		self.mus_inp = np.append(self.mus_inp, [0, 0, 0]).reshape([-1, self.num_steps, 3])

		self.mus_target = np.append(music[1:], time_tags[1:], 1)
		self.mus_target = np.append(self.mus_target, [0, 0, 0]).reshape([-1, self.num_steps, 3])

	def network_creation(self):
		cell = tf.contrib.rnn.OutputProjectionWrapper(

			tf.nn.rnn_cell.LSTMCell(self.neurons, activation=tf.nn.relu, forget_bias=self.f_bias ),

			output_size=self.output_size)

		output, state = tf.nn.dynamic_rnn(cell, self.inpt, dtype=tf.float64)

		# self.loss = tf.reduce_mean(tf.square(self.oupt - output))
		self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=self.oupt)

		optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

		train_ = optimiser.minimize(self.loss)

		return train_

	def train(self):
		self.read_wav()

		network = self.network_creation()

		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			datlen = len(self.mus_inp)

			for i in range(self.num_iterations):

				for seri_x in range(datlen):
					sess.run(network, feed_dict={self.inpt: self.mus_inp[seri_x].reshape([1, -1, 3]),
												 self.oupt: self.mus_target[seri_x].reshape([1, -1, 3])})

					print(' {} / {} '.format(seri_x, datlen))

					error = self.loss.eval(feed_dict={self.inpt: self.mus_inp[seri_x].reshape([1, -1, 3]),
													  self.oupt: self.mus_target[seri_x].reshape([1, -1, 3])})

					print('loss => ', error)

		saver.save(sess, "graphs/music_mix")

	def get_music(self):
		pass


em = EdmMaker(r'music\wav_vers\anc.wav')
em.train()
# em.read_wav()
