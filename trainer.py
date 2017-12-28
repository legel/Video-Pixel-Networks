import tensorflow as tf
import numpy as np
from logger import Logger
import os


class Trainer:
    def __init__(self, sess, model, data_generator, config):
        self.sess = sess
        self.model = model
        self.data_generator = data_generator
        self.config = config

        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None

        # init the global step , the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        if not os.path.exists(self.config.summary_dir):
            os.makedirs(self.config.summary_dir)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        self.logger = Logger(self.sess, self.config.summary_dir)

        if self.config.load:
            self.load()

    def save(self, step=0):
        self.saver.save(self.sess, self.config.checkpoint_dir, step)
        Logger.info("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            Logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            Logger.info("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def train(self):
        Logger.info("Training...")
        lstm_input = np.zeros((2, self.config.batch_size, self.config.input_shape[0], self.config.input_shape[1], self.config.conv_lstm_filters))
       
        step = 0
        for epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.epochs_num):
            losses = []
            epoch = self.cur_epoch_tensor.eval(self.sess)
            for itr in range(self.config.iters_per_epoch):
                warmup_batch, train_batch = self.data_generator.next_batch()
                lstm_state = self.sess.run(self.model.final_lstm_state, {self.model.sequences: warmup_batch, self.model.initial_lstm_state: lstm_input}) 
                loss, _ = self.sess.run([self.model.loss, self.model.optimizer], {self.model.sequences: train_batch, self.model.initial_lstm_state: lstm_state}) 
                losses.append(loss)
                  
                step += 1
                if len(losses) > 5:
                    losses.pop(0)

                Logger.info('step #{0}:     loss={1}'.format( step , round(np.mean(losses),5)))
        Logger.info('Saving model...')
        self.save(step)


    def test(self):
        Logger.info("Testing...")

        initial_lstm_state = np.zeros((2, self.config.batch_size, self.config.input_shape[0], self.config.input_shape[1], self.config.conv_lstm_filters))
        warmup_batch, test_batch = self.data_generator.next_batch()
        feed_dict = {self.model.sequences: warmup_batch, self.model.initial_lstm_state: initial_lstm_state}
        lstm_state = self.sess.run(self.model.final_lstm_state, feed_dict)
        prev_frame = test_batch[:, 0, :, :, :]

        for frame in range(self.config.truncated_steps):
            feed_dict = {self.model.inference_prev_frame: prev_frame, self.model.initial_lstm_state: lstm_state}
            encoder_state, lstm_state = self.sess.run([self.model.encoder_state, self.model.inference_lstm_state], feed_dict)
            current_frame = np.zeros([1] + self.config.input_shape)

            for i in range(self.config.input_shape[0]):
                for j in range(self.config.input_shape[1]):
                    feed_dict = {self.model.inference_encoder_state: encoder_state, self.model.inference_current_frame: current_frame}
                    output, summaries = self.sess.run([self.model.inference_output, self.model.test_summaries[frame]], feed_dict)

                    output = np.argmax(output, axis=3)

                    current_frame[:, i, j, 0] = output[:, i, j].copy()

            prev_frame = current_frame.copy()

        Logger.info("Testing finished")
