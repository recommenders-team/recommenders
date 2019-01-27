"""define base class model"""
import abc
import tensorflow as tf
from ..deeprec_utils import *
import numpy as np
import os, time, collections



__all__ = ["BaseModel"]


class BaseModel(object):

    def __init__(self, hparams, iterator_creator, graph=None):
        """
        Initializing the model. Create common logics which are needed by all deeprec models, such as loss function, parameter set.
        :param hparams:  a tf.contrib.training.HParams object, hold the entire set of hyperparameters.
        """
        if not graph:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        self.iterator = iterator_creator(hparams, self.graph)

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name='layer_keeps')
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.placeholder(tf.bool, shape=(), name='is_training')

            self.initializer = self._get_initializer()

            self.logit = self._build_graph()
            self.pred = self._get_pred(self.logit, self.hparams.method)

            self.loss = self._get_loss()
            self.saver = tf.train.Saver(max_to_keep=self.hparams.epochs)
            self.update = self._build_train_opt()
            self.init_op = tf.global_variables_initializer()
            self.merged = self._add_summaries()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)




    def _get_loss(self):
        """
        Make loss function, consists of data loss and regularization loss
        :return:  loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        return self.loss

    def _get_pred(self, logit, task):
        """
        Make final output as prediction score, according to different tasks.
        :param logit: base prediction value
        :param task:  regression or classification
        :return: transformed score
        """
        if task == 'regression':
            pred = tf.identity(logit)
        elif task == 'classification':
            pred = tf.sigmoid(logit)
        else:
            raise ValueError("method must be regression or classification, but now is {0}".format(task))
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _l2_loss(self):
        """
        Construct L2-norm for loss function.
        Returns:
            l2 regular terms
        """
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param)))
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param)))
        return l2_loss

    def _l1_loss(self):
        """
        Construct L1-norm for loss function.
        Returns:
            l1 regular terms
        """
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(l1_loss, tf.multiply(self.hparams.embed_l1, tf.norm(param, ord=1)))
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(l1_loss, tf.multiply(self.hparams.layer_l1, tf.norm(param, ord=1)))
        return l1_loss

    def _cross_l_loss(self):
        """
        Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1)))
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2)))
        return cross_l_loss

    def _get_initializer(self):
        """
        Get an initializer method for parameters.
        Returns:
            An initializer according to config.
        """
        if self.hparams.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=self.hparams.init_value)
        elif self.hparams.init_method == 'uniform':
            return tf.random_uniform_initializer(-self.hparams.init_value, self.hparams.init_value)
        elif self.hparams.init_method == 'normal':
            return tf.random_normal_initializer(stddev=self.hparams.init_value)
        elif self.hparams.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif self.hparams.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif self.hparams.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_IN', uniform=False)
        elif self.hparams.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_IN', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=self.hparams.init_value)

    def _compute_data_loss(self):
        """
        Compute data loss, currently supporting cross_entropy_loss, square_loss and log_loss.
        :return:
        """
        if self.hparams.loss == 'cross_entropy_loss':
            data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                logits=tf.reshape(self.logit, [-1]), \
                labels=tf.reshape(self.iterator.labels, [-1])))
        elif self.hparams.loss == 'square_loss':
            data_loss = tf.sqrt(tf.reduce_mean(
                tf.squared_difference(tf.reshape(self.pred, [-1]), tf.reshape(self.iterator.labels, [-1]))))
        elif self.hparams.loss == 'log_loss':
            data_loss = tf.reduce_mean(tf.losses.log_loss(predictions=tf.reshape(self.pred, [-1]),
                                                          labels=tf.reshape(self.iterator.labels, [-1])))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _compute_regular_loss(self):
        """
        Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        regular_loss = tf.reduce_sum(regular_loss)
        return regular_loss

    def _train_opt(self):
        """
        Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == 'adadelta':
            train_step = tf.train.AdadeltaOptimizer(
                lr)
        elif optimizer == 'adagrad':
            train_step = tf.train.AdagradOptimizer(
                lr)
        elif optimizer == 'sgd':
            train_step = tf.train.GradientDescentOptimizer(
                lr)
        elif optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(
                lr)
        elif optimizer == 'ftrl':
            train_step = tf.train.FtrlOptimizer(
                lr)
        elif optimizer == 'gd':
            train_step = tf.train.GradientDescentOptimizer(
                lr)
        elif optimizer == 'padagrad':
            train_step = tf.train.ProximalAdagradOptimizer(
                lr)  # .minimize(self.loss)
        elif optimizer == 'pgd':
            train_step = tf.train.ProximalGradientDescentOptimizer(
                lr)
        elif optimizer == 'rmsprop':
            train_step = tf.train.RMSPropOptimizer(
                lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(
                lr)
        return train_step

    def _build_train_opt(self):
        """
        Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.loss))
        if self.hparams.is_clip_norm:
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                         for gradient in gradients]
        train_op = train_step.apply_gradients(zip(gradients, variables))
        return train_op

    def _active_layer(self, logit, activation, layer_idx=-1):
        """
        transform the input value with an activation. May use dropout.
        :param logit: input value
        :param activation:  a string indicating the type of activation function.
        :param layer_idx:  index of current layer. Used to retrieve corresponding parameters
        :return:
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        """
        Apply an activation function upon the input value.
        Args:
            logit: the input value.
            activation: a string indicating the activation function type.
        Returns:
            A tensor after applying activation function on logit.
        """
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """
        Apply drops upon the input value.
        Args:
            logit: the input value.
            keep_prob: a scalar probability that each element is kept.

        Returns:
            A tensor of the same shape of logit.

        """
        logit = tf.nn.dropout(x=logit, keep_prob=keep_prob)
        return logit

    def train(self, sess, feed_dict):
        """
        Go through the optimization step once with training data in feed_dict.
        Args:
            sess: the model session object.
            feed_dict: feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run([self.update, self.loss, self.data_loss, self.merged],
                        feed_dict=feed_dict)

    def eval(self, sess, feed_dict):
        """
        Evaluate the data in feed_dict with current model.
        Args:
            sess: the model session object.
            feed_dict: feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            A list of evaluated results, including total loss value, data loss value,
            predicted scores, and ground-truth labels.

        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.loss, self.data_loss, self.pred, self.iterator.labels],
                        feed_dict=feed_dict)

    def infer(self, sess, feed_dict):
        """
        Given feature data (in feed_dict), get predicted scores with current model.
        Args:
            sess: the model session object.
            feed_dict: Instances to predict. This is a dictionary that maps graph elements to values.

        Returns:
            Predicted scores for the given instances.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred],
                        feed_dict=feed_dict)

    def load_model(self, model_path=None):
        """
        Load an existing model.
        Args:
            model_path: model path.
        Raises:
            IOError if the restore operation is failed.
        """
        act_path = self.hparams.load_saved_model
        if model_path is not None:
            act_path = model_path

        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))

    def fit(self, train_file, valid_file, test_file=None):
        """
        Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        Args:
            train_file: training data set.
            valid_file: validation set.
            test_file: test set.

        Returns:
            An instance of self.
        """
        if self.hparams.write_tfevents:
            self.writer = tf.summary.FileWriter(self.hparams.SUMMARIES_DIR, self.sess.graph)

        train_sess = self.sess
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            for batch_data_input in self.iterator.load_data_from_file(train_file):
                step_result = self.train(train_sess, batch_data_input)
                (_, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print('step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}' \
                          .format(step, step_loss, step_data_loss))

            train_end = time.time()
            train_time = train_end - train_start

            if self.hparams.save_model:
                if epoch % self.hparams.save_epoch == 0:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + 'epoch_' + str(epoch))

            eval_start = time.time()
            train_res = self.run_eval(train_file)
            eval_res = self.run_eval(valid_file)
            train_info = ', '.join(
                [str(item[0]) + ':' + str(item[1])
                 for item in sorted(train_res.items(), key=lambda x: x[0])])
            eval_info = ', '.join(
                [str(item[0]) + ':' + str(item[1])
                 for item in sorted(eval_res.items(), key=lambda x: x[0])])
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ', '.join(
                    [str(item[0]) + ':' + str(item[1])
                     for item in sorted(test_res.items(), key=lambda x: x[0])])
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print('at epoch {0:d}'.format(
                    epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info + ' test info: ' + test_info)
            else:
                print('at epoch {0:d}'.format(epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info)
            print('at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}'.format(epoch, train_time, eval_time))

        if self.hparams.write_tfevents:
            self.writer.close()

        return self

    def run_eval(self, filename):
        """
        Evaluate the given file and returns some evaluation metrics.
        Args:
            filename: A file name that will be evaluated.

        Returns:
            A dictionary contains evaluation metrics.
        """
        load_sess = self.sess
        preds = []
        labels = []
        for batch_data_input in self.iterator.load_data_from_file(filename):
            _, _, step_pred, step_labels = self.eval(load_sess, batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        return res

    def predict(self, infile_name, outfile_name):
        """
        Make predictions on the given data, and output predicted scores to a file.
        Args:
            infile_name: Input file name.
            outfile_name: Output file name.

        Returns:
            An instance of self.
        """
        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, 'w') as wt:
            for batch_data_input in self.iterator.load_data_from_file(infile_name):
                step_pred = self.infer(load_sess, batch_data_input)
                step_pred = np.reshape(step_pred, -1)
                wt.write('\n'.join(map(str, step_pred)))
        return self




