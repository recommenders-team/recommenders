# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os.path import join
import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric


__all__ = ["BaseModel"]


class BaseModel:
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.graph = graph if graph is not None else tf.Graph()
        self.iterator = iterator_creator(hparams, self.graph)
        self.train_num_ngs = (
            hparams.train_num_ngs if "train_num_ngs" in hparams else None
        )

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name="layer_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.placeholder(tf.bool, shape=(), name="is_training")
            self.group = tf.placeholder(tf.int32, shape=(), name="group")

            self.initializer = self._get_initializer()

            self.logit = self._build_graph()
            self.pred = self._get_pred(self.logit, self.hparams.method)

            self.loss = self._get_loss()
            self.saver = tf.train.Saver(max_to_keep=self.hparams.epochs)
            self.update = self._build_train_opt()
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.init_op = tf.global_variables_initializer()
            self.merged = self._add_summaries()

        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(self.init_op)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        return self.loss

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        
        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)
        
        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.embed_l1, tf.norm(param, ord=1))
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            obj: Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _get_initializer(self):
        if self.hparams.init_method == "tnormal":
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "uniform":
            return tf.random_uniform_initializer(
                -self.hparams.init_value, self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "normal":
            return tf.random_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "xavier_normal":
            return tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        elif self.hparams.init_method == "xavier_uniform":
            return tf.contrib.layers.xavier_initializer(uniform=True, seed=self.seed)
        elif self.hparams.init_method == "he_normal":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=False, seed=self.seed
            )
        elif self.hparams.init_method == "he_uniform":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=True, seed=self.seed
            )
        else:
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )

    def _compute_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logit, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.squared_difference(
                        tf.reshape(self.pred, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.losses.log_loss(
                    predictions=tf.reshape(self.pred, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logit, (-1, group))
            labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            obj: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)

    def _train_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "adam":
            train_step = tf.train.AdamOptimizer(lr)
        elif optimizer == "ftrl":
            train_step = tf.train.FtrlOptimizer(lr)
        elif optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "padagrad":
            train_step = tf.train.ProximalAdagradOptimizer(lr)
        elif optimizer == "pgd":
            train_step = tf.train.ProximalGradientDescentOptimizer(lr)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "lazyadam":
            train_step = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr)
        return train_step

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.
        
        Args:
            logit (obj): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters
        
        Returns:
            obj: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation)

    def _activate(self, logit, activation):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.
        Args:
            logit (obj): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            obj: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

    def train(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def eval(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict,)

    def infer(self, sess, feed_dict):
        """Given feature data (in feed_dict), get predicted scores with current model.
        Args:
            sess (obj): The model session object.
            feed_dict (dict): Instances to predict. This is a dictionary that maps graph elements to values.

        Returns:
            list: Predicted scores for the given instances.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred], feed_dict=feed_dict)

    def load_model(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: model path.

        Raises:
            IOError: if the restore operation failed.
        """
        act_path = self.hparams.load_saved_model
        if model_path is not None:
            act_path = model_path

        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))

    def fit(self, train_file, valid_file, test_file=None):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_file (str): test set.

        Returns:
            obj: An instance of self.
        """
        if self.hparams.write_tfevents:
            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            for batch_data_input in self.iterator.load_data_from_file(train_file):
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            if self.hparams.save_model:
                if epoch % self.hparams.save_epoch == 0:
                    save_path_str = join(self.hparams.MODEL_DIR, "epoch_" + str(epoch))
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=save_path_str,
                    )

            eval_start = time.time()
            train_res = self.run_eval(train_file)
            eval_res = self.run_eval(valid_file)
            train_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(train_res.items(), key=lambda x: x[0])
                ]
            )
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        if self.hparams.write_tfevents:
            self.writer.close()

        return self

    def run_eval(self, filename):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        load_sess = self.sess
        preds = []
        labels = []
        for batch_data_input in self.iterator.load_data_from_file(filename):
            step_pred, step_labels = self.eval(load_sess, batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        return res

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """
        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(infile_name):
                step_pred = self.infer(load_sess, batch_data_input)
                step_pred = np.reshape(step_pred, -1)
                wt.write("\n".join(map(str, step_pred)))
                # line break after each batch.
                wt.write("\n")
        return self

    def _attention(self, inputs, attention_size):
        """Soft alignment attention implement.
        
        Args:
            inputs (obj): Sequences ready to apply attention.
            attention_size (int): The dimension of attention operation.

        Returns:
            obj: Weighted sum after attention.
        """
        hidden_size = inputs.shape[2].value
        if not attention_size:
            attention_size = hidden_size

        attention_mat = tf.get_variable(
            name="attention_mat",
            shape=[inputs.shape[-1].value, hidden_size],
            initializer=self.initializer,
        )
        att_inputs = tf.tensordot(inputs, attention_mat, [[2],[0]])

        query = tf.get_variable(
            name="query",
            shape=[attention_size],
            dtype=tf.float32,
            initializer=self.initializer,
        )
        att_logits = tf.tensordot(att_inputs, query, axes=1, name="att_logits")
        att_weights = tf.nn.softmax(att_logits, name="att_weights")
        output = inputs * tf.expand_dims(att_weights, -1)
        return output

    def _fcn_net(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = tf.tensordot(
                        hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                    ) + curr_b_nn_layer

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                tf.summary.histogram(
                    "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
                )
                tf.summary.histogram(
                    "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
                )
                nn_output = tf.tensordot(
                    hidden_nn_layers[-1], w_nn_output, axes=1
                ) + b_nn_output
                self.logit = nn_output
                return nn_output
