# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import pandas as pd


def pandas_input_fn(
    df,
    y_col=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    num_threads=1
):
    """Pandas input function for TensorFlow high-level API Estimator.

    tf.estimator.inputs.pandas_input_fn cannot handle array/list column properly.
    If the df does not include any array/list data column, one can simply use TensorFlow's pandas_input_fn.

    For more information, see (https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn)

    Args:
        df (pd.DataFrame): Data containing features
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function
        num_epochs (int): Number of number of epochs to iterate over data. If None will run forever.
        shuffle (bool): If True, shuffles the data queue.
        num_threads (int): Number of threads used for reading and enqueueing.

    Returns:
        (tf.estimator.inputs.numpy_input_fn) function that has signature of ()->(dict of features, targets)
    """

    X_df = df.copy()
    if y_col is not None:
        y = X_df.pop(y_col).values
    else:
        y = None

    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([l for l in values], dtype=np.float32)
        X[col] = values

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=num_threads
    )

    return input_fn


class TrainLogHook(tf.train.SessionRunHook):
    def __init__(
        self,
        model_dir,
        model,
        true_df,
        y_col,
        eval_df,
        every_n_iter=1000,
        eval_fn=None,
        **eval_kwargs
    ):
        """Training log hook for TensorFlow high-level API Estimator.

        Args:
            model_dir (str): The directory to save the summaries to.
            model (tf.estimator.Estimator): Model to evaluate.
            true_df (pd.DataFrame): Ground-truth data.
            y_col (str): Label column name in true_df
            eval_df (pd.DataFrame): Evaluation data. May not include the label column as
                some evaluation functions do not allow.
            every_n_iter (int): Evaluation frequency (steps).
            eval_fn (function): Evaluation function that has signature of
                (true_df, prediction_df, **eval_kwargs)->(float). If None, loss is calculated on true_df.
            **eval_kwargs: Evaluation function's keyword arguments. Note, prediction column name should be 'prediction'
        """
        self.model_dir = model_dir
        self.model = model
        self.true_df = true_df
        self.y_col = y_col
        self.eval_df = eval_df
        self.every_n_iter = every_n_iter
        self.eval_fn = eval_fn
        self.eval_kwargs = eval_kwargs

        self.summary_writer = None
        self.global_step_tensor = None

    def begin(self):
        self.summary_writer = tf.summary.FileWriterCache.get(self.model_dir)
        self.global_step_tensor = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        requests = {'global_step': self.global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        global_step = run_values.results['global_step']

        if global_step % self.every_n_iter == 0:
            loss = self.model.evaluate(
                input_fn=pandas_input_fn(
                    df=self.true_df,
                    y_col=self.y_col
                ),
                steps=None
            )['average_loss']

            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='evaluation/average_loss', simple_value=loss)])
            self.summary_writer.add_summary(loss_summary, global_step)

            if self.eval_fn is not None:
                predictions = list(self.model.predict(input_fn=pandas_input_fn(df=self.eval_df)))
                prediction_df = self.eval_df.copy()
                prediction_df['prediction'] = [p['predictions'][0] for p in predictions]
                result = self.eval_fn(self.true_df, prediction_df, **self.eval_kwargs)

                result_summary = tf.Summary(
                    value=[tf.Summary.Value(tag='evaluation/'+self.eval_fn.__name__, simple_value=result)]
                )
                self.summary_writer.add_summary(result_summary, global_step)

                tf.logging.info(
                    "Evaluation:{0} = {1:.5f}, step = {2}".format(self.eval_fn.__name__, result, global_step)
                )

    def end(self, session):
        self.summary_writer.flush()
