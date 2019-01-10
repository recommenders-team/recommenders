# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
TensorFlow utility for Azure ML (AML)
"""

# https://stackoverflow.com/questions/45532365/is-there-any-tutorial-for-tf-train-sessionrunhook
import tensorflow as tf


class AmlTfLogHook(tf.train.SessionRunHook):
    # TODO check if it is 'iter' or  'step'
    def __init__(self, aml_run, model, X_eval, y_eval, metrics=None, every_n_iter=50):
        """tf.estimator.EvalSpec hook class to log evaluation loss via azureml.core.Run

        Args:
            aml_run (azureml.core.Run): AML run object
            model (tf.Estimator):
            X_eval (pd.DataFrame):
            y_eval (pd.Series):
            metrics (list or tuple): A list of metrics name to log.
                All the metrics should be added to the model prior.
            every_n_iter (int):
        """
        self.aml_run = aml_run
        self.model = model
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.metrics = metrics if metrics is not None else ['loss']
        self.every_n_iter = every_n_iter
        self.iter = 0

    def begin(self):
        self.iter = 0

    def before_run(self, run_context):
        # return SessionRunArgs(self.your_tensor)
        self.iter += 1

    def after_run(self, run_context, run_values):
        # run_values.results
        # If wan to stop loop, run_context.request_stop() -- SessionRunContext
        # loss_value = run_values.results
        # losses = run_context.session.graph.get_collection('losses')
        # #   print(run_context.session.run(losses))
        # run.log('training_acc', np.float(acc_train))

        if self.iter % self.every_n_iter == 0:
            print("{}-iter evaluating...".format(self.iter))

            eval_input_fn = tf.estimator.inputs.pandas_input_fn(
                x=self.X_eval,
                y=self.y_eval,
                batch_size=1,
                num_epochs=1,
                shuffle=False
            )
            eval_metrics = self.model.evaluate(input_fn=eval_input_fn)
            print(eval_metrics)

            for m in self.metrics:
                self.aml_run.log(m, eval_metrics[m])
