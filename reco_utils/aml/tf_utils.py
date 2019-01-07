# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
TensorFlow utility for Azure ML (AML)
"""
# https://stackoverflow.com/questions/45532365/is-there-any-tutorial-for-tf-train-sessionrunhook
# import tensorflow as tf
#
#
# class AmlLogHook(tf.train.SessionRunHook):
#     def __init__(self, run):
#         """tf.estimator.EvalSpec hook class to log evaluation loss via azureml.core.Run
#
#         Args:
#             run (azureml.core.Run): AML run object
#         """
#         self.run = run
#
#     def before_run(self, run_context):
#         print('Before calling session.run().')
#         return SessionRunArgs(self.your_tensor)
#
#     def after_run(self, run_context, run_values):
#         print('Done running one step. The value of my tensor: %s',
#               run_values.results)
#         if you - need - to - stop - loop:
#             run_context.request_stop()
    #
    # def after_run(self, run_context, run_values):
    #     loss_value = run_values.results
    #     losses = run_context.session.graph.get_collection('losses')
    #     #   print(run_context.session.run(losses))
    #     run.log('training_acc', np.float(acc_train))