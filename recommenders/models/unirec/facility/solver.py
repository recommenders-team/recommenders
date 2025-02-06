# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import time
from inspect import signature

from recommenders.models.unirec.utils.general import dict2str
from recommenders.models.unirec.facility.trainer import Trainer


class Solver(Trainer):
    def __init__(self, config, model, accelerator):
        super(Solver, self).__init__(config, model, accelerator)

    def fit(
        self,
        train_data,
        valid_data=None,
        save_model=False,
        load_pretrained_model=False,
        model_file=None,
        verbose=2,
    ):
        logger = self.logger
        self.logger.info("Start solving.")
        if "verbose" in signature(self.model.solve).parameters:
            self.model.solve(train_data, verbose)
        else:
            self.model.solve(train_data)
        logger.info("\nSolving is completed.")

        valid_result = None
        if valid_data:
            valid_data = self.accelerator.prepare(valid_data)
            logger.debug("Valid after training...")
            valid_start_time = time.time()
            valid_result = self.evaluate(valid_data, load_best_model=False, verbose=verbose)
            valid_score = self._get_key_metric(valid_result, self.config["key_metric"])
            valid_end_time = time.time()
            valid_score_output = ("evaluating" + " [time" + ": %.2fs, " + "%s: %f]") % (
                valid_end_time - valid_start_time,
                self.key_metric,
                valid_score,
            )
            valid_result_output = "complete scores on valid set: \n" + dict2str(valid_result)
            logger.info(valid_score_output)
            logger.info(valid_result_output)
        if save_model:
            # Set the epoch and step as -1 to specilize the models that are not optimized by SGD.
            self.save_model(self.saved_model_file, None, None, -1, -1, valid_result, self.config)
