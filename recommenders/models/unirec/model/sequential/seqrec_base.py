# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.

import torch.nn as nn

from unirec.model.base.recommender import BaseRecommender
import unirec.model.modules as modules


class SeqRecBase(BaseRecommender):

    # def collect_data(self, inter_data, to_device=True):
    #     samples = {
    #         'user_id': inter_data[0],
    #         'item_id': inter_data[1],
    #         'label': inter_data[2],
    #         'item_seq': inter_data[3],
    #         'item_seq_len': inter_data[4]
    #     }
    #     if to_device:
    #         for key in samples.keys():
    #             samples[key] = samples[key].to(self.device, non_blocking=True)
    #     return samples

    def add_annotation(self):
        super(SeqRecBase, self).add_annotation()
        self.annotations.append("SeqRecBase")
