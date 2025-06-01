import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import Module
import torch.nn.init as init

"""
def trunc_normal(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)
"""

class EmbeddingDotBias(Module):
    "Base dot model for collaborative filtering."
    def __init__(self, n_factors, n_users, n_items, y_range=None):
        
        super().__init__()
        self.y_range = y_range
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [Embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]

        # Initialize with truncated normal
        for emb in [self.u_weight, self.i_weight, self.u_bias, self.i_bias]:
            init.trunc_normal_(emb.weight, std=0.01)

    def forward(self, x):
        users,items = x[:,0],x[:,1]
        dot = self.u_weight(users)* self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.y_range is None: return res
        return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]

    @classmethod
    def from_classes(cls, n_factors, classes, user=None, item=None, y_range=None):
        "Build a model with `n_factors` by inferring `n_users` and  `n_items` from `classes`"
        if user is None: user = list(classes.keys())[0]
        if item is None: item = list(classes.keys())[1]
        res = cls(n_factors, len(classes[user]), len(classes[item]), y_range=y_range)
        res.classes,res.user,res.item = classes,user,item
        return res

    def _get_idx(self, arr, is_item=True):
        "Fetch item or user (based on `is_item`) for all in `arr`"
        assert hasattr(self, 'classes'), "Build your model with `EmbeddingDotBias.from_classes` to use this functionality."
        classes = self.classes[self.item] if is_item else self.classes[self.user]
        c2i = {v:k for k,v in enumerate(classes)}
        try: return torch.tensor([c2i[o] for o in arr])
        except KeyError as e:
            message = f"You're trying to access {'an item' if is_item else 'a user'} that isn't in the training data. If it was in your original data, it may have been split such that it's only in the validation set now."
            raise modify_exception(e, message, replace=True)

    def bias(self, arr, is_item=True):
        "Bias for item or user (based on `is_item`) for all in `arr`"
        idx = self._get_idx(arr, is_item)
        layer = (self.i_bias if is_item else self.u_bias).eval().cpu()
        #return to_detach(layer(idx).squeeze(),gather=False)
        return layer(idx).squeeze().detach()

    def weight(self, arr, is_item=True):
        "Weight for item or user (based on `is_item`) for all in `arr`"
        idx = self._get_idx(arr, is_item)
        layer = (self.i_weight if is_item else self.u_weight).eval().cpu()
        #return to_detach(layer(idx),gather=False)
        return layer(idx).detach()