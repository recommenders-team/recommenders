# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torch.optim as optim



def activation(input, kind):
    """
    Apply an activation function to the input data
    :param input: Input data
    :param kind: Type of activation
    """
    if kind == 'selu':
        return F.selu(input)
    elif kind == 'relu':
        return F.relu(input)
    elif kind == 'relu6':
        return F.relu6(input)
    elif kind == 'sigmoid':
        return F.sigmoid(input)
    elif kind == 'tanh':
        return F.tanh(input)
    elif kind == 'elu':
        return F.elu(input)
    elif kind == 'lrelu':
        return F.leaky_relu(input)
    elif kind == 'swish':
        return input*F.sigmoid(input)
    elif kind == 'none':
        return input
    else:
        raise ValueError('Unknown non-linearity type')

def MSEloss(inputs, targets, size_avarage=False):
    """
    Masked Mean Square Error Loss http://pytorch.org/docs/master/nn.html#torch.nn.MSELoss
    :param input: Input data
    :param targets: Target data
    :param size_avarage: if True, losses are averaged over observations for each minibatch, if False, the losses are
    summed for each minibatch
    """
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(size_average=size_avarage)
    loss = criterion(inputs * mask.float(), targets)
    num_ratings = (Variable(torch.Tensor([1.0])) if size_avarage
                   else num_ratings)
    return loss, num_ratings


def add_gpu(model, gpu_ids_input):
    """ Add GPUs to the model
    :param model: PyTroch model.
    :param gpu_ids_input: GPU input string.
    """
    gpu_ids = [int(g) for g in gpu_ids_input.split(',')]
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    model = model.cuda()
    return model


def init_optimizer(model, optimization_method, lr, wd, momentum=0.9,
                   lr_steps=[24, 36, 48, 66, 72], gamma=0.5):
    """ Initialize the optimizer and (optionally) the scheduler
    :param model: PyTroch model.
    :param optimization_method: Optimization method (adam, adagrad, momentum,
                                rmsprop).
    :param lr: Learning rate.
    :param wd: Weight decay.
    :param momentum: Momentum.
    :param lr_steps: Epochs in which lr decreases.
    :param gamma: Multiplier factor to decreate the lr.
    """
    if optimization_method == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=wd)
    elif optimization_method == "adagrad":
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=lr,
                                  weight_decay=wd)
    elif optimization_method == "momentum":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=wd)
        scheduler = MultiStepLR(optimizer,
                                milestones=lr_steps,
                                gamma=gamma)
    elif optimization_method == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=lr,
                                  momentum=momentum,
                                  weight_decay=wd)
    else:
        raise ValueError(
            'Unknown optimizer kind {}'.format(optimization_method))
    return optimizer, scheduler
