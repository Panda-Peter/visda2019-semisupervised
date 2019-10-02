# --------------------------------------------------------
# Domain Adaptation
# Copyright (c) 2018 VARMS
# Licensed under The MIT License [see LICENSE for details]
# Written by VARMS
# --------------------------------------------------------
from losses.cross_entropy import CrossEntropyLoss, SmoothCrossEntropy

__factory = {
    'CrossEntropy': CrossEntropyLoss,
    'SmoothCrossEntropy': SmoothCrossEntropy,
}

def names():
    return sorted(__factory.keys())


def create(name):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()
