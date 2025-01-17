# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:25:20 2018

@author: USUARIO
"""

import os
from datetime import datetime


def insertSuffix(filename: str, suffix: str, newExt=None):
    """Insert a suffix between a filename and an extension.

    Can change the extension if instructed to

    Parameters
    ----------
    filename : str
        Full filename.
    suffix : str
        Suffix to add.
    newExt : str, optional
        New file extension if not None. The default is None.

    Returns
    -------
    str
        Changed filename.

    """
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def getUniqueName(name: str):
    """Return a new version of a filename."""

    n = 1
    while os.path.exists(name + '.txt'):
        if n > 1:
            # NEW: preventing first digit of date being replaced too
            pos = name.rfind('_{}'.format(n - 1))
            name = name[:pos] + '_{}'.format(n) + name[pos+len(str(n))+1:]
            # name = name.replace('_{}'.format(n - 1), '_{}'.format(n)) #old 
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name
