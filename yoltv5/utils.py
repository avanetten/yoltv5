#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:16:53 2018

@author: avanetten

Copied from: https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/utils.py
"""

import math
import numpy as np
from subprocess import Popen, PIPE, STDOUT
from statsmodels.stats.weightstats import DescrStatsW


###############################################################################
def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))


###############################################################################
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """

    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    # weighted mean of data (equivalent to np.average(array, weights=weights))
    mean = weighted_stats.mean
    # standard deviation with default degrees of freedom correction
    std = weighted_stats.std
    # variance with default degrees of freedom correction
    var = weighted_stats.var

    return (mean, std, var)


###############################################################################
def twinx_function(x, raw=False):
    V = 3./x
    if raw:
        return V
    else:
        return ["%.1f" % z for z in V]
    # return [z for z in V]


###############################################################################
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x,
                        [x < x0],
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


###############################################################################
def _file_len(fname):
    '''Return length of file'''
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except:
        return 0


###############################################################################
def _run_cmd(cmd):
    '''Write to stdout, etc,(incompatible with nohup)'''
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print(line.replace('\n', ''))
    return



# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
###############################################################################
def convert(size, box):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]
    Return yolo coords: normalized (x, y, w, h)'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmid = (box[0] + box[1])/2.0
    ymid = (box[2] + box[3])/2.0
    w0 = box[1] - box[0]
    h0 = box[3] - box[2]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x,y,w,h)
 
 
# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
###############################################################################
def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x, y, w, h = box
    dw = 1./size[0]
    dh = 1./size[1]

    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh

    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]
   
