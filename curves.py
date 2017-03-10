# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 14:27:28 2016

@author: Andy1
"""

import numpy as np

def gen_logistic(t, A, K, B, Q, C, T):
    return A + (K-A)/(C+Q*np.exp(-B*(t-T)))
    
def gen_logistic2(t, K, B, C, T, v):
    return (K)/np.power((C+np.exp(-B*(t-T))),(1/v))
    
#def genlog_dec (t, B, Q, C, T):
#    return 1 + (0-1)/(C+Q*np.exp(-B*(t-T)))
   
#def genlog_dec (t, B, T):
#    return 1 + (0-1)/(1+np.exp(-B*(t-T)))
    
#def genlog_dec (t, B, T, v):
#    return 1 + -1/np.power((1+np.exp(-B*(t-T))),1/v)
    
def genlog_dec (t, B, T):
    return 1 + -1/(1+np.exp(-B*(t-T)))
    
def genlog_inc (t, B, T):
    return 1/(1+np.exp(-B*(t-T)))