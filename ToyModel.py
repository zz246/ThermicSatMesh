import os
import sys
import pdb
import h5py
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import interpolate
import multiprocessing
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import pickle
import corner
import logging
import itertools
import pymc3 as pm
import theano.tensor as tt
import theano.compile.ops as tco
import subprocess
import healpy as hp
import spiceypy as spice
import cmath
import scipy
from scipy.stats import spearmanr
import glob
import numpy as np
from itertools import accumulate
#--------------------------------------------------
#---- Mie theory / Compute phase function and Qext/Qsca for a range of scatterer sizes

if __name__ == '__main__':
    
    Dthick          = 3.0
    etameltLOG10    = 17.0
    
    Ts              = 100.0
    Rsurface        = 0.3
    Reflection      = 1e-5
    ReflectionDepth = Dthick * 0.5
    
    asize           = 0.01 # m
    scatterer       = 'Void'
    
    IncidJup        = 10.0 
    IncidCmb        = 10.0 
    ThetaMWR        = 0.0
    
    #ConvolveJup is the integral of TB_JupSynch*sintheta*dtheta*dphi
    #SpecularJup is the Specularly reflected Jupiter Synchrotron into MWR if Rsurface=1
    
    #==========================================
    Surface         = 'Rough'
    
    intrinsic   = Intrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=asize,scatterer=scatterer,Surface=Surface)
    extrinsic   = Extrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=asize,scatterer=scatterer,Surface=Surface,Incid=IncidJup)
    
    Model_Rough = intrinsic[:,149] 
    Model_Rough = Model_Rough + ( extrinsic[:,149] + Rsurface/math.pi ) * ( ConvolveJup*math.cos(ThetaJup*math.pi/180.0) + ConvolveCmb*math.cos(ThetaCmb*math.pi/180.0) ) 
    
    #==========================================
    Surface         = 'Specular'
    
    intrinsic   = Intrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=asize,scatterer=scatterer,Surface=Surface)
    extrinsicJup= Extrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=asize,scatterer=scatterer,Surface=Surface,Incid=IncidJup)
    extrinsicCmb= Extrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=asize,scatterer=scatterer,Surface=Surface,Incid=IncidCmb)
    
    Model_Specular = intrinsic[:,149,int(ThetaMWR)] 
    Model_Specular = Model_Specular + extrinsicJup[:,149,int(ThetaMWR)] * ConvolveJup * math.cos(ThetaJup*math.pi/180.0) 
    Model_Specular = Model_Specular + extrinsicCmb[:,149,int(ThetaMWR)] * ConvolveCmb * math.cos(ThetaCmb*math.pi/180.0) 
    Model_Specular = Model_Specular + ( Rsurface + (1-Rsurface)*(1-np.cos(IncidJup*math.pi/180.0))**5 ) * SpecularJup
    Model_Specular = Model_Specular + ( Rsurface + (1-Rsurface)*(1-np.cos(IncidCmb*math.pi/180.0))**5 ) * SpecularCmb
    
    
    
    pdb.set_trace()
