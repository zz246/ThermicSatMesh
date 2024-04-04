import numpy as np
import math
import miepython

def Nratio(scatterer):
    constC          = 3.0e8    
    Freq_list       = np.array([0.6,1.25,2.5,5.0,10.0,22.0])*1e9    # GHz
    
    Temperature     = 100.0
    theta           = 300.0/Temperature-1.0
    alpha           = (0.00504+0.0062*theta)*np.exp(-22.1*theta)
    beta            = 0.0207/Temperature * np.exp(335.0/Temperature) / ( np.exp(335.0/Temperature)-1.0 )**2.0 + 1.1610e-11*(Freq_list/1e9)**2.0 + np.exp(-9.963+0.0372*(Temperature-273.16))
    Eice            = 3.1884 + 0.00091*(Temperature-273.0) + complex(0,1)*(alpha/(Freq_list/1e9)+beta*(Freq_list/1e9))    
    Nice            = np.sqrt(Eice)        
    
    if (scatterer=='Void'):
        Evoid = complex(1,0)
    else:
        Evoid = complex(7.7,-0.2)
    
    Nvoid = np.sqrt(Evoid)        
    N_ratio = Nvoid/Nice
    
    return N_ratio

def NiceSurface():
    constC          = 3.0e8    
    Freq_list       = np.array([0.6,1.25,2.5,5.0,10.0,22.0])*1e9    # GHz
    
    Temperature     = 100.0
    theta           = 300.0/Temperature-1.0
    alpha           = (0.00504+0.0062*theta)*np.exp(-22.1*theta)
    beta            = 0.0207/Temperature * np.exp(335.0/Temperature) / ( np.exp(335.0/Temperature)-1.0 )**2.0 + 1.1610e-11*(Freq_list/1e9)**2.0 + np.exp(-9.963+0.0372*(Temperature-273.16))
    Eice            = 3.1884 + 0.00091*(Temperature-273.0) + complex(0,1)*(alpha/(Freq_list/1e9)+beta*(Freq_list/1e9))    
    Nice            = np.sqrt(Eice)        
        
    return Nice

def Qext(asize,scatterer):
    N_ratio = Nratio(scatterer)
    
    Qext                        = np.zeros(6)
    Qsca                        = np.zeros(6)
    Wvl_list                    = np.array([0.5,0.24,0.12,0.06,0.03,0.014])     # m   
    for i in [0,1,2,3,4,5]:     
        x                       = 2.0*math.pi*asize/Wvl_list[i]
        tmp                     = miepython.mie(N_ratio[i],x)
        Qext[i]                 = tmp[0]
        Qsca[i]                 = tmp[1]
    
    return Qext,Qsca
