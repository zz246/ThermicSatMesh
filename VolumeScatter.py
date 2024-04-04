import utils
import Thermal

def Layer_VolumeScatter(nz,Dthick,etameltLOG10,Ts):
    Dthick          = Dthick*1e3                
    etamelt         = 10.0**etameltLOG10

    if (Dthick==3.0):
        Depth_List  = np.array([ 3.0,   3.0,   3.0,    1.5,    0.6,    0.16]) * 1e3
    if (Dthick==10.0):
        Depth_List  = np.array([ 10.0,   10.0,   5.5,    2.0,    0.7,    0.17]) * 1e3
    if (Dthick==30.0):
        Depth_List  = np.array([ 30.0,   20.0,   8.0,    2.8,    0.7,    0.17]) * 1e3

    #------------
    constC          = 3.0e8    
    Freq_list       = np.array([0.6,1.25,2.5,5.0,10.0,22.0])*1e9    # GHz
    Wvl_list        = np.array([0.5,0.24,0.12,0.06,0.03,0.014])     # m
    #-------        
    
    Rbottom_list         = np.zeros(6)
    Absorption_list      = np.zeros((6,nz))
    Temperature_list     = np.zeros((6,nz))            
    
    for Freq in range(6):
        #------------            
        Depth = Depth_List[Freq]
        dz  = Depth/(nz*1.0)
        z   = np.arange(nz)*dz + dz/2.0
        #------------
        tempwater   = 270.0
        theta       = 1.0-300.0/tempwater
        e0          = 77.66-103.3*theta
        e1          = 0.0671*e0
        e2          = 3.52+7.52*theta
        v1          = 20.2+146.4*theta+316.0*theta*theta
        v2          = 39.8*v1
        jdex        = complex(0,1)
        Ewater      = e2+(e1-e2)/(1.0-jdex*Freq_list[Freq]/1e9/v2)+(e0-e1)/(1.0-jdex*Freq_list[Freq]/1e9/v1)
        Nwaterbottom= np.sqrt(Ewater)
        #-------
        theta           = 300.0/tempwater-1.0
        alpha           = (0.00504+0.0062*theta)*np.exp(-22.1*theta)
        beta            = 0.0207/tempwater * np.exp(335.0/tempwater) / ( np.exp(335.0/tempwater)-1.0 )**2.0 + 1.1610e-11*(Freq_list[Freq]/1e9)**2.0 + np.exp(-9.963+0.0372*(tempwater-273.16))
        Eice            = 3.1884 + 0.00091*(tempwater-273.0) + complex(0,1)*(alpha/(Freq_list[Freq]/1e9)+beta*(Freq_list[Freq]/1e9))    
        #Dust_e          = complex(7.0, 0.046)
        #Eice            = Eice * ( 1.0 + 3.0*fv*(Dust_e-Eice)/(Dust_e+2.0*Eice-fv*(Dust_e-Eice)) )
        Nicebottom      = np.sqrt(Eice)
        #---- Thermal Model  (Anton)----#
        if True:
            PlanetEu = {"R": 1560800.0, "g" : 1.315, "Dshell" : Dthick, "Tm" : 270.0, "Ts" : Ts}
            EOSEu    = {"etamelt" : etamelt, "rho_const" : 917.0, "alpha_const" : 1.54e-4, "k_const" : 2.6 , "Cp_const" : 2093.0}
            Rarray, Tarray = ConvectionDeschampsSotin(PlanetEu, EOSEu)
            #-------------------------------------------
            tmpx = PlanetEu["R"] - Rarray
            tmpy = Tarray

            sorta = np.argsort(tmpx)
            tmpx  = tmpx[sorta]   
            tmpy  = tmpy[sorta]   

            FuncTemp = interpolate.interp1d(tmpx,tmpy)
            
        Temperature = FuncTemp(z)
        #------------
        theta           = 300.0/Temperature-1.0
        alpha           = (0.00504+0.0062*theta)*np.exp(-22.1*theta)
        beta            = 0.0207/Temperature * np.exp(335.0/Temperature) / ( np.exp(335.0/Temperature)-1.0 )**2.0 + 1.1610e-11*(Freq_list[Freq]/1e9)**2.0 + np.exp(-9.963+0.0372*(Temperature-273.16))
        Eice            = 3.1884 + 0.00091*(Temperature-273.0) + complex(0,1)*(alpha/(Freq_list[Freq]/1e9)+beta*(Freq_list[Freq]/1e9))    
        #Dust_e          = complex(7.0, 0.046)
        #Eice            = Eice * ( 1.0 + 3.0*fv*(Dust_e-Eice)/(Dust_e+2.0*Eice-fv*(Dust_e-Eice)) )
        Nice            = np.sqrt(Eice)        
        #-------------
        RRSbottom   = ( Nwaterbottom - Nicebottom )/( Nwaterbottom + Nicebottom )
        RRSbottom   = np.abs(RRSbottom)
        RRSbottom   = RRSbottom**2

        RRPbottom   = ( Nwaterbottom - Nicebottom )/( Nwaterbottom + Nicebottom )
        RRPbottom   = np.abs(RRPbottom)
        RRPbottom   = RRPbottom**2

        RRbottom    = (RRSbottom+RRPbottom)/2.0
        
        Rbottom_list[Freq] = RRbottom                        
        #-------------
        
        Absorption_list[Freq,:]      = 4.0*math.pi*Nice.imag/Wvl_list[Freq]*dz
        Temperature_list[Freq,:]     = Temperature                
    
    if (Dthick==3.0):
        Rbottom_list[3:6]       = 0.0            
    if (Dthick==10.0):
        Rbottom_list[2:6]       = 0.0                        
    if (Dthick==30.0):
        Rbottom_list[1:6]       = 0.0
                
    return Rbottom_list,Absorption_list,Temperature_list,Depth_List

def Intrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=None,scatterer='Void',Surface='Rough'):
    ReflectionDepth = ReflectionDepth*1e3
    
    if (Surface=='Rough'):
        if (asize in not None):
            [Qext,Qsca] = Qext(asize,scatterer)
        else:
            Qext = np.ones(6)
            Qsca = np.ones(6)
        
        nz              = 50
        ntime           = 150
        ntheta          = 90
        theta           = np.arange(ntheta)*1.0+0.5
        dtheta          = 1.0*math.pi/180.0
        dphi            = 2.0*math.pi
        
        [Rbottom_list,Absorption_list,Temperature_list,Depth_List] = Layer_VolumeScatter(nz,Dthick,etameltLOG10,Ts)
        Emission_list   = 1.0-np.exp(-Absorption_list[:,:,np.newaxis]/np.cos(theta[np.newaxis,np.newaxis,:]*math.pi/180.0))
        
        ToutputAccum    = np.zeros((6,ntime))
        
        for Freq in range(6):
            Depth       = Depth_List[Freq]
            dz          = Depth/(nz*1.0)
            z           = np.arange(nz)*dz + dz/2.0
            Rbottom     = Rbottom_list[Freq]
        
            Tup         = np.zeros((ntime,nz+1,ntheta))
            Tdn         = np.zeros((ntime,nz+1,ntheta))
            #--- initialize --#
            Tup[0,0:nz,:]       = Temperature_list[Freq,:,np.newaxis]*Emission_list[Freq,:,:]
            if (Rbottom>0):
                Tup[0,nz,:]     = 270.0 * 1.0 * (1-Rbottom)
            else:
                Tup[0,nz,:]     = Tup[0,nz-1,:]
    
            Tdn[0,1:nz+1,:]     = Temperature_list[Freq,0:nz,np.newaxis]*Emission_list[Freq,0:nz,:]
            Tdn[0,0,:]          = 0.0
            #----------------------------- Volume Scattering
            dR          = np.zeros(nz)
            
            select      = z<=ReflectionDepth
            if (np.sum(select)>0):
                dR[select]  = dz * Reflection 
            
            select      = (z<=ReflectionDepth) & ((z+dz/2.0)>ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (ReflectionDepth-z[select]+dz/2.0) * Reflection
            
            select      = (z>ReflectionDepth) & ((z-dz/2.0)<ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (dz/2.0-(z[select]-ReflectionDepth)) * Reflection 
        
            dRQ         = dR  * Qext[Freq] / Qext[0]
            dRS         = dR * Qsca[Freq] / Qext[0]
            #-----------------------------
            dA          = np.copy(Absorption_list[Freq,:])
            
            for itime in range(1,ntime):
                for i in range(1,nz):
                    Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                
                
                    Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer nz-1 bottom
                i=nz
                                                        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.cos(theta[:]*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * Rbottom/math.pi )                                                  
                Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer 0 surface
                i=0
            
                Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.cos(theta*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * (Rsurface/math.pi) )                    
                #---------------------------
                Toutput = np.sum(Tup[itime-1,0,:] * np.cos(theta*math.pi/180.0) * np.sin(theta*math.pi/180.0) * dtheta * dphi) * (1-Rsurface) / math.pi                   
                ToutputAccum[Freq,itime] = ToutputAccum[Freq,itime-1] + Toutput
    
    if (Surface=='Specular'):
        if (asize in not None):
            [Qext,Qsca] = Qext(asize,scatterer)
        else:
            Qext = np.ones(6)
            Qsca = np.ones(6)
            
        nz              = 50
        ntime           = 150
        ntheta          = 90
        theta           = np.arange(ntheta)*1.0+0.5
        dtheta          = 1.0*math.pi/180.0
        dphi            = 2.0*math.pi
        
        [Rbottom_list,Absorption_list,Temperature_list,Depth_List] = Layer_VolumeScatter(nz,Dthick,etameltLOG10,Ts)
        Emission_list   = 1.0-np.exp(-Absorption_list[:,:,np.newaxis]/np.cos(theta[np.newaxis,np.newaxis,:]*math.pi/180.0))
                
        NiceSurface = NiceSurface()
        OutputIce   = np.arcsin(np.sin(theta*math.pi/180.0) * NiceSurface[0].real / 1.0) * 180.0/math.pi
        OutputIce   = OutputIce.astype(int)
        select      = OutputIce<0
        OutputIce[select] = -1000
        
        tmp_Rsurface    = Rsurface + (1-Rsurface)*(1-np.cos(theta*math.pi/180.0))**5
        select          = OutputIce<0
        tmp_Rsurface[select] = 1.0
        
        ToutputAccum        = np.zeros((6,ntime,ntheta))
        
        for Freq in range(6):       
            Depth   = Depth_List[Freq]
            dz      = Depth/(nz*1.0)
            z       = np.arange(nz)*dz + dz/2.0
            Rbottom     = Rbottom_list[Freq]
        
            Tup         = np.zeros((ntime,nz+1,ntheta))
            Tdn         = np.zeros((ntime,nz+1,ntheta))
            #--- initialize --#
            Tup[0,0:nz,:]       = Temperature_list[Freq,:,np.newaxis]*Emission_list[Freq,:,:]
            if (Rbottom>0):
                Tup[0,nz,:]     = 270.0 * 1.0 * (1-Rbottom)
            else:
                Tup[0,nz,:]     = Tup[0,nz-1,:]
    
            Tdn[0,1:nz+1,:]     = Temperature_list[Freq,0:nz,np.newaxis]*Emission_list[Freq,0:nz,:]
            Tdn[0,0,:]          = 0.0
            #----------------------------- Volume Scattering
            dR          = np.zeros(nz)
            
            select      = z<=ReflectionDepth
            if (np.sum(select)>0):
                dR[select]  = dz * Reflection
            
            select      = (z<=ReflectionDepth) & ((z+dz/2.0)>ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (ReflectionDepth-z[select]+dz/2.0) * Reflection
            
            select      = (z>ReflectionDepth) & ((z-dz/2.0)<ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (dz/2.0-(z[select]-ReflectionDepth)) * Reflection
        
            dRQ         = dR  * Qext[Freq] / Qext[0]
            dRS         = dR * Qsca[Freq] / Qext[0]
            #-----------------------------
            dA          = np.copy(Absorption_list[Freq,:])

            for itime in range(1,ntime):
                for i in range(1,nz):
                    Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                
                
                    Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer nz-1 bottom
                i=nz
                                                        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.cos(theta[:]*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * Rbottom/math.pi )                                                  
                Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer 0 surface
                i=0
            
                Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
            
                                                    
                Tdn[itime,i,:]  = Tdn[itime,i,:] + Tup[itime-1,i,:] * tmp_Rsurface             
                #---------------------------
            
                select = OutputIce>=0
                func   = interpolate.interp1d( OutputIce[select], Tup[itime-1,0,select]  * ( 1 - tmp_Rsurface[select] ), fill_value='extrapolate', kind='nearest' )
                ToutputAccum[Freq,itime,:] = ToutputAccum[Freq,itime-1,:] + func(theta)
                                                                     
    return ToutputAccum

def Extrinsic_VolumeScatter(Dthick,etameltLOG10,Ts,Rsurface,Reflection,ReflectionDepth,asize=None,scatterer='Void',Surface='Rough',Incid=0.0):
    ReflectionDepth = ReflectionDepth*1e3
    
    if (Surface=='Rough'):
        if (asize in not None):
            [Qext,Qsca] = Qext(asize,scatterer)
        else:
            Qext = np.ones(6)
            Qsca = np.ones(6)
                
        nz              = 50
        ntime           = 150
        ntheta          = 90
        theta           = np.arange(ntheta)*1.0+0.5
        dtheta          = 1.0*math.pi/180.0
        dphi            = 2.0*math.pi
        
        [Rbottom_list,Absorption_list,Temperature_list,Depth_List] = Layer_VolumeScatter(nz,Dthick,etameltLOG10,Ts)
        Emission_list   = 1.0-np.exp(-Absorption_list[:,:,np.newaxis]/np.cos(theta[np.newaxis,np.newaxis,:]*math.pi/180.0))
        
        ToutputAccum    = np.zeros((6,ntime))
        
        for Freq in range(6):
            Depth       = Depth_List[Freq]
            dz          = Depth/(nz*1.0)
            z           = np.arange(nz)*dz + dz/2.0
            Rbottom     = Rbottom_list[Freq]
        
            Tup         = np.zeros((ntime,nz+1,ntheta))
            Tdn         = np.zeros((ntime,nz+1,ntheta))
            #--- initialize --#
            Tdn[0,0,:]  = 1.0 * (1-Rsurface)/math.pi
            #----------------------------- Volume Scattering
            dR          = np.zeros(nz)
            
            select      = z<=ReflectionDepth
            if (np.sum(select)>0):
                dR[select]  = dz * Reflection 
            
            select      = (z<=ReflectionDepth) & ((z+dz/2.0)>ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (ReflectionDepth-z[select]+dz/2.0) * Reflection
            
            select      = (z>ReflectionDepth) & ((z-dz/2.0)<ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (dz/2.0-(z[select]-ReflectionDepth)) * Reflection 
        
            dRQ         = dR  * Qext[Freq] / Qext[0]
            dRS         = dR * Qsca[Freq] / Qext[0]
            #-----------------------------
            dA          = np.copy(Absorption_list[Freq,:])
            
            for itime in range(1,ntime):
                for i in range(1,nz):
                    Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                
                
                    Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer nz-1 bottom
                i=nz
                                                        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.cos(theta[:]*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * Rbottom/math.pi )                                                  
                Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer 0 surface
                i=0
            
                Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.cos(theta*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * (Rsurface/math.pi) )                    
                #---------------------------
                Toutput = np.sum(Tup[itime-1,0,:] * np.cos(theta*math.pi/180.0) * np.sin(theta*math.pi/180.0) * dtheta * dphi) * (1-Rsurface) / math.pi                   
                ToutputAccum[Freq,itime] = ToutputAccum[Freq,itime-1] + Toutput
    
    if (Surface=='Specular'):
        if (asize in not None):
            [Qext,Qsca] = Qext(asize,scatterer)
        else:
            Qext = np.ones(6)
            Qsca = np.ones(6)
            
        nz              = 50
        ntime           = 150
        ntheta          = 90
        theta           = np.arange(ntheta)*1.0+0.5
        dtheta          = 1.0*math.pi/180.0
        dphi            = 2.0*math.pi
        
        [Rbottom_list,Absorption_list,Temperature_list,Depth_List] = Layer_VolumeScatter(nz,Dthick,etameltLOG10,Ts)
        Emission_list   = 1.0-np.exp(-Absorption_list[:,:,np.newaxis]/np.cos(theta[np.newaxis,np.newaxis,:]*math.pi/180.0))
                
        NiceSurface = NiceSurface()
        OutputIce   = np.arcsin(np.sin(theta*math.pi/180.0) * NiceSurface[0].real / 1.0) * 180.0/math.pi
        OutputIce   = OutputIce.astype(int)
        select      = OutputIce<0
        OutputIce[select] = -1000
        
        tmp_Rsurface    = Rsurface + (1-Rsurface)*(1-np.cos(theta*math.pi/180.0))**5
        select          = OutputIce<0
        tmp_Rsurface[select] = 1.0
        
        IncidIce    = np.arcsin(np.sin(Incid*math.pi/180.0) * 1.0 / NiceSurface[0].real)*180.0/math.pi
        
        ToutputAccum        = np.zeros((6,ntime,ntheta))
        
        for Freq in range(6):       
            Depth   = Depth_List[Freq]
            dz      = Depth/(nz*1.0)
            z       = np.arange(nz)*dz + dz/2.0
            Rbottom     = Rbottom_list[Freq]
        
            Tup         = np.zeros((ntime,nz+1,ntheta))
            Tdn         = np.zeros((ntime,nz+1,ntheta))
            #--- initialize --#
            Tdn[0,0,int(IncidIce)]  = 1.0 * ( 1.0 - ( Rsurface + (1-Rsurface)*(1-np.cos(Incid*math.pi/180.0))**5 ) ) / ( np.sin(IncidIce*math.pi/180.0)*dtheta*dphi )             
            #----------------------------- Volume Scattering
            dR          = np.zeros(nz)
            
            select      = z<=ReflectionDepth
            if (np.sum(select)>0):
                dR[select]  = dz * Reflection
            
            select      = (z<=ReflectionDepth) & ((z+dz/2.0)>ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (ReflectionDepth-z[select]+dz/2.0) * Reflection
            
            select      = (z>ReflectionDepth) & ((z-dz/2.0)<ReflectionDepth)
            if (np.sum(select)>0):
                dR[select]  = (dz/2.0-(z[select]-ReflectionDepth)) * Reflection
        
            dRQ         = dR  * Qext[Freq] / Qext[0]
            dRS         = dR * Qsca[Freq] / Qext[0]
            #-----------------------------
            dA          = np.copy(Absorption_list[Freq,:])

            for itime in range(1,ntime):
                for i in range(1,nz):
                    Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                    Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
                
                
                    Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                    Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer nz-1 bottom
                i=nz
                                                        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.cos(theta[:]*math.pi/180.0)*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * Rbottom/math.pi )                                                  
                Tdn[itime,i,:]  = Tdn[itime,i,:] +          Tdn[itime-1,i-1,:]*np.exp(-dRQ[i-1]/np.cos(theta*math.pi/180.0)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0))   
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tdn[itime-1,i-1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)  * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )           
                Tdn[itime,i,:]  = Tdn[itime,i,:] + np.sum( (Tup[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)    * ((1.0-np.exp(-dRS[i-1]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i-1]/np.cos(theta*math.pi/180.0)) )                    

                #---- layer 0 surface
                i=0
            
                Tup[itime,i,:]  = Tup[itime,i,:] +          Tup[itime-1,i+1,:]*np.exp(-dRQ[i]/np.cos(theta*math.pi/180.0))  * np.exp(-dA[i] / np.cos(theta*math.pi/180.0))   
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tup[itime-1,i+1,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi) * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )        
                Tup[itime,i,:]  = Tup[itime,i,:] + np.sum( (Tdn[itime-1,i,:]*np.sin(theta[:]*math.pi/180.0)*dtheta*dphi)   * ((1.0-np.exp(-dRS[i]/np.cos(theta[:]*math.pi/180.0)))*(1.0/4.0/math.pi)) * np.exp(-dA[i]/np.cos(theta*math.pi/180.0)) )          
            
                                                    
                Tdn[itime,i,:]  = Tdn[itime,i,:] + Tup[itime-1,i,:] * tmp_Rsurface             
                #---------------------------
            
                select = OutputIce>=0
                func   = interpolate.interp1d( OutputIce[select], Tup[itime-1,0,select]  * ( 1 - tmp_Rsurface[select] ), fill_value='extrapolate', kind='nearest' )
                ToutputAccum[Freq,itime,:] = ToutputAccum[Freq,itime-1,:] + func(theta)

    return ToutputAccum
