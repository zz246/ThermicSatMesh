import numpy as np
import interpolate

pi        = np.pi
SECSINYEAR = 86400.0 * 365.0
SECSINMY   = SECSINYEAR * 1e6

#%% SIMPLIFIED CALCULATION OF THE TEMPERATURE PROFILE SUING DESCAMPS & SOTIN (2001)
def ConvectionDeschampsSotin(Planet, EOS):
    '''
    Convection temperature profile based on Deschamps and Sotin 2001
    '''
    Ea = 59.4e3
    Rg = 8.31446261815324
    
    #Get constants
    R       = Planet["R"]
    g       = Planet["g"]
    Dshell  = Planet["Dshell"]
    
    rho     = EOS["rho_const"]
    etamelt = EOS["etamelt"]
    alpha   = EOS["alpha_const"]
    k       = EOS["k_const"]
    Cp      = EOS["Cp_const"]
    
    Tm      = Planet["Tm"]
    Ts      = Planet["Ts"]
    dT      = Tm - Ts
    
    #Temperature dependent viscosity
    eta     = lambda T: etamelt * np.exp(Ea / Rg / Tm * (Tm/T - 1))
    
    #Equation 18
    c1      = 1.43
    c2      = -0.03
    B       = Ea / 2 / Rg / c1
    C       = c2*dT
    
    #Temp of convective region
    Tc      = B * (np.sqrt(1 + 2/B * (Tm - C)) - 1)
    
    #Viscosity of convective region
    etac    = eta(Tc)
    
    #Rayleigh number
    Ra      = alpha * Cp * rho**2 * g * dT * Dshell**3 / etac / k
    
    #Rayleigh number of lower thermal boundary layer
    Radelta = 0.28 * Ra**0.21
    
    #Thickness of lower thermal boundary layer
    delta   = (etac * k / (alpha * Cp * rho**2 * g * (Tm - Tc)) * Radelta)**(1/3)
    
    #Heat flux entering bottom
    phibot  = k * (Tm - Tc) / delta
    
    #Spherical geometry correction by PlanetProfile. Heat flux leaving top
    qTop    = ((R - Dshell) / R)**2 * phibot
    Planet["HeatFLux"] = qTop
    
    #Thickness of conductive lid
    elid    = k * (Tc - Ts) / qTop
    
    #From PlanetProfile based on Solomatov 1995
    RaCrit  = 20.9 * (1.0 * Ea * (Tm - Ts) / Rg / Tc**2)**4
    
    if Ra < RaCrit:
        #print("Convection not happening")
        delta  = 0
        elid   = Dshell
        Tarray = np.array([Tm, Ts])
        rarray = np.array([Dshell, 0])
        return R - rarray,Tarray

    Tarray = np.array([Tm, Tc, Tc, Ts])
    rarray = np.array([Dshell, Dshell - delta, elid, 0])
    return R - rarray,Tarray

#%% COMPUTE VISCOUS TIMESCALE
def Pressure(rho, R, g0, d):
    G = 6.67430e-11
    P = (d * rho * (3 * g0 * R + 2 * d * G * pi * rho * (d - 3 * R))) / (3 * (R - d))
    return P

def Viscosity(etaref, Tref, T):
    Ea   = 59400.0
    Tref = 273.0
    R    = 8.3145
    eta  = etaref * np.exp (Ea / (R * Tref) * (Tref/T - 1))
    return eta

def ViscousTimescale(rho, R, g0, etaref, Tref, r,Rarray, Tarray):
    temperature_profile_interp = interpolate.interp1d(Rarray, Tarray)
    
    T   = temperature_profile_interp(r)
    eta = Viscosity(etaref, Tref, T)
    P   = Pressure(rho, R, g0, R - r)    
    tau = eta / P
    return tau

def auto_sphere(image_file):
    # create a figure window (and scene)
    fig = mlab.figure(size=(1200, 1200))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 1
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,
                                       phi_resolution=Nrad)

    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)
