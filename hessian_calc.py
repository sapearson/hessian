# coding: utf-8

"""This code takes initial conditions (x,y,z,vx,vy,vz) in kpc and kpc/Myr for an orbit in an arbitrary potential 
and returns the Eigenvalues of the Hessian for this orbit. 
The main function is eigenvalues_Heassian() which outputs the three eigenvalues for the orbit inputted. 
The action/angles/frequencies are also calculated for this orbit. """

__author__ = "spearson <spearson@astro.columbia.edu>"

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import sys
import scipy.optimize as so
import streamteam.integrate as si
from streamteam.potential.lm10 import LM10Potential
import astropy.units as u
from streamteam.dynamics.actionangle import find_actions, fit_isochrone
from streamteam.potential import IsochronePotential
import logging
from astropy import log as logger
logger.setLevel(logging.DEBUG)
from astropy.constants import G

usys = (u.kpc, u.Myr, u.Msun)
G = G.decompose(usys).value

cache_path = "/home/spearson/Hessian/stream-team/hessian"

#np.random.seed(42)#Current issues: it uses the orbit integration function in several steps. I could write that smarter.


#------------------------------------------------Step 1----------------------------------------------------------#
# Find (J,theta) for Pal 5 (stream fannining) in LM10 using Sanders' code
def find_J_theta(w0):
    """Input initial conditions in x,y,z,vx,vy,vz for a givin orbit, integrate this orbit forward in time,
    and obtain action, angles and frequencies for this orbit. Also retuns best paramters for isochrone"""

    potential = LM10Potential() #imported from stream-team

# the integrator requires a function that computes the acceleration
    acceleration = lambda t, *args: potential.acceleration(*args)
    integrator = si.LeapfrogIntegrator(acceleration)

#Convert from initial conditions for Pal5's orbit to new units
#(x,y,z,vx,vy,vz) in kpc and kpc/Myr
#v_new = ([-56.969978, -100.396842, -3.323685]*u.km/u.s).to(u.kpc/u.Myr).value
#print v_new

#    w0 = w0 #[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917] #[x]=kpc and [v]=kpc/MYR
    t,w = integrator.run(w0, dt=1., nsteps=6000)

# w is now the orbit with shape (time, number of orbits, number of dimensions)
# so in this case it is (6000, 1, 6) -- e.g., to plot the orbit in the
#X-y plane:
#Check that we get the same orbit as with our own leapfrog integrator for Pal5
#plt.plot(w[:,0,0], w[:,0,1])
#plt.show()

#np.squeeze removes any 1-D dimension of ndarray.
    phase_space = np.squeeze(w)

#Save 3 action, 3 angles, 3 frequencies for this orbit:
    actions,angles,freqs = find_actions(t, phase_space, N_max=6, usys=usys)
#Our units are [actions] = kpc*kpc/Myr (where Sanders' are kpc*km/s)
# Below we store the best fit parameters from Isochrone potential
    M, b = fit_isochrone(w,usys=usys)
    #print actions, angles
    Iso = IsochronePotential(M,b,usys=usys)
    return actions,angles,freqs,M,b,Iso   

#------------------------------------------------Step 2----------------------------------------------------------#
#Take a grid of (J,theta), use best toy (isocrhone) to analytically convert to (x,v)
def grid_of_AA(w0):
    """This function outputs a grid of action/angles around the actual action and angle for inputted orbit"""

#Start with the actions and angles for the orbit above and maka a grid of actions                                                        
    ngrid = 5
    fractional_stepsize = np.linspace(0.9,1.1,ngrid).reshape(1,5) #20% variation                                                         
    actions,angles,freqs,M,b,Iso = find_J_theta(w0)
    Pal5_actions =  actions #nd.array(3,)                                                                                                
    action_grid = fractional_stepsize * Pal5_actions.reshape(3,1) # Computes 3x5 "matrix" with J1,J2,J3 and the 20 % variations          
    action_grid = np.meshgrid(*action_grid)  #We want a 3D grid with all combinations of the J1,J2,J3s, not sure about *                  
    J1,J2,J3 = map(np.ravel,action_grid) #Flattens the 5x5x5 3d grids, (ask adr)                                                         
    action_array = np.vstack((J1,J2,J3)).T #We now stack them and have a 125x3 array with all possible combinations                      
    angle_array = angles.reshape(1,3)
    angle_array = np.squeeze(angle_array)
    return action_array,angle_array


def spol_to_cart(r,p,t,p_r,p_p,p_t):
    """Performs coordinate transformation from spherical                                  
    polar coordinates with (r,phi,theta,p_r,p_phi,p_theta)                                
    having usual meanings to cartesian coordinates."""
    x1 = r*np.sin(t)*np.cos(p)
    x2 = r*np.sin(t)*np.sin(p)
    x3 = r*np.cos(t)
    v1 = np.cos(p)*np.sin(t)*p_r - r*np.sin(p)*np.sin(t)*p_p + r*np.cos(p)*np.cos(t)*p_t#np.cos(t)*np.cos(p)*p_r + r*np.cos(t)*np.sin(p)*p_p + r*np.sin(t)*np.cos(p)*p_t
    v2 = np.sin(p)*np.sin(t)*p_r + r*np.cos(p)*np.sin(t)*p_p + r*np.sin(p)*np.cos(t)*p_t#np.cos(t)*np.sin(p)*p_r - r*np.cos(t)*np.cos(p)*p_p + r*np.sin(t)*np.sin(p)*p_t
    v3 = np.cos(t)*p_r-r*np.sin(t)*p_t#np.sin(t)*p_r - r*np.cos(t)*p_t

    #XV = x1,x2,x3,v1,v2,v3
    X = x1,x2,x3
    V = v1,v2,v3
    return X,V

def angact_to_xv_iso(act,ang,M,b):                                                        
    """Calculate x and v for a given set of agtion angles using the analytically          
    solvable isochrone potential with best fit parameters b and M.                        
    Function takes in array of action and angles (1x3,1x3) and the best fit parameters for b         
    and M in the isocrhone potential and returns X,V = [( x1,x2,x3), (v1,v2,v3)] for those action       
    angles.                                                                               
    We use Appendix 2 in McGill&Binney 1990
    J1,2,3 = Jr,phi,theta = Jr,Lz,L-Lz"""                                            

    k = G*M                                                                               
    Lz = act[1]                    #This is J2                                                     
    L = act[2]+Lz                 #J3+J2 (A8)                                              
    l_s = np.sqrt(1-(Lz**2/L**2))        #A9   
    H = (-2.*k**2)/(2.*act[0]+L+np.sqrt(4.*b*k+L**2))**2     #A7                                 
    a = -k/(2.*H)-b                       #A10                                             
    e = np.sqrt(1+L**2/(2.*H*a**2))       #A11                                             
    omega = np.sqrt(k)/(2.*(a+b)**(3/2)) * (1. + L/(np.sqrt(4*b*k+L**2)))                    
    
    def psi_func(x,a,e,b,ang):
        """This function is defined to solve for psi in A12
        It takes in an array of initial guesses for psi (x),a,e,b,ang
        and returns function A12 for which we can find the roots."""
        return x - a*e/(a+b)*np.sin(x) - ang[0]  #A12: Function to get psi                          
    x = np.pi/2.
    sol = so.root(psi_func,x, (a,e,b,ang))
    psi = sol.x
    
    r = a*np.sqrt((1-e*np.cos(psi))*(1-e*np.cos(psi)+2*b/a)) #A13                            
    Gamma = np.sqrt((a+b)/k)*((a+b)*psi-a*e*np.sin(psi))  # A16                           
    Lambda = np.arctan(np.sqrt((1+e)/(1-e))*np.tan(0.5*psi)) + L/(np.sqrt(L**2+4*b*k))*np.arctan(np.sqrt((a*(1+e)+2*b)/(a*(1-e)+2*b))*np.tan(0.5*psi))   #A17
    chi = ang[2] - omega*Gamma + Lambda     #A13                                              
    theta = np.arcsin(l_s*np.sin(chi))# A14                                              
    u_f = np.arcsin(1/np.tan(theta)*np.sqrt(1/l_s**2-1)) # we need this to find phi (Sander's code)
    phi = ang[1]+u_f-np.sign(Lz)*ang[2]  #ang2, ang3
    p_r = np.sqrt(k/(a+b))*(a*e*np.sin(psi))/r    #A15                                    
    p_t = L*l_s*np.cos(chi)/np.cos(theta)      #A15                                      
    p_p = Lz                                                                            
    
    # We now need to convert from spherical polar coord to cart. coord.                   
    xv_coordinates = spol_to_cart(r,phi,theta,p_r,p_p,p_t)                       
    return xv_coordinates                                           

def grid_of_xv(w0):
    """Loop over function above so we ge grid of x,v not just x,v
    This function outputs initial conditions for new "grid-orbits".
    It outputs an nd.array with shape(n,6) where 6 includes the
    6-phace space coordinates"""
    act,ang = grid_of_AA(w0)   #125x3 and 1x3
    a,an,fr,M,b,Iso = find_J_theta(w0)
    n = len(act)
    xv_coordinates_grid = []
    for k in range(n):
        X,V = angact_to_xv_iso(act[k,:],ang[:],M,b)
        xv_coordinates_grid.append(np.append(X[:],V[:])) 

    xv_coordinates_grid = np.array(xv_coordinates_grid)
    return xv_coordinates_grid

#-----------------------------------------------Step 3-----------------------------------------------------------#
#Integrate orbits for all (x,v) on grid in LM10
#xv_coordinates are our new initial conditions in usys.
# Compute "true" (J,theta) for all orbits using Sander's
#Basically same step as step 1, but loop over all new orbits
def grid_of_J_theta_orbit(w0):
    """We now get action/angle grid for desired orbit (in the arbitrary potential).
    This will be used to calculate the Hessian for this orbit."""
    actions,angles,freqs,M,b,Iso = find_J_theta(w0) #actually just do this to get M,b
    params = []
    xv_coordinates = grid_of_xv(w0) #get grid of x,v from grid of action agnles shape (n,6)
    
    n = len(xv_coordinates) # this should be same as len(act) from action grid
    acceleration = lambda t, *args: Iso.acceleration(*args)
    integrator = si.LeapfrogIntegrator(acceleration)
    for k in range(n):
    #    xv_coordinates, Iso = grid_of_xv(w0)
        w0 = xv_coordinates[k,:] #These are our initial conditions for all the new orbits
        t,w = integrator.run(w0, dt=1., nsteps=100000) #we integrate each of these orbits to get J,theta for each orbit
        usys = (u.kpc, u.Myr, u.Msun) #Our init system                                                                            
        phase_space = np.squeeze(w)
    #params.append(find_actions(t, phase_space, N_max=6, usys=usys))
        actions,angles = Iso.action_angle(phase_space[:,:3], phase_space[:,3:])
        params.append(np.append(actions[0],angles[0]))
    params = np.array(params)
    return params

#----------------------------------------------Step 5--------------------------------------------------------#
# Interpolate this new grid of (J,theta) to have uniform grid
def intorpolate_J_theta_grid():
    return

#----------------------------------------------Step 6----------------------------------------------------#
# Compute second derivative of Hamiltonian with respect to uniform grid of J's
#Instead of this just do dOmega/dJ=D. 
#To check our method try seeing if dH/dJ = omega (compare to omega obtained from Sanders' method)
# We get three frequencies out for each point on new grid. 

def hessian_freq():
    """Calculaste Hessian based on derivative of frequencies with respect to J for given orbit"""
    # We will have interpolated frequencies uniform in action/angle space for arbitrary potential.
    # our desired value is the central point of J,theta,omega in this grid?

def freq_check():
    """Check that the derivative of the hamiltonian with respect to J yields the frequencies obtained with Sanders"""
    #isocrhone in cartesian coordinates


#----------------------------------------------Step 7---------------------------------------------#
#Find eigenvalues of Hessian
def eigenvalues_Hessian():
    """Calculate the three eigenvalues of Hessian for a given orbit"""
    return #lamda1,lambda2,lambda3



#-----------To currently run code-----------#(will be more elegant when entire code is written)
w0=[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917]
params = grid_of_J_theta_orbit(w0)
#I now want to check that the outputted grid of J/theta matches the inputted grid if we use the isochrone            
actions = params[:,:3]                                                                                               
action_array,angle_array = grid_of_AA(w0)                     
print '---------New actions from x,v -> J, theta----------'
print actions                                                                                                     
print '---------Inputted actions from grid----------'                                                              
print action_array[:2,:]
