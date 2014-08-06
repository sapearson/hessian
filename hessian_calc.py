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

cache_path = "/home/spearson/Hessian/stream-team/hessian"

#np.random.seed(42)
#Current issues: it uses the orbit integration function in several steps. I could write that smarter.


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
    usys = (u.kpc, u.Myr, u.Msun) #Our init system

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
    Iso = IsochronePotential(M,b,usys=usys)
    return actions,angles,freqs,M,b,Iso   

#------------------------------------------------Step 2----------------------------------------------------------#
#Take a grid of (J,theta), use best toy (isocrhone) to analytically convert to (x,v)
def grid_of_AA(w0):
    """This function outputs a grid of action/angles around the actual action and angle for inputted orbit"""

#Start with the actions and angles for the orbit above and maka a grid of actions                                                        
    ngrid = 5
    fractional_stepsize = np.linspace(0.8,1.2,ngrid).reshape(1,5) #20% variation                                                         
    actions,angles,freqs,M,b,Iso = find_J_theta(w0)
    Pal5_actions =  actions #nd.array(3,)                                                                                                
    action_grid = fractional_stepsize * Pal5_actions.reshape(3,1) # Computes 3x5 "matrix" with J1,J2,J3 and the 20 % variations          
    action_grid = np.meshgrid(*action_grid)  #We want a 3D grid with all combinations of the J1,J2,J3s, not sure about *                  
    J1,J2,J3 = map(np.ravel,action_grid) #Flattens the 5x5x5 3d grids, (ask adr)                                                         
    action_array = np.vstack((J1,J2,J3)).T #We now stack them and have a 125x3 array with all possible combinations                      
    angle_array = angles.reshape(1,3)
    return action_array,angle_array


def grid_of_xv(w0):
    """The function computes grid of x,y,z,vx,vy,vz around initial inputted orbit.
    The grid is computed in action/angle space around the J,theta for the particular orbit using the 'grid_of_AA() function',
    and then f(w) finds the x,y,z,vx,vy,vz grid, which is outputted."""

    actions,angles,freqs,M,b,Iso = find_J_theta(w0)
#First we want to go from our J,theta grid to x,v so we can run orbits in LM10
# function to minimize f to get w = (x,v)
    action_array,angle_array = grid_of_AA(w0)
    def f(w,action_array,angle_array):
        try:
            J_prime,theta_prime = Iso.action_angle(w[:3],w[3:])
        except ValueError: # unbound                                                                                                                                                                                                                                                
            return np.zeros_like(w) + 1e6
        return np.squeeze(np.hstack((action_array,angle_array)) - np.hstack((J_prime,theta_prime)))

# We have to try different initial conditions for w to compute J_prime and theta_prime
    n = len(action_array)
    n=2
    xv_coordinates = []
    for k in range(n):
        root_w = []
        for i in range(100):
            w0 = np.append(np.random.random(size=3)*10, np.random.random(size=3)/10.)
            s = so.root(f,w0,args=(action_array[k,:],angle_array[0,:]))
            root_w.append(s.x)    #what is x?
        root_w = np.array(root_w)
        xv_coordinates.append(np.median(root_w,axis=0)) #median of all the output from f(w)
#The array below is what we want for initial conditions for the new orbits from grid
    xv_coordinates = np.array(xv_coordinates)   # we want array instead of list
    return xv_coordinates

#-----------------------------------------------Step 3-----------------------------------------------------------#
#Integrate orbits for all (x,v) on grid in LM10
#xv_coordinates are our new initial conditions in usys.
# Compute "true" (J,theta) for all orbits using Sander's
#Basically same step as step 1, but loop over all new orbits
def grid_of_J_theta_orbit(w0):
    """We now get action/angle grid for desired orbit. This will be used to calculate the Hessian for this orbit."""
    actions,angles,freqs,M,b,Iso = find_J_theta(w0)
    params = []
    xv_coordinates = grid_of_xv(w0)
    n=2
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


def freq_check():
    """Check that the derivative of the hamiltonian with respect to J yields the frequencies obtained with Sanders"""



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
#action_array,angle_array = grid_of_AA(w0)
print action_array[:2,:]
