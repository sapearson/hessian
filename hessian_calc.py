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
    actions,angles,freqs,M,b,Iso = find_J_theta(w0)
    ngrid = 5
    fractional_stepsize = np.linspace(0.9,1.1,ngrid).reshape(1,ngrid) #20% variation                                                         
    Pal5_actions =  actions #nd.array(3,)                                                                                                
    action_grid = fractional_stepsize * Pal5_actions.reshape(3,1) # Computes 3x5 "matrix" with J1,J2,J3 and the 20 % variations       # I don't need grid, just two points in each direction around each J.   
    action_array = action_grid.T # Now I have a (5,3) matrix with five different sets of 3 action. The central one is the initial actions.
    #action_grid = np.meshgrid(*action_grid)  #We want a 3D grid with all combinations of the J1,J2,J3s, not sure about *                  
    #J1,J2,J3 = map(np.ravel,action_grid) #Flattens the 5x5x5 3d grids, (ask adr)                                                         
    #action_array = np.vstack((J1,J2,J3)).T #We now stack them and have a 125x3 array with all possible combinations                      
    angle_array = angles.reshape(1,3)
    angle_array = np.squeeze(angle_array)
    return action_array,angle_array


def grid_of_xv(w0):
    """Loop over function above so we ge grid of x,v not just x,v
    This function outputs initial conditions for new "grid-orbits".
    It outputs an nd.array with shape(n,6) where 6 includes the
    6-phace space coordinates"""
    act,ang = grid_of_AA(w0)   #125x3 and 1x3
#    print act, ang
    a,an,fr,M,b,Iso = find_J_theta(w0)
    n = len(act)
    potential = Iso
    xv_coordinates_grid = []
    for k in range(n):
        X,V = potential.phase_space(act[k,:],ang[:])
        xv_coordinates_grid.append(np.append(X,V)) 

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
       
       # act,ang,freqs = find_actions(t, phase_space, N_max=6, usys=usys)
      #  params.append(np.append(act,ang,freqs)) #This gives us act,ang,freqs
     #   print params.shape
        actions,angles = Iso.action_angle(phase_space[:,:3], phase_space[:,3:]) # Use LM10 here, this is just a check for iso coordinate trans
        params.append(np.append(actions[0],angles[0])) 
               
    params = np.array(params)
    return params



#I should plot this new grid of j,theta



#----------------------------------------------Step 5--------------------------------------------------------#
# Interpolate this new grid of (J,theta) to have uniform grid
def interpolate_J_theta_grid():
    """Since we want a uniform grid of J,theta,omega for our Hessian differentiation, we need to 
    interpolate the grid obtained in function grid_of_J_theta_orbit. Ideally we want it to still just 
    output a (5x3) matrix for each variable, where the central row is for our actual orbit."""
    return act,ang,freq

#----------------------------------------------Step 6----------------------------------------------------#
# Compute second derivative of Hamiltonian with respect to uniform grid of J's
#Instead of this just do dOmega/dJ=D. 
#To check our method try seeing if dH/dJ = omega (compare to omega obtained from Sanders' method)
# We get three frequencies out for each point on new grid. 

def hessian_freq():
    """Calculaste Hessian based on derivative of frequencies with respect to J for given orbit"""
    # We will have interpolated frequencies uniform in action/angle space for arbitrary potential.
    # our desired value is the central point of J,theta,omega in this grid?
    # keep two of J's constant while differentiating with respect to 1.
    act, ang, freq = interpolate_J_theta_grid()
    
    # the actions should still be a grid, let's say they are (5,3), 5 different sets of 3 actions
    # the frequencies are also a grid of the same size as actions
    # I will have 9 matrix elements for the hessian. 
    # freq[2,:] and act[2,:] is all the "center-point/true" of our orbit.

    #Forward scheme, only using some of points on grid
  #  freq_dif_11 = (freq[3,0]-freq[2,0])/(act[3,0]-act[2,0]) - (freq[4,0]-2*freq[3,0]+freq[2,0])/(2*(act[3,0]-act[2,0]))
   # freq_dif_12 = (freq[3,0]-freq[2,0])/(act[3,1]-act[2,1]) - (freq[4,0]-2*freq[3,0]+freq[2,0])/(2*(act[3,1]-act[2,1]))
  #  freq_dif_13 = (freq[3,0]-freq[2,0])/(act[3,2]-act[2,2]) - (freq[4,0]-2*freq[3,0]+freq[2,0])/(2*(act[3,2]-act[2,2]))
   # freq_dif_21 = (freq[3,1]-freq[2,1])/(act[3,0]-act[2,0]) - (freq[4,1]-2*freq[3,1]+freq[2,1])/(2*(act[3,0]-act[2,0]))
 #   freq_dif_22 = (freq[3,1]-freq[2,1])/(act[3,1]-act[2,1]) - (freq[4,1]-2*freq[3,1]+freq[2,1])/(2*(act[3,1]-act[2,1]))
  #  freq_dif_23 = (freq[3,1]-freq[2,1])/(act[3,2]-act[2,2]) - (freq[4,1]-2*freq[3,1]+freq[2,1])/(2*(act[3,2]-act[2,2]))
   # freq_dif_31 = (freq[3,2]-freq[2,2])/(act[3,0]-act[2,0]) - (freq[4,2]-2*freq[3,2]+freq[2,2])/(2*(act[3,0]-act[2,0]))
 #   freq_dif_32 = (freq[3,2]-freq[2,2])/(act[3,1]-act[2,1]) - (freq[4,2]-2*freq[3,2]+freq[2,2])/(2*(act[3,1]-act[2,1]))
  #  freq_dif_33 = (freq[3,2]-freq[2,2])/(act[3,2]-act[2,2]) - (freq[4,2]-2*freq[3,2]+freq[2,2])/(2*(act[3,2]-act[2,2]))

    
    #Central scheme, using all points
    freq_dif_11 = (freq[3,0]-freq[1,0])/(act[3,0]-act[1,0]) - (freq[4,0]-2*freq[2,0]+freq[0,0])/(2*(act[3,0]-act[1,0]))
    freq_dif_12 = (freq[3,0]-freq[1,0])/(act[3,1]-act[1,1]) - (freq[4,0]-2*freq[2,0]+freq[0,0])/(2*(act[3,1]-act[1,1]))
    freq_dif_13 = (freq[3,0]-freq[1,0])/(act[3,2]-act[1,2]) - (freq[4,0]-2*freq[2,0]+freq[0,0])/(2*(act[3,2]-act[1,2]))
    freq_dif_21 = (freq[3,1]-freq[1,1])/(act[3,0]-act[1,0]) - (freq[4,1]-2*freq[2,1]+freq[0,1])/(2*(act[3,0]-act[1,0]))
    freq_dif_22 = (freq[3,1]-freq[1,1])/(act[3,1]-act[1,1]) - (freq[4,1]-2*freq[2,1]+freq[0,1])/(2*(act[3,1]-act[1,1]))
    freq_dif_23 = (freq[3,1]-freq[1,1])/(act[3,2]-act[1,2]) - (freq[4,1]-2*freq[2,1]+freq[0,1])/(2*(act[3,2]-act[1,2]))
    freq_dif_31 = (freq[3,2]-freq[1,2])/(act[3,0]-act[1,0]) - (freq[4,2]-2*freq[2,2]+freq[0,2])/(2*(act[3,0]-act[1,0]))
    freq_dif_32 = (freq[3,2]-freq[1,2])/(act[3,1]-act[1,1]) - (freq[4,2]-2*freq[2,2]+freq[0,2])/(2*(act[3,1]-act[1,1]))
    freq_dif_33 = (freq[3,2]-freq[1,2])/(act[3,2]-act[1,2]) - (freq[4,2]-2*freq[2,2]+freq[0,2])/(2*(act[3,2]-act[1,2]))


    D = np.matrix([[freq_dif_11, freq_dif_12, freq_dif_13], [freq_dif_21, freq_dif_22, freq_dif_23], [freq_dif_31,freq_dif_32,freq_dif_33]]) #Hessian matrix
    return D

def freq_check():
    """Check that the derivative of the hamiltonian with respect to J yields the frequencies obtained with Sanders"""
    # Hamiltonian of LM10
    


     #isocrhone in cartesian coordinates


#----------------------------------------------Step 7---------------------------------------------#
#Find eigenvalues of Hessian
def eigenvalues_Hessian():
    """Calculate the three eigenvalues of Hessian for a given orbit"""
    D = hessian_freq()
    eigen_values = np.linalg.eigvals(D)
    return #lamda1,lambda2,lambda3


#To run code:
#inpput orbit initial conditions in kpc and kpc/myr
#w0 = []
#eig = eigenvalues_Hessian(w0)
#print eig 


#-----------To currently run code-----------#(will be more elegant when entire code is written)
w0=[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917]

params = grid_of_J_theta_orbit(w0)
#I now want to check that the outputted grid of J/theta matches the inputted grid if we use the isochrone            
actions = params[:,:3]                                                                                               
angles = params[:,3:6]
#freq = params[:,6:9]
#freq = params[:,6:9]
action_array,angle_array = grid_of_AA(w0)                     
print '---------New actions from x,v -> J, theta----------'
print 'actions'
print actions
print 'angles'
print angles                                                                                                   
print '---------Inputted actions from grid----------'                                                              
print action_array[:,:], angle_array[:]
