# coding: utf-8

"""This code takes initial conditions (x,y,z,vx,vy,vz) in kpc and kpc/Myr for an orbit in an arbitrary potential
and returns the Eigenvalues of the Hessian for this orbit.
The main function is eigenvalues_Heassian() which outputs the three eigenvalues for the orbit inputted.
The action/angles/frequencies are also calculated for this orbit. """

__author__ = "spearson <spearson@astro.columbia.edu>"

from __future__ import division

# Standard library
import logging
import sys

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inter
from astropy import log as logger
logger.setLevel(logging.DEBUG)
from astropy.constants import G

# Stream-Team
import streamteam.integrate as si
from streamteam.potential.lm10 import LM10Potential
from streamteam.potential.apw import PW14Potential
from streamteam.dynamics.actionangle import find_actions, fit_isochrone
from streamteam.potential import IsochronePotential
from streamteam.units import galactic

G = G.decompose(galactic).value

cache_path = "/home/spearson/Hessian/stream-team/hessian"

#---------------------- We want to check both LM10() triax & LM10(q1=1,q2=1,q3=1)-------------------------

#---------------------Hessian test - APW-----------------#
ppars = {'a': 6.5,
         'b': 0.26,
         'c': 0.3,
         'm_disk': 65000000000.0,
         'm_spher': 20000000000.0,
         'phi': 1.570796,
         'psi': 1.570796,
         'q1': 1.3,
         'q2': 1.0,
         'q3': 0.8,
         'r_h': 30.0,
         'theta': 1.570796,
         'v_h': 0.5600371815834104}
potential = PW14Potential(**ppars)
w0 = [20.,2.5,4,0.,0.1,0.17]
ngrid = 3
fractional_stepsize = np.array([0.75,1.0,1.25]).reshape(1,ngrid)





#----------------------------------Triaxial run: LM10()--------------------------------------------#
#potential = LM10Potential()

#w0=[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917] #stream-fanning stream LM10
#w0 =[8.161671207, 0.224760075, 16.962073974, -0.20454243,0.0,0.0] # polar orbit, thin stream LM10
#ngrid = 3  #should be 6
#fractional_stepsize = np.array([0.4,1.0,1.2]).reshape(1,ngrid)
#fractional_stepsize = np.array([0.4,0.75,1.0,1.2,1.4,1.6]).reshape(1,ngrid)
#fractional_stepsize = np.array([1.,1.25,1.5]).reshape(1,ngrid)
#fractional_stepsize = np.array([1.,1.1,1.2,1.3,1.4,1.5]).reshape(1,ngrid) #should be used for n=6

#---------------------------------SPHERICAL RUN: lm10(q1=1,q2=1,q3=1)------------------------------#
#potential = LM10Potential(q1=1.,q2=1.,q3=1.)
#w0=[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917]
#fractional_stepsize = np.array([0.5,1.,1.5]).reshape(1,ngrid) #for spherical lm10


#----------------------------------------------------------------------------------------------------------------#
#------------------------------------------------Step 1----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
# Find (J,theta) for Pal 5 (stream fannining) in LM10 using Sanders' code
def find_J_theta(w0):
    """Input initial conditions in x,y,z,vx,vy,vz for a givin orbit, integrate this orbit forward in time,
    and obtain action, angles and frequencies for this orbit. Also retuns best paramters for isochrone"""

#    potential = LM10Potential() #imported from stream-team

#Convert from initial conditions for Pal5's orbit to new units
#(x,y,z,vx,vy,vz) in kpc and kpc/Myr
#v_new = ([-56.969978, -100.396842, -3.323685]*u.km/u.s).to(u.kpc/u.Myr).value
#print v_new

#    w0 = w0 #[8.161671207, 0.224760075, 16.962073974, -0.05826389, -0.10267707,-0.00339917] #[x]=kpc and [v]=kpc/MYR
    t,w = potential.integrate_orbit(w0, dt=0.2, nsteps=50000, Integrator=si.DOPRI853Integrator)

#np.squeeze removes any 1-D dimension of ndarray.
    phase_space = np.squeeze(w)

#Save 3 action, 3 angles, 3 frequencies for this orbit:

    actions,angles,freqs = find_actions(t[::10], phase_space[::10], N_max=6, units=galactic)
    print '-------------------Pal5 actions in LM10-----------------'
    print actions
    print  '-------------------Pal5 angles in LM10-----------------'
    print angles
    print '-------------------Pal5 frequencies in LM10-----------------'
    print freqs
    print '-----------------------------------------------------------'

#Our units are [actions] = kpc*kpc/Myr (where Sanders' are kpc*km/s)
# Below we store the best fit parameters from Isochrone potential
    M, b = fit_isochrone(w, units=galactic)
    #print actions, angles
    iso = IsochronePotential(M,b,units=galactic)
    return actions,angles,freqs,M,b,iso

#------------------------------------------------Step 2----------------------------------------------------------#
#Take a grid of (J,theta), use best toy (isocrhone) to analytically convert to (x,v)
def grid_of_AA(w0):
    """This function outputs a grid of action/angles around the actual action and angle for inputted orbit"""

#Start with the actions and angles for the orbit above and maka a grid of actions
    actions,angles,freqs,M,b,iso = find_J_theta(w0)
    Pal5_actions = actions
    action_grid = fractional_stepsize * Pal5_actions.reshape(3,1)
    action_grid = np.meshgrid(*action_grid)
    J1,J2,J3 = map(np.ravel,action_grid)
    action_array = np.vstack((J1,J2,J3)).T


    #print '-------------actions from grid----------'
   # print action_array
    # I need an angle array that is of the same size as the action_array but with the same angles in each line (len(action_array),3)
    l = len(action_array)
    angle_array = angles.reshape(1,3)
    angle_array = np.repeat(angle_array,l)
    angle_array = angle_array.reshape(3,l)
    angle_array = angle_array.T
    angle_array = np.squeeze(angle_array)

   # print '-------------angles from grid----------'
   # print angle_array


    plt.figure(1)
    plt.plot(action_array[:,0],action_array[:,1],linestyle='none', marker = '.')
   # plt.show()
    plt.xlabel('J1')
    plt.ylabel('J2')
    plt.figure(2)
    plt.plot(action_array[:,0],action_array[:,2], linestyle = 'none', marker = '*')
   # plt.show()
    plt.xlabel('J1')
    plt.ylabel('J3')
    np.save('Inputted_Actions.npy',action_array)


    return action_array,angle_array


def grid_of_xv(w0):
    """Loop over function above so we ge grid of x,v not just x,v
    This function outputs initial conditions for new "grid-orbits".
    It outputs an nd.array with shape(n,6) where 6 includes the
    6-phace space coordinates"""
    act,ang = grid_of_AA(w0)   #125x3 and 1x3
#    print act, ang
    print '------------check w0--------------------'
    print w0
    a,an,fr,M,b,iso = find_J_theta(w0)  #I only do this do get Iso

    n = len(act)
    pot = iso
    xv_coordinates_grid = []
    for k in range(n):
        X,V = pot.phase_space(act[k,:],ang[k,:])             #Something goes wrong here - input exact same actions + angles but get out different xvs
        xv_coordinates_grid.append(np.append(X,V))

    xv_coordinates_grid = np.array(xv_coordinates_grid)
  #  print '------------Check XV grid--------------------'
  #  print xv_coordinates_grid
   # print xv_coordinates_grid.shape
    return xv_coordinates_grid

#-----------------------------------------------Step 3-----------------------------------------------------------#
#Integrate orbits for all (x,v) on grid in LM10
#xv_coordinates are our new initial conditions in usys.
# Compute "true" (J,theta) for all orbits using Sander's
#Basically same step as step 1, but loop over all new orbits
def grid_of_J_theta_orbit(w0):
    """We now get action/angle grid for desired orbit (in the arbitrary potential).
    This will be used to calculate the Hessian for this orbit."""
    actions,angles,freqs,M,b,iso = find_J_theta(w0) #actually just do this to get M,b
    params = []
    xv_coordinates = grid_of_xv(w0) #get grid of x,v from grid of action agnles shape (n,6)
    #print '-------XV coordinates check-------'
    #print xv_coordinates
    n = len(xv_coordinates) # this should be same as len(act) from action grid
    # acceleration = lambda t, *args: potential.acceleration(*args)
    # integrator = si.LeapfrogIntegrator(acceleration)
    allvalues = dict()
    allvalues['actions'] = list()
    allvalues['angles'] = list()
    allvalues['freqs'] = list()
    for k in range(n):
    #    xv_coordinates, Iso = grid_of_xv(w0)
        w0 = xv_coordinates[k,:] #These are our initial conditions for all the new orbits
         # t,w = integrator.run(w0, dt=1., nsteps=100000) #we integrate each of these orbits to get J,theta for each orbit
        t,w = potential.integrate_orbit(w0, dt=0.2, nsteps=50000, Integrator=si.DOPRI853Integrator) # new way of both integrating orbit in specified pot and getting time, pos, vel
        phase_space = np.squeeze(w)
       # sd.plot_orbits(w)  #   - very useful when wanting to plot the integrated orbits
       # plt.show()

        act,ang,freq = find_actions(t, phase_space, N_max=6, units=galactic, toy_potential=iso) # now using the same best fit iso for all new actions

        allvalues['actions'].append(act)
        allvalues['angles'].append(ang)
        allvalues['freqs'].append(freq)


       # actions,angles = Iso.action_angle(phase_space[:,:3], phase_space[:,3:]) # Use LM10 here, this is just a check for iso coordinate trans
      #  params.append(np.append(actions[0],angles[0]))



    act = allvalues.get('actions',0)
    act = np.array(act)
    ang = allvalues.get('angles', 0)
    ang = np.array(ang)
    freq = allvalues.get('freqs', 0)
    freq = np.array(freq)


    plt.figure(3)
    plt.plot(act[:,0],act[:,1],linestyle='none', marker = '.')
    plt.xlabel('J1')
    plt.ylabel('J2')
   # plt.show()
    plt.figure(4)
    plt.plot(act[:,0],act[:,2], linestyle = 'none', marker = '*')
    plt.xlabel('J1')
    plt.ylabel('J3')
    plt.show()
   # params = np.array(params)
    np.save('Outputted_Actions.npy',act)
# return params
    return act,ang,freq


#----------------------------------------------Step 5--------------------------------------------------------#
# Interpolate this new grid of (J,theta) to have uniform grid
def interpolate_J_freq_grid(w0):
    """Since we want a uniform grid of J,theta,omega for our Hessian differentiation, we need to
    interpolate the grid obtained in function grid_of_J_theta_orbit. Ideally we want it to still just
    output a (5x3) matrix for each variable, where the central row is for our actual orbit."""
    # use inter.LinearNDInterpolator()
    Pal5_J_init,angles_init,freqs_init,M,b,iso = find_J_theta(w0)
    act,ang,freq = grid_of_J_theta_orbit(w0)
    J1,J2,J3 = act[:,0], act[:,1], act[:,2]
    f1,f2,f3 = freq[:,0], freq[:,1], freq[:,2]
    pts = np.array([J1,J2,J3]).T


    # Now using interpolater to be able to "call" any frequiency of any grid point in J1,J2,J3
    f_i = inter.LinearNDInterpolator(pts,f1)   #f must have same dimenstions as first dimension of pts#
    f_j = inter.LinearNDInterpolator(pts,f2)
    f_k = inter.LinearNDInterpolator(pts,f3)

    print '--------------Initial frequencies-----------'
    print freqs_init
    print '--------------Frequencies for interpolation----------------'
    print f_i([[Pal5_J_init[0],Pal5_J_init[1], Pal5_J_init[2]]])
    print f_j([[Pal5_J_init[0],Pal5_J_init[1], Pal5_J_init[2]]])
    print f_k([[Pal5_J_init[0],Pal5_J_init[1], Pal5_J_init[2]]])

    return f_i,f_j,f_k

#----------------------------------------------Step 6----------------------------------------------------#
# Compute second derivative of Hamiltonian with respect to uniform grid of J's
#Instead of this just do dOmega/dJ=D.
#To check our method try seeing if dH/dJ = omega (compare to omega obtained from Sanders' method)
# We get three frequencies out for each point on new grid.

def hessian_freq(w0):
    """Calculaste Hessian based on derivative of frequencies with respect to J for given orbit"""
    # We will have interpolated frequencies uniform in action/angle space for arbitrary potential.

    # keep two of J's constant while differentiating with respect to 1.
    J,angles_init,freqs_init,M,b,iso = find_J_theta(w0)
    f1,f2,f3 = interpolate_J_freq_grid(w0)  #returns frequency of some arbitrary choise of actions (within the grid size)
    h = 0.001

    J1 = J[0]
    J2 = J[1]
    J3 = J[2]

    J1_m2 = J1 - 2*h
    J1_m1 = J1 - h
    J1_p1 = J1 + h
    J1_p2 = J1 + 2*h

    J2_m2 = J2 - 2*h
    J2_m1 = J2 - h
    J2_p1 = J2 + h
    J2_p2 = J2 + 2*h

    J3_m2 = J3 - 2*h
    J3_m1 = J3 - h
    J3_p1 = J3 + h
    J3_p2 = J3 + 2*h


    # I need to choose the stepsize... have two different steps back and forth
   # 0,1,2,3,4, 2 is my pal5 init

    #Central scheme, using all points, computing 9 matrix elements for Hessian

    freq_dif_11 = (f1([[J1_p1,J2,J3]]) - f1([[J1_m1,J2,J3]]))/(J1_p1-J1_m1) - (f1([[J1_p2,J2,J3]]) -2*f1([[J1_p1,J2,J3]]) + 2*f1([[J1_m1,J2,J3]]) - f1([[J1_m2,J2,J3]]))/(6*(J1_p1-J1_m1))
    freq_dif_12 = (f1([[J1,J2_p1,J3]]) - f1([[J1,J2_m1,J3]]))/(J2_p1-J2_m1) - (f1([[J1,J2_p2,J3]]) -2*f1([[J1,J2_p1,J3]]) + 2*f1([[J1,J2_m1,J3]]) - f1([[J1,J2_m2,J3]]))/(6*(J2_p1-J2_m1))
    freq_dif_13 = (f1([[J1,J2,J3_p1]]) - f1([[J1,J2,J3_m1]]))/(J3_p1-J3_m1) - (f1([[J1,J2,J3_p2]]) -2*f1([[J1,J2,J3_p1]]) + 2*f1([[J1,J2,J3_m1]]) - f1([[J1,J2,J3_m2]]))/(6*(J3_p1-J3_m1))

    freq_dif_21 = (f2([[J1_p1,J2,J3]]) - f2([[J1_m1,J2,J3]]))/(J1_p1-J1_m1) - (f2([[J1_p2,J2,J3]]) -2*f2([[J1_p1,J2,J3]]) + 2*f2([[J1_m1,J2,J3]]) - f2([[J1_m2,J2,J3]]))/(6*(J1_p1-J1_m1))
    freq_dif_22 = (f2([[J1,J2_p1,J3]]) - f2([[J1,J2_m1,J3]]))/(J2_p1-J2_m1) - (f2([[J1,J2_p2,J3]]) -2*f2([[J1,J2_p1,J3]]) + 2*f2([[J1,J2_m1,J3]]) - f2([[J1,J2_m2,J3]]))/(6*(J2_p1-J2_m1))
    freq_dif_23 = (f2([[J1,J2,J3_p1]]) - f2([[J1,J2,J3_m1]]))/(J3_p1-J3_m1) - (f2([[J1,J2,J3_p2]]) -2*f2([[J1,J2,J3_p1]]) + 2*f2([[J1,J2,J3_m1]]) - f2([[J1,J2,J3_m2]]))/(6*(J3_p1-J3_m1))

    freq_dif_31 = (f3([[J1_p1,J2,J3]]) - f3([[J1_m1,J2,J3]]))/(J1_p1-J1_m1) - (f3([[J1_p2,J2,J3]]) -2*f3([[J1_p1,J2,J3]]) + 2*f3([[J1_m1,J2,J3]]) - f3([[J1_m2,J2,J3]]))/(6*(J1_p1-J1_m1))
    freq_dif_32 = (f3([[J1,J2_p1,J3]]) - f3([[J1,J2_m1,J3]]))/(J2_p1-J2_m1) - (f3([[J1,J2_p2,J3]]) -2*f3([[J1,J2_p1,J3]]) + 2*f3([[J1,J2_m1,J3]]) - f3([[J1,J2_m2,J3]]))/(6*(J2_p1-J2_m1))
    freq_dif_33 = (f3([[J1,J2,J3_p1]]) - f3([[J1,J2,J3_m1]]))/(J3_p1-J3_m1) - (f3([[J1,J2,J3_p2]]) -2*f3([[J1,J2,J3_p1]]) + 2*f3([[J1,J2,J3_m1]]) - f3([[J1,J2,J3_m2]]))/(6*(J3_p1-J3_m1))

    D = np.array([[freq_dif_11, freq_dif_12, freq_dif_13], [freq_dif_21, freq_dif_22, freq_dif_23], [freq_dif_31,freq_dif_32,freq_dif_33]]) #Hessian matrix
    D = np.squeeze(D)
    print '----------------Hessian-----------------'
    print D
    return D

def freq_check():
    """Check that the derivative of the hamiltonian with respect to J yields the frequencies obtained with Sanders"""
    # Hamiltonian of LM10



     #isocrhone in cartesian coordinates


#----------------------------------------------Step 7---------------------------------------------#
#Find eigenvalues of Hessian
def eigenvalues_Hessian(w0):
    """Calculate the three eigenvalues of Hessian for a given orbit"""
    D = hessian_freq(w0)
    eig_values, eig_vectors = np.linalg.eigh(D) # eigh -> for hermitian/symmetric matrices
    print '---------------Eigenvalues---------------'
    print eig_values
    return eig_values



# Spherical orbit (change LM10 to q1=1,q2=1,q3=1)
#v_new = ([-42.640370,-114.249685,-17.028021]*u.km/u.s).to(u.kpc/u.Myr).value
#w0 =[8.161671207, 0.224760075, 16.962073974, -0.04360883, -0.11684454, -0.01741476]


#Triaxial thin stream orbot (LM10)
#v_new = ([-127.427534,-424.623417,69.720859]*u.km/u.s).to(u.kpc/u.Myr).value
#v_new = ([-73.468882, -99.851251,12.677573]*u.km/u.s).to(u.kpc/u.Myr).value
#v_new = ([-200.0,0.0, 0.0]*u.km/u.s).to(u.kpc/u.Myr).value
#print v_new
#w0= [8.161671207, 0.224760075, 16.962073974, -0.13032169,  0.43426753,  0.07130437]

#i = interpolate_J_freq_grid(w0)
#D = hessian_freq(w0)
lamda = eigenvalues_Hessian(w0)

