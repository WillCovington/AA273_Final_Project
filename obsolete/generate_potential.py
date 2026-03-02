# the next step towards propogating our orbit is generate the gravitational potential
# from that we can compute the accelerations on the spacecraft and propogate the orbit forward in time

import numpy as np
import pyshtools as pysh # this library was designed specifically for spherical harmonics

def perturb_potential_map(C, S, N, mu, R, r_eval, nlat=361, nlon=721):
    # description of our inputs
    # C, S: our spherical harmonic coefficients
    # N: truncation degree (if we can pass in the 1200x1200 matrix everytime, we can just directly truncate it here)
    # mu: grav parameter for the moon
    # R: the moon's mean radius
    # r_eval: our orbiting radius (altitude plus R)
    # nlat: the number of latitude points we want to evaluate over (like our vertical resolution, basically)
    # nlon: same deal as nlat, but for longitude (basically our horizontal resolution)

    # first we truncate down our C and S
    Cn = C[:N+1, :N+1].copy() # copying so we don't do something untoward with our original matrix