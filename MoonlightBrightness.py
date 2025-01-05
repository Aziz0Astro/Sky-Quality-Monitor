#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 07:33:29 2023

@author: abdulazizabdulaziz
"""

# Import modules
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Read data from file(UTC Time, Local Time, and Brightness Near Zenith in Magnitudes)
data = pd.read_csv("first_quarter6.csv", skiprows=37, sep=";",usecols=[0,1,5]).to_numpy()

# Plot the measured zenith brightness over that night
plt.plot(data[:,2])
plt.rcParams["figure.figsize"] = [20.00, 10.0]
plt.xlabel("Time(minutes)")
plt.ylabel("Britghness Near Zenith(Magnitudes)")
#plt.ylim(max(data[:,2]), min(data[:,2]))
plt.ylim(20,18)
plt.xticks([0,100,200,300,400,500],[data[0,1],data[100,1],data[200,1],data[300,1],data[400,1],data[500,1]])
plt.show()
plt.close()

#utcoffset = -5*u.hour  # Eastern Standard Time
myTime = Time(list(data[:,0])).jd # from UTC time to Julian Date
#myTime = Time.now()                     # current time

# Location: Lat, Lon, Elev (km), Timezone (HH)
mylat, mylon, myele = +41.9671,  -71.1831, 0.026  # Wheaton Coll. Observatory
myloc = {'lon': mylon, 'lat': mylat, 'elevation': myele}

i = 0 # counter
el = np.array([]) # to store the elevation of the moon over the night
V = np.array([]) # to store the magnitude of the moon over the night
while i < len(myTime):
    # Read the elevations and magnitudes every 60 time values. 
    # Horizons Class does not accept more than 60 for security reasons
    myMoon   = Horizons(id=301, location=myloc, epochs=myTime[i:i+60])   # 301 is Moon
    eph      = myMoon.ephemerides(refraction=True)
    myel, myV = eph['EL'], eph['V']
    # Append the elevations and magnitude to one array
    for e_el in myel:
        el = np.append(el,e_el)

    for e_V in myV:
        V = np.append(V,e_V)
        
    i += 60
        

print(el,V)


m = V # Magnitudes of the moon

rho_deg = 90 - el # The Moon/sky separation

Z = 0 # The zenith distance of the sky position 

Zm = rho_deg # Zenith distance of the moon

B0_m = np.mean(data[300:470,2]) # Night-sky brightness without the moon in Magnitudes

B0_nL = 34.08*np.exp(20.7233-0.92104*B0_m) # Night-sky brightness without the moon in nanoLamberts(nL)

deltaV_actual = data[:,2] - B0_m # Difference between observed zenith brightness in magnitudes 
                                 # and night-sky brightness without the moon in magnitudes

# Illuminance of the moon outisde the atmosphere
def Istar(m):
    return 10**(-0.4*(m+16.57))

# Scattering function
def f(rho_deg):
    rho = np.deg2rad(rho_deg)
    return (10**(5.36))*(1.06+(np.cos(rho)**2)) + 10**(6.15-(rho_deg/40))

# Optical pathlength along a line of sight in units of air masses
def X(Z_deg):
    Z = np.deg2rad(Z_deg)
    return (1-0.96*(np.sin(Z)**2))**-0.5

# Difference between observed zenith brightness in magnitudes 
# and night-sky brightness without the moon in magnitudes
def deltaV(Bmoon, * args):
    args = np.array(args[0], dtype=np.float64)
    return -2.5*np.log10((Bmoon+B0_nL)/B0_nL) - args

# Brightness near zenith due to the moon in nanoLamberts
def Bmoon(k, * args):
    args = np.array(args[0], dtype=np.float64)
    return f(rho_deg)*Istar(m)*(10**(-0.4*k*X(Zm)))*(1-10**(-0.4*k*X(Z))) - args

# Solve for Bmoon using the deltaV function
nrows = deltaV_actual.shape[0]
Bmoon_nL = fsolve(deltaV, np.ones(nrows), args=deltaV_actual)
print(Bmoon_nL)

# Plot time versus britghness due to the moon near zenith(magnitudes)
plt.plot(Bmoon_nL[50:500])
plt.xlabel("Time(minutes)")
plt.ylabel("Britghness Due to the Moon Near Zenith(Magnitudes)")
plt.xticks([0,100,200,300,400,500],[data[0,1],data[100,1],data[200,1],data[300,1],data[400,1],data[500,1]])
plt.show()
plt.close()

# Solve for extinction coefficient k using the Bmoon function
mrows = Bmoon_nL.shape[0]
k = fsolve(Bmoon, np.ones(mrows)*-5, args=Bmoon_nL)
print(k)

# Plot time versus xxtinction coefficient k(mag/airmass)
plt.plot(k)
plt.xlabel("Time(minutes)")
plt.ylabel("Extinction Coefficient k(mag/airmass)")
plt.xticks([0,100,200,300,400,500],[data[0,1],data[100,1],data[200,1],data[300,1],data[400,1],data[500,1]])
plt.show()
plt.close()




