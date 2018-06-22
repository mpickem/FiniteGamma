#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.special import digamma
from mpmath import zeta
import sys

def ek(kx,ky,kz): # tight binding
  return -2*np.cos(kx/40*2*np.pi)-2*np.cos(ky/40*2*np.pi)-2*np.cos(kz/40*2*np.pi)

def occ(e,mu,Gamma,beta):
  return 0.5 + digamma(0.5 + beta/2.0/np.pi*(Gamma - 1j*(e-mu))).imag/np.pi

# we cannot use the polygamma function
# since it only accepts real numbers
# psi_n = zeta(n+1) * (n-1)! * (-1)**(n+1)
# this zeta is the Hurwitz zeta function
def sigma_xx(e,mu,Gamma,beta,Z):
  return Z**2*beta/(4*np.pi**3*Gamma) * \
         (zeta(2,0.5+beta/2/np.pi*(Gamma-1j*(e-mu))).real - \
          zeta(3,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))).real * Gamma*beta/2/np.pi * (-2) )
def alpha_xx(e,mu,Gamma,beta,Z):
  return Z**2*beta/(4*np.pi**3*Gamma) * \
          ( (e-mu) * zeta(2,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))).real - \
            (e-mu)*Gamma*beta/2/np.pi * zeta(3,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))).real * (-2) - \
            Gamma**2*beta/2/np.pi * zeta(3,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))).imag * (-2) )

# energy grid
epsilon = np.fromfunction(ek, (40,40,40))

n_target = 0.40 # element 0,1 # for a specific spin
Gamma = 0.1
beta = 20
Z = 0.7

mu_start = -100
mu_end = 100
mu_bisec = (mu_end+mu_start)/2
mu_diff = mu_end-mu_start


n_start = np.average(occ(epsilon,mu_start,Gamma,beta))
n_end = np.average(occ(epsilon,mu_end,Gamma,beta))
n_bisec = np.average(occ(epsilon,mu_bisec,Gamma,beta))

iterator = 0

while ( np.abs(n_target - n_bisec)  > 1e-6 ):
  n_bisec = np.average(occ(epsilon,mu_bisec,Gamma,beta))
  mu_save = mu_bisec # we save this
  # reasoning: if the target occupation is in the acceptable range
  # we afterwards immediately adjust the mu again
  # mu_save now represents the according mu for the target occupation
  print(iterator,': ',n_bisec, mu_save)
  iterator += 1
  if n_bisec > n_target:
    mu_end = mu_bisec
    n_end = n_bisec
    mu_bisec = (mu_bisec + mu_start)/2.0
  else:
    mu_start = mu_bisec
    n_start = n_bisec
    mu_bisec = (mu_end + mu_bisec)/2.0

print()
print('target occupation:', n_target)
print('achieved occupation:', n_bisec)
print('calculated chemical potential:', mu_save)

# now we can calculate some fancy properties
# zeta functions are not broadcastable unfortunately
sigma_value = 0.0
alpha_value = 0.0
for eki in np.nditer(epsilon):
  sigma_value += sigma_xx(eki,mu_save,Gamma,beta,Z)
  alpha_value += alpha_xx(eki,mu_save,Gamma,beta,Z)
sigma_value = sigma_value/epsilon.size
alpha_value = alpha_value/epsilon.size

print('sigma_xx: ', sigma_value)
print('alpha_xx: ', alpha_value)

print('Done.')
print()
