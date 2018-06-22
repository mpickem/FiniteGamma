#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.special import digamma
import sys

def ek(kx,ky,kz): # tight binding
  return -2*np.cos(kx/100*2*np.pi)-2*np.cos(ky/100*2*np.pi)-2*np.cos(kz/100*2*np.pi)

def occ(e,mu,Gamma,beta):
  return 0.5 + digamma(0.5 + beta/2.0/np.pi*(Gamma - 1j*(e-mu))).imag/np.pi

# energy grid
epsilon = np.fromfunction(ek, (200,200,200))

n_target = 0.40 # element 0,1 # for a specific spin
Gamma = 0.1
beta = 20

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
print('Done.')
print()
