#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.special import digamma
from mpmath import zeta
import sys

def occ(e,mu,Gamma,beta):
  return 0.5 + digamma(0.5 + beta/2.0/np.pi*(Gamma - 1j*(e-mu))).imag/np.pi

# we cannot use the polygamma function
# since it only accepts real numbers
# psi_n = zeta(n+1) * n! * (-1)**(n+1)
# this zeta is the Hurwitz zeta function
def get_polygamma(e,mu,Gamma,beta):
  psi_1 = zeta(2,0.5+beta/2/np.pi*(Gamma+1j*(e-mu)))
  psi_2 = zeta(3,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))) * (-2)
  psi_3 = zeta(4,0.5+beta/2/np.pi*(Gamma+1j*(e-mu))) * 6
  return psi_1, psi_2, psi_3

def sigma_xx(e,mu,Gamma,beta,Z,psi_1,psi_2):
  return Z**2*beta/(4*np.pi**3*Gamma) * \
         (psi_1.real - \
          psi_2.real * Gamma*beta/2/np.pi * (-2) )
def alpha_xx(e,mu,Gamma,beta,Z,psi_1,psi_2):
  return Z**2*beta/(4*np.pi**3*Gamma) * \
          ( (e-mu) * psi_1.real - \
            (e-mu)*Gamma*beta/2/np.pi * psi_2.real - \
            Gamma**2*beta/2/np.pi * psi_2.imag )
def sigma_xy(e,mu,Gamma,beta,Z,psi_1,psi_2,psi_3):
  return (-Z**3)*3/(8*np.pi**2*Gamma**2) * \
         (beta/2/np.pi**2 * psi_1.real - \
          beta**2*Gamma/4/np.pi**3 * psi_2.real - \
          beta**3*Gamma**2*2/(3*8*2*np.pi**4) * psi_3.real)


# k-mesh calculation
kmeshx = np.linspace(0,1,50,endpoint=False)
kmeshy = np.linspace(0,1,50,endpoint=False)
kmeshz = np.linspace(0,1,50,endpoint=False)
t_hopping = 1.0

# Hk calculation
Hk = -2*t_hopping * ( np.cos(kmeshx*2*np.pi)[:,None,None] + np.cos(kmeshy*2*np.pi)[None,:,None] \
                    + np.cos(kmeshz*2*np.pi)[None,None,:])


n_target = 0.485 # element 0,1 # for a specific spin
Gamma = 0.3
beta = 100
Z = 0.5

mu_start = -100
mu_end = 100
mu_bisec = (mu_end+mu_start)/2
mu_save = mu_bisec
mu_diff = mu_end-mu_start


n_start = np.average(occ(Hk,mu_start,Gamma,beta))
n_end = np.average(occ(Hk,mu_end,Gamma,beta))
n_bisec = np.average(occ(Hk,mu_bisec,Gamma,beta))

iterator = 0


print('Finite Gamma calculation program')
print()
print('Chemical potential search:')

while ( np.abs(n_target - n_bisec)  > 1e-7 ):
  n_bisec = np.average(occ(Hk,mu_bisec,Gamma,beta))
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

# for progress output
Hk_progress = Hk.size/50
# now we can calculate some fancy properties
# zeta functions are not broadcastable unfortunately
sigma_xx_value = 0.0
sigma_xy_value = 0.0
alpha_xx_value = 0.0
print()
print('Calculation progress of quantities:')
for index, eki in np.ndenumerate(Hk.flatten()):
  if (np.mod(index[0],Hk_progress) == 0):
    print(index[0]/Hk.size * 100,'%')
  # polygamma functions
  psi_1,psi_2,psi_3 = get_polygamma(eki,mu_save,Gamma,beta)
  # quantities
  sigma_xx_value += sigma_xx(eki,mu_save,Gamma,beta,Z, psi_1, psi_2)
  sigma_xy_value += sigma_xy(eki,mu_save,Gamma,beta,Z, psi_1, psi_2, psi_3)
  alpha_xx_value += alpha_xx(eki,mu_save,Gamma,beta,Z, psi_1, psi_2)

sigma_xx_value = sigma_xx_value/Hk.size
sigma_xy_value = sigma_xy_value/Hk.size
alpha_xx_value = alpha_xx_value/Hk.size


print('sigma_xx: ', sigma_xx_value)
print('sigma_xy: ', sigma_xy_value)
print('alpha_xx: ', alpha_xx_value)

print('Done.')
print()
