#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.special import digamma
from mpmath import zeta
import sys

class FiniteGamma(object):
  def __init__(self, Gamma, beta, Z):
    self.Gamma = Gamma
    self.beta = beta
    self.Z = Z

  def occ(self,e,mu):
    return 0.5 + digamma(0.5 + self.beta/2.0/np.pi*(self.Gamma - 1j*(e-mu))).imag/np.pi

  # we cannot use the polygamma function
  # since it only accepts real numbers
  # psi_n = zeta(n+1) * n! * (-1)**(n+1)
  # this zeta is the Hurwitz zeta function
  def get_polygamma(self,e,mu):
    psi_1 = zeta(2,0.5+self.beta/2/np.pi*(self.Gamma+1j*(e-mu)))
    psi_2 = zeta(3,0.5+self.beta/2/np.pi*(self.Gamma+1j*(e-mu))) * (-2)
    psi_3 = zeta(4,0.5+self.beta/2/np.pi*(self.Gamma+1j*(e-mu))) * 6
    return psi_1, psi_2, psi_3

  def sigma_xx(self,e,mu,psi_1,psi_2):
    return self.Z**2*self.beta/(4*np.pi**3*self.Gamma) * \
           (psi_1.real - \
            psi_2.real * self.Gamma*self.beta/2/np.pi * (-2) )

  def alpha_xx(self,e,mu,psi_1,psi_2):
    return self.Z**2*self.beta/(4*np.pi**3*self.Gamma) * \
            ( (e-mu) * psi_1.real - \
              (e-mu)*self.Gamma * self.beta/2/np.pi * psi_2.real - \
              self.Gamma**2*self.beta/2/np.pi * psi_2.imag )

  def sigma_xy(self,e,mu,psi_1,psi_2,psi_3):
    return (-self.Z**3*self.beta/(32*np.pi**6*self.Gamma**2)) * \
           ( psi_3.real * self.beta**2 * self.Gamma**2 + \
             psi_2.real * 3*self.beta*np.pi*self.Gamma - \
             psi_1.real * 6*np.pi**2 )

  def alpha_xy():
    pass


class TightBinding(object):
  def __init__(self, t, kx, ky=1, kz=1):
    self.t = t
    self.kx = kx
    self.ky = ky
    self.kz = kz
    self.hk = None

  def create_hk(self):
    if self.ky == 1 and self.kz == 1:
      self.create_1d_hk()
    elif self.kz == 1:
      self.create_2d_hk()
    else:
      self.create_3d_hk()

  def create_3d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.kmeshz = np.linspace(0,1,self.kz,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:,None] \
                          + np.cos(self.kmeshz*2*np.pi)[None,None,:] )

  def create_2d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:] )

  def create_1d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi) )

class Bisection(object):
  def __init__(self, response, hk, ntarget):
    self.response = response # FiniteGamma object
    self.hk = hk # hamiltonian array
    self.ntarget = ntarget # target occupation number


  def findmu(self, mu_start = -100, mu_end = 100, threshold = 1e-7, progress=False):
    mu_bisec = (mu_start+mu_end)/2
    mu_save = mu_bisec # we save this

    n_bisec = np.average(resp.occ(self.hk, mu_bisec))
    n_start = np.average(resp.occ(self.hk, mu_start))
    n_end = np.average(resp.occ(self.hk, mu_end))

    iterator = 0

    while ( np.abs(self.ntarget - n_bisec)  > threshold ):
      n_bisec = np.average(resp.occ(tb.hk, mu_bisec))
      mu_save = mu_bisec # we save this
      # reasoning: if the target occupation is in the acceptable range
      # we afterwards immediately adjust the mu again
      # mu_save now represents the according mu for the target occupation
      if progress:
        print(iterator,': ',n_bisec, mu_save)
      iterator += 1
      if n_bisec > self.ntarget:
        mu_end = mu_bisec
        n_end = n_bisec
        mu_bisec = (mu_bisec + mu_start)/2.0
      else:
        mu_start = mu_bisec
        n_start = n_bisec
        mu_bisec = (mu_end + mu_bisec)/2.0

    return mu_save, n_bisec


class Quantities(object):
  def __init__(self, response, hk, mu):
    self.response = response
    self.hk = hk
    self.mu = mu

  def calculate(self, progress=False):
    Hk_progress = self.hk.size//50
    # now we can calculate some fancy properties
    # zeta functions are not broadcastable unfortunately
    sigma_xx_value = 0.0
    sigma_xy_value = 0.0
    alpha_xx_value = 0.0

    for index, eki in np.ndenumerate(self.hk.flatten()):
      if (progress and (np.mod(index[0],Hk_progress) == 0)):
        print(index[0]/self.hk.size * 100,'%')
      # polygamma functions
      psi_1,psi_2,psi_3 = self.response.get_polygamma(eki,self.mu)
      # quantities
      sigma_xx_value += self.response.sigma_xx(eki,self.mu,psi_1, psi_2)
      sigma_xy_value += self.response.sigma_xy(eki,self.mu,psi_1, psi_2, psi_3)
      alpha_xx_value += self.response.alpha_xx(eki,self.mu,psi_1, psi_2)

    sigma_xx_value = sigma_xx_value/self.hk.size
    sigma_xy_value = sigma_xy_value/self.hk.size
    alpha_xx_value = alpha_xx_value/self.hk.size

    return sigma_xx_value, sigma_xy_value, alpha_xx_value







print('Finite Gamma calculation program')
print()

# response object
resp = FiniteGamma(Gamma = 0.1, beta = 10, Z = 0.5)
# tight binding object
tb = TightBinding(t = 0.1, kx=80, ky=80, kz=1)
tb.create_hk()

print('Chemical potential search:')
bisec = Bisection(resp, tb.hk, ntarget = 0.3)
mu,occ = bisec.findmu()

print()
print('target occupation:', bisec.ntarget)
print('achieved occupation:', occ)
print('calculated chemical potential:', mu)

print('Physical properties:')
quant = Quantities(resp, tb.hk, mu)
quant.calculate(progress=True)

print('Done.')
