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

  # this is broadcastable
  def occ(self,e,mu):
    return 0.5 + digamma(0.5 + self.beta/2.0/np.pi*(self.Gamma - 1j*(e-mu))).imag/np.pi

  # we cannot use the polygamma function
  # since it only accepts real numbers
  # psi_n = zeta(n+1) * n! * (-1)**(n+1)
  # this zeta is the Hurwitz zeta function

  # these are not broadcastable
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


class Hamiltonian(object):
  htype = 'General'
  def __init__(self, kx, ky, kz):
    self.kx = kx
    self.ky = ky
    self.kz = kz
    self.nkp = kx*ky*kz
    self.hk = None
    self.ndim = None
    self.__define_dim()

  def __define_dim(self):
    if self.ky == 1 and self.kz == 1:
      self.ndim = 1
    elif self.kz == 1:
      self.ndim = 2
    else:
      self.ndim = 3

class TightBinding(Hamiltonian):
  htype = 'Tightbinding'
  def __init__(self, t, kx, ky=1, kz=1):
    Hamiltonian.__init__(self, kx, ky, kz)
    self.t = t
    self.__create_hk()

  def __create_hk(self):
    if self.ndim == 1:
      self.__create_1d_hk()
    elif self.ndim == 2:
      self.__create_2d_hk()
    else:
      self.__create_3d_hk()

  def __create_3d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.kmeshz = np.linspace(0,1,self.kz,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:,None] \
                          + np.cos(self.kmeshz*2*np.pi)[None,None,:] )

  def __create_2d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:] )

  def __create_1d_hk(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi) )

class Wannier(Hamiltonian):
  htype = 'Wannier'
  def __init__(self, fname, kx, ky, kz):
    Hamiltonian.__init__(self, kx, ky, kz)
    self.fname = fname
    self.__create_hk()

  def __create_hk(self):
    with open(self.fname, 'r') as f:
      first_line = f.readline()
      nkpfile, ndimfile = map(int, first_line.split()[:2])
      if (self.nkp != nkpfile):
        print('Differing number of k-points in Wannier File.')
        sys.exit()
      elif (self.ndim != ndimfile):
        print('Differing number of bands in Wannier File.')
        sys.exit()
      else:
        pass
      # extract hk now
      self.hk = None


class FGProblem(object):
  def __init__(self, response, hk, ntarget):
    self.resp = response # FiniteGamma object
    self.hk = hk # hamiltonian array
    self.ntarget = ntarget # target occupation number
    self.mu = None
    self.nbisec = None
    self.sxx = None
    self.sxy = None
    self.axx = None
    self.axy = None

  def findmu(self, mu_start = -100, mu_end = 100, threshold = 1e-7, progress=False):
    mu_bisec = (mu_start+mu_end)/2
    mu_save = mu_bisec # we save this

    n_bisec = np.average(self.resp.occ(self.hk, mu_bisec))
    n_start = np.average(self.resp.occ(self.hk, mu_start))
    n_end = np.average(self.resp.occ(self.hk, mu_end))

    iterator = 0

    while ( np.abs(self.ntarget - n_bisec)  > threshold ):
      n_bisec = np.average(self.resp.occ(self.hk, mu_bisec))
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

    self.mu =  mu_save
    self.nbisec = n_bisec

  def calcprop(self, progress=False):
    sxx = sxy = axx = 0.0
    Hk_progress = self.hk.size/50
    for index, eki in np.ndenumerate(self.hk.flatten()):
      if (progress and (np.mod(index[0],Hk_progress) == 0)):
        print(index[0]/self.hk.size * 100,'%')
      p1,p2,p3 = self.resp.get_polygamma(eki, self.mu)
      sxx += self.resp.sigma_xx(eki, self.mu, p1, p2)
      sxy += self.resp.sigma_xy(eki, self.mu, p1, p2, p3)
      axx += self.resp.alpha_xx(eki, self.mu, p1, p2)
    self.sxx = sxx/self.hk.size
    self.sxy = sxy/self.hk.size
    self.axx = axx/self.hk.size

if __name__ == '__main__':
  print('Finite Gamma calculation program')
  print()

  # wann = Wannier('SVO_k20.hk', 20, 20, 20)
  # sys.exit()

  # tight binding object - automatically creates .hk array
  tb = TightBinding(t = 0.1, kx=20, ky=20, kz=1)

  beta_list = []
  mu = []
  sxx = []
  sxy = []
  axx = []

  # temperature loop
  for i in xrange(100,1,-1):
    beta = i/10
    beta_list.append(beta)
    print(beta)
    # response object
    resp = FiniteGamma(Gamma=0.1, beta=beta, Z=0.7)
    # class which combines the resp + hamiltonian
    comb = FGProblem(resp, tb.hk, ntarget = 0.6)

    # find chemical potential
    comb.findmu(mu_start=-10, mu_end=10, threshold=1e-10)
    # calculatie properties
    comb.calcprop(progress=False)

    mu.append(comb.mu)
    sxx.append(comb.sxx)
    sxy.append(comb.sxy)
    axx.append(comb.axx)

  np.savetxt('results.dat', np.c_[beta_list,mu,sxx,sxy,axx])
  print('Done.')
