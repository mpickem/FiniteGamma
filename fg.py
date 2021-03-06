#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.special import digamma
from mpmath import zeta
import sys
import h5py

class FiniteGamma(object):
  def __init__(self, Gamma, beta, Z):
    self.Gamma = Gamma
    self.beta = beta
    self.Z = Z

  # this is broadcastable
  def occ(self,e,mu):
    return 0.5 - digamma(0.5 + self.beta/2.0/np.pi*(self.Gamma + 1j*(e-mu))).imag/np.pi

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
            psi_2.real * self.Gamma*self.beta/2/np.pi)

  def alpha_xx(self,e,mu,psi_1,psi_2):
    return self.Z**2*self.beta**2/(4*np.pi**3*self.Gamma) * \
            ( (e-mu) * psi_1.real - \
              (e-mu)*self.Gamma * self.beta/2/np.pi * psi_2.real - \
              self.Gamma**2*self.beta/2/np.pi * psi_2.imag )

  def sigma_xy(self,e,mu,psi_1,psi_2,psi_3):
    return (-self.Z**3*self.beta/(16*np.pi**4*self.Gamma**2)) * \
           ( - psi_3.real * self.beta**2 * self.Gamma**2 / 4 / np.pi**2 + \
             psi_2.real * 3*self.beta*self.Gamma / 2 / np.pi - \
             psi_1.real * 3 )

  def alpha_xy(self,e,mu,psi_1,psi_2,psi_3):
    return self.beta**2 * self.Z**3 / 16 / self.Gamma**2 / np.pi**4 * \
           ( psi_3.real * (e-mu)*self.Gamma**2*self.beta**2 / 4 / np.pi**2 + \
             psi_3.imag * self.Gamma**3*self.beta**2 / 4 / np.pi**2 - \
             psi_2.real * 3 *(e-mu)*self.Gamma*self.beta / 2 / np.pi - \
             psi_2.imag * self.Gamma**2 * self.beta / 2 / np.pi + \
             psi_1.real * 3 *(e-mu) )


class Hamiltonian(object):
  htype = 'General'
  def __init__(self, kx, ky, kz):
    self.kx = kx
    self.ky = ky
    self.kz = kz
    self.nkp = kx*ky*kz
    self.hk = None # H(k)
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
  ''' Create a tightbinding Hamiltonian with 1 2 or 3 dimensions
      lattice constant is fixed to a=1
  '''
  htype = 'Tightbinding'
  def __init__(self, t, kx, ky=1, kz=1):
    Hamiltonian.__init__(self, kx, ky, kz)
    self.t = t
    self.__create()

  def __create(self):
    if self.ndim == 1:
      self.__create_1d()
    elif self.ndim == 2:
      self.__create_2d()
    else:
      self.__create_3d()

  def __create_3d(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.kmeshz = np.linspace(0,1,self.kz,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:,None] \
                          + np.cos(self.kmeshz*2*np.pi)[None,None,:] )
    self.vkx =  2*self.t * np.sin(self.kmeshx*2*np.pi)
    self.vky =  2*self.t * np.sin(self.kmeshy*2*np.pi)
    self.vkz =  2*self.t * np.sin(self.kmeshz*2*np.pi)

  def __create_2d(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.kmeshy = np.linspace(0,1,self.ky,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi)[:,None] \
                          + np.cos(self.kmeshy*2*np.pi)[None,:] )
    self.vkx =  2*self.t * np.sin(self.kmeshx*2*np.pi)
    self.vky =  2*self.t * np.sin(self.kmeshy*2*np.pi)

  def __create_1d(self):
    self.kmeshx = np.linspace(0,1,self.kx,endpoint=False)
    self.hk = -2*self.t * ( np.cos(self.kmeshx*2*np.pi) )
    self.vkx = 2*self.t * np.sin(self.kmeshx*2*np.pi)

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
  def __init__(self, response, hamilton, ntarget):
    self.resp = response     # FiniteGamma object
    self.hamilton = hamilton # Hamiltonian object
    self.ntarget = ntarget   # target occupation number
    self.mu = None
    self.nbisec = None
    self.sxx = None
    self.sxy = None
    self.axx = None
    self.axy = None

  def findmu(self, mu_start = -100, mu_end = 100, threshold = 1e-7, progress=False):
    mu_bisec = (mu_start+mu_end)/2
    mu_save = mu_bisec # we save this

    n_bisec = np.average(self.resp.occ(self.hamilton.hk, mu_bisec))
    n_start = np.average(self.resp.occ(self.hamilton.hk, mu_start))
    n_end = np.average(self.resp.occ(self.hamilton.hk, mu_end))

    iterator = 0

    while ( (iterator < 250) and (np.abs(self.ntarget - n_bisec)  > threshold) ):
      n_bisec = np.average(self.resp.occ(self.hamilton.hk, mu_bisec))
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
    else:
      if (iterator == 250):
        print('Particle number not converged')
        print('Exiting...')
        sys.exit(0)

    self.mu =  mu_save
    self.nbisec = n_bisec

  # this is for 2 dimensions and higher
  def calcprop(self, progress=False):
    sxx = sxy = axx = axy = 0.0
    Hk_progress = self.hamilton.hk.size/50
    # flatten hk array and iterate over it
    for index, eki in np.ndenumerate(self.hamilton.hk.flatten()):
      # get the polygamma functions
      p1,p2,p3 = self.resp.get_polygamma(eki, self.mu)
      # unravel the index again for the fermi velocities vkx,vky
      ikx,iky = np.unravel_index(index, self.hamilton.hk.shape)[:2]
      # calculate the properties with attached fermi velocities
      sxx += self.resp.sigma_xx(eki, self.mu, p1, p2) * self.hamilton.vkx[ikx]**2
      sxy += self.resp.sigma_xy(eki, self.mu, p1, p2, p3) * self.hamilton.vkx[ikx]*self.hamilton.vky[iky]
      axx += self.resp.alpha_xx(eki, self.mu, p1, p2) * self.hamilton.vkx[ikx]**2
      axy += self.resp.alpha_xy(eki, self.mu, p1, p2, p3) * self.hamilton.vkx[ikx]*self.hamilton.vky[iky]

      # progress output
      if (progress and (np.mod(index[0]+1,Hk_progress) == 0)):
        print((index[0]+1)/self.hamilton.hk.size*100,'%')

    # k summations always have to be normalized
    self.sxx = sxx/self.hamilton.hk.size
    self.sxy = sxy/self.hamilton.hk.size
    self.axx = axx/self.hamilton.hk.size
    self.axy = axy/self.hamilton.hk.size

def main():
  print('Finite Gamma calculation program')
  print()


  # HDF5 output
  try:
    hdf5 = h5py.File('results.hdf5','w-')
  except IOError:
    print('File already exists...')
    print('Exiting.')
    sys.exit(1)

  # make this size creation automatic ....
  mu_array = np.zeros((9,250), dtype=np.float64)
  sxx_array = np.zeros_like(mu_array, dtype=np.float64)
  sxy_array = np.zeros_like(mu_array, dtype=np.float64)
  axx_array = np.zeros_like(mu_array, dtype=np.float64)
  axy_array = np.zeros_like(mu_array, dtype=np.float64)

  # wann = Wannier('SVO_k20.hk', 20, 20, 20)
  # sys.exit()

  # tight binding object - automatically creates .hk array
  tb = TightBinding(t = 1, kx=40, ky=40, kz=1)

  for n in xrange(1,10): # from 1 to 9
    nparticles = n/10

    # for saving the values
    beta_list = []
    mu = []; sxx = []; sxy = []; axx = []; axy = []

    # temperature loop
    for i in xrange(1000,0,-4):
      beta = i
      beta_list.append(beta)
      print('particles: ', n, '  beta: ',beta)
      # response object
      resp = FiniteGamma(Gamma=0.3, beta=beta, Z=0.5)
      # class which combines the resp + hamiltonian
      comb = FGProblem(resp, tb, ntarget = nparticles)

      # find chemical potential
      comb.findmu(mu_start=-10, mu_end=10, threshold=1e-8, progress=False)
      # calculatie properties
      comb.calcprop(progress=False)

      mu.append(comb.mu)
      sxx.append(comb.sxx)
      sxy.append(comb.sxy)
      axx.append(comb.axx)
      axy.append(comb.axy)

    sxx_array[n-1,:] = np.array(sxx, dtype=np.float64)
    sxy_array[n-1,:] = np.array(sxy, dtype=np.float64)
    axx_array[n-1,:] = np.array(axx, dtype=np.float64)
    axy_array[n-1,:] = np.array(axy, dtype=np.float64)

    # np.savetxt('results_{:02}.dat'.format(n), np.c_[beta_list,mu,sxx,sxy,axx,axy])


  hdf5['output/sxx'] = sxx_array
  hdf5['output/sxy'] = sxy_array
  hdf5['output/axx'] = axx_array
  hdf5['output/axy'] = axy_array
  hdf5['beta'] = np.array(beta_list)

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('\nKilled by user.')
    sys.exit(0)
  else:
    print('\nDone.')
