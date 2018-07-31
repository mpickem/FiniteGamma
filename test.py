#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import fg
import h5py
import sys

def main():
  # initiate output
  try:
    f = h5py.File('half-filling.hdf5','w-')
  except IOError:
    print('File already exists...')
    print('Exiting.')
    sys.exit(1)

  # create Hamiltonian
  ham = fg.TightBinding(t=1,kx=80, ky=80)

  # initiate temporary lists
  beta_list = []
  sxy_list = []

  for beta in xrange(1000,0,-4):
    print(beta)
    resp = fg.FiniteGamma(Z=0.5,Gamma=0.3,beta=beta)
    problem = fg.FGProblem(resp,ham.hk,ntarget=0.5) # half-filling
    problem.findmu()
    problem.calcprop()
    beta_list.append(beta)
    sxy_list.append(problem.sxy)

  # transform lists to np arrays
  beta_array = np.array(beta_list, dtype=np.float64)
  sxy_array = np.array(sxy_list, dtype=np.float64)


  # output
  f['beta'] = beta_array
  f['output/sxy'] = sxy_array
  f.close()

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('\nKilled by user.')
    sys.exit(0)
  else:
    print('\nDone.')
