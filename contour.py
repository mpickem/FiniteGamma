#! /usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb


x = np.linspace(-3,3,601)
y = np.linspace(-3,3,601)
xv, yv = np.meshgrid(x,y) # vectorized

betaeval = scipy.special.beta(xv,yv)


g = plt.figure(figsize=(8,8))
ax = g.add_axes([0.1,0.1,0.8,0.8])
ax.pcolormesh(x,y,betaeval,vmin=-30,vmax=30)

cax = g.add_axes([0.91,0.1,0.02,0.8])
clb.ColorbarBase(ax=cax)
plt.show()
