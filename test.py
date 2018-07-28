import fg

ham = fg.Wannier('SVO_k20.hk', 20, 20, 20)
for key, value in ham.__dict__.iteritems():
  print key, value

resp = fg.FiniteGamma(Z=0.5, beta=40, Gamma=1.2)
for key, value in resp.__dict__.iteritems():
  print key, value

comb = fg.FGProblem(resp, ham.hk, ntarget=1.0) # 1e in the t2g subspace

# comb.findmu(progress = True)
# comb.calcprop(progress = True)
