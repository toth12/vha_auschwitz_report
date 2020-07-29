from pyemma import msm,plots
from pyemma import plots as mplt
import numpy as np
import pdb
import matplotlib.pyplot as plt

P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
              [0.1,  0.75, 0.05, 0.05, 0.05],
              [0.05,  0.1,  0.8,  0.0,  0.05],
              [0.0,  0.2, 0.0,  0.8,  0.0],
              [0.0,  0.02, 0.02, 0.0,  0.96]])

M = msm.markov_model(P)
pos = np.array([[2.0,-1.5],[1,0],[2.0,1.5],[0.0,-1.5],[0.0,1.5]])
pl = mplt.plot_markov_model(M, pos=pos)

#pl[0].show()

A = [0]
B = [4]
tpt = msm.tpt(M, A, B)

# get tpt gross flux
F = tpt.gross_flux
print('**Flux matrix**:')
print(F)
print('**forward committor**:')
print(tpt.committor)
print('**backward committor**:')
print(tpt.backward_committor)
# we position states along the y-axis according to the commitor
tptpos = np.array([tpt.committor, [0,0,0.5,-0.5,0]]).transpose()
print('\n**Gross flux illustration**: ')
pl = mplt.plot_flux(tpt, pos=tptpos, arrow_label_format="%10.4f", attribute_to_plot='gross_flux')


"""Note that in the gross flux above we had fluxes between states 1 and 3.
 These fluxes just represent that some reactive trajectories leave 1, go to 3, 
 go then back to 1 and then go on to the target state 4 directly or via state 2.
  This detour via state 3 is usually of no interest, and their contribution is 
  already contained in the main paths 0->1 and 1->{2,4}. Therefore we remove all n
  onproductive recrossings by taking the difference flux between pairs of states that 
  have fluxes in both directions. This gives us the net flux."""



# get tpt net flux
Fp = tpt.net_flux
# or: tpt.flux (it's the same!)
print('**Net-Flux matrix**:')
print(Fp)
# visualize
pl = mplt.plot_flux(tpt, pos=tptpos, arrow_label_format="%10.4f", attribute_to_plot='net_flux')



""""

Quantitatively, the flux can be interpreted as the number of transition events along a certain pathway per time unit (where time unit means the lag time used to construct the transition matrix). In our toy model, we don’t have a physical lag time, so let’s call the time unit just ‘1 step’. The total flux from A->B is given by the total flux that leaves A, or identically the total flux that enters B. It is identical between the gross flux and the net flux.
The total flux can also be shown to be identical to the inverse round-trip time between A and B (that is the sum of the mean first passage times from A to B and from B to A). Often we are interested in the A->B time or the corresponding rate. This is already computed in the TPT object as well:
"""


print('Total TPT flux = ', tpt.total_flux)
print('Rate from TPT flux = ', tpt.rate)
print('A->B transition time = ', 1.0/tpt.rate)

print('mfpt(0,4) = ', M.mfpt(0, 4))

pl = mplt.plot_flux(tpt, pos=tptpos, flux_scale=100.0/tpt.total_flux, arrow_label_format="%3.1f")
fig = pl[0]
fig.axes[0].set_ylabel('committor')

#Text(0,0.5,'committor')

(bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.95)
cumflux = 0
print("Path flux\t\t%path\t%of total\tpath")
for i in range(len(bestpaths)):
    cumflux += bestpathfluxes[i]
    print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])


Fsub = tpt.major_flux(fraction=0.95)
print(Fsub)
Fsubpercent = 100.0 * Fsub / tpt.total_flux
plt=mplt.plot_network(Fsubpercent, pos=tptpos, state_sizes=tpt.stationary_distribution, arrow_label_format="%3.1f")
pdb.set_trace()