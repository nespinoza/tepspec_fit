# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import math
import utils
import numpy as np
import glob
import pickle


# Check lightcurves and planet data:
fname = glob.glob('data/*.pkl')
print '\n\t Available datasets:'
for i in range(len(fname)):
    print '\t ('+str(i)+') '+fname[i]
print '\n'
dataset = fname[int(raw_input('\t Which one? '))]
print '\n\t LD laws to fit:'
ld_laws = ['quadratic','squareroot','logarithmic']
for i in range(len(ld_laws)):
    print '\t ('+str(i)+') '+ld_laws[i]
ld_law = ld_laws[int(raw_input('\t Which one? '))]
pdata = utils.planet_data(raw_input('\t Planet name? '))

# Load lighturves:
LC = pickle.load(open(dataset,'r'))

# Load times:
times = LC['t']

# Loading comp-file:
print '\t Reading in comp file if available (',dataset[:-3]+'.dat',')'
if os.path.exists(dataset[:-3]+'dat'):
    fcomp = open(dataset[:-3]+'dat','r')
    firstline = fcomp.readline()
    idx_comp = firstline.split(':')[1].split(',')
    idx_comp[-1] = idx_comp[-1].split('\n')[0]
    idx_comp = np.array(idx_comp).astype('int')
    secondline = fcomp.readline()
    if secondline == '':
        idx_times = range(len(times))
    else:
        exec 'idx_times = '+secondline
        thirdline = fcomp.readline()
        if thirdline != '':
            outliers = np.array(thirdline.split(',')).astype('int')
            for i in range(len(outliers)):
                outlier = outliers[i]
                idx_out = np.where(outlier==idx_times)[0]
                if len(idx_out)>0:
                    idx_times = np.append(idx_times[:idx_out[0]],idx_times[idx_out[0]+1:])
    fcomp.close()
else:
    idx_comp = range(LC['cLC'].shape[1])
    idx_times = range(len(times))

# Create results folder:
if not os.path.exists('results'):
    os.mkdir('results')

out_dir_name = 'results/'+dataset[:-4].split('/')[-1]
if not os.path.exists(out_dir_name):
    os.mkdir(out_dir_name)

times = times[idx_times]
# Get t0. We try the corresponding TT at the beggining of 
# observations, supposing we caught the transit. However, it 
# could be that we only caught the egress, and because of this 
# we try n and n-1:
n = int(((times[0]-pdata['TT'])/pdata['PER']))+1
ndown = n-1
t0 = pdata['TT'] + n*pdata['PER']
t0down = pdata['TT'] + ndown*pdata['PER']
diff1 = np.abs(times[0]-t0)
diff2 = np.abs(times[0]-t0down)
d = np.min([diff1,diff2])
if d == diff2:
    t0 = t0down

pdata['TT'] = t0
# Now that we have t0, we fit the data. We pass all the external parameters
# in order to search for correlations:
dt = times-times[0]
#g = LC['g'][idx_times]
#X = np.copy(g)
#X = np.copy(LC['Z'][idx_times])
#X = np.vstack((X,g))
#X = np.vstack((X,g**2))
#X = np.vstack((X,g**3))
X = np.copy(dt)
#X = np.vstack((X,dt**2))
#X = np.copy(LC['Z'][idx_times])
X = np.vstack((X,dt**2))
X = np.vstack((X,dt**3))
#X = np.vstack((X,dt**4))

# First, we fit the white-light curve:
if not os.path.exists(out_dir_name+'/wl_mcmc.pkl'):
    #out_wl_dict = utils.fit_lcs(times,LC['oLC'][idx_times],LC['cLC'][idx_times,:][:,idx_comp],\
    #out_wl_dict = utils.fit_lcs(times,LC['cLC'][idx_times,2],LC['cLC'][idx_times,:][:,idx_comp],\
    #                         pdata,Xin=X,plot=True,white_light=True,ld_law = ld_law)
    out_wl_dict = utils.simple_fit_lcs(times,LC['oLC'][idx_times],LC['cLC'][idx_times,:][:,idx_comp],\
                              pdata,Xin=X,plot=True,white_light=True,ld_law = ld_law)
    fout = open(out_dir_name+'/wl_mcmc.pkl','wb')
    pickle.dump(out_wl_dict,fout)
    fout.close()

else:
    fin = open(out_dir_name+'/wl_mcmc.pkl','rb')
    out_wl_dict = pickle.load(fin) 
    fin.close()

pdata['TT'] = np.median(out_wl_dict['t0'])
pdata['DEPTH'] = (np.median(out_wl_dict['p']))**2
pdata['AR'] = np.median(out_wl_dict['a'])
pdata['I'] = np.median(out_wl_dict['inc'])

sys_model = utils.get_sys_model(out_wl_dict)
lc_model = utils.get_lc_model(out_wl_dict,pdata)
plt.style.use('ggplot')
plt.plot((out_wl_dict['times']-np.median(out_wl_dict['t0']))*24,out_wl_dict['target']/sys_model,'.')
plt.plot((out_wl_dict['times']-np.median(out_wl_dict['t0']))*24,lc_model)
plt.xlabel('Hours from mid-transit')
plt.ylabel('Relative flux')
plt.show()

# Apply common-mode correction:
#X = out_wl_dict['target']/lc_model

# Now, fit wavelength-by-wavelength:
for i in range(LC['oLCw'].shape[1]):
    print 'wbin ',i
    if not os.path.exists(out_dir_name+'/wbin_'+str(i)+'.pkl'):
        try:
           out_dict = utils.simple_fit_lcs(times,LC['oLCw'][idx_times,i],LC['cLCw'][idx_times,:,:][:,idx_comp,:][:,:,i],\
                                           pdata,Xin=X,plot=False,ld_law = ld_law)
           #out_dict = utils.fit_lcs(times,LC['oLCw'][idx_times,i],LC['cLCw'][idx_times,:,:][:,idx_comp,:][:,:,i],#
           #                         pdata,Xin=X,plot=False,ld_law = ld_law)
           fout = open(out_dir_name+'/wbin_'+str(i)+'.pkl','wb')
           pickle.dump(out_dict,fout)
           fout.close()
        except:
           print 'Bin fit failed.'

# Now, get SVD of the comparison stars:
#V,eigenvals,PC = utils.classic_PCA(comparisons.transpose())

# Fit to out-of-transit data:
#idx = np.where((times<t0-((pdata['T14']/2.))) | (times>t0+((pdata['T14']/2.))))[0]
#plt.plot(times,target,'.-')
#plt.plot(times[idx],target[idx],'o')
#plt.show()
# Add constant to predictor matrix:
#X = np.ones(len(idx))
#X = np.vstack((X,PC[:,idx]))
#print X.shape
# Perform linear fit:
#coeffs = utils.linear_fit(X,target[idx])
#print coeffs
#model = coeffs[0]
#for i in range(1,PC.shape[0]):
#    print i
#    model = model + coeffs[i]*PC[i-1,:]
#plt.plot(times,target,'.-')
#plt.plot(times,model,'-')
#plt.plot(times,target/model,'.-')
#plt.show()
# Check the lightcurves and identify nans, outliers and/or zero-count data:
#for i in range(comparisons.shape[1]):
#    plt.plot(times,comparisons[:,i]/np.median(comparisons[:,i]),color='black',alpha=0.1)
#plt.plot(times,target/np.median(target))
#plt.plot([t0,t0],[-1,1])
#plt.show()
