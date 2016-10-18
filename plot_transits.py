import matplotlib.pyplot as plt
import utils
plt.style.use('ggplot')
import pickle
import numpy as np
import glob

# Check lightcurves and planet data:
fname = glob.glob('data/*.pkl')
print '\n'
for i in range(len(fname)):
    print '\t ('+str(i)+') '+fname[i]
print '\n'
dataset = fname[int(raw_input('\t Which one? '))]
pdata = utils.planet_data(raw_input('\t Planet name? '))
out_dir_name = 'results/'+dataset[:-4].split('/')[-1]

in_data = pickle.load(open(dataset,'r'))
files = glob.glob(out_dir_name+'/wbin*.pkl')
out_wl_dict = pickle.load(open(out_dir_name+'/wl_mcmc.pkl','rb'))
pdata['TT'] = np.median(out_wl_dict['t0'])
pdata['DEPTH'] = (np.median(out_wl_dict['p']))**2
pdata['AR'] = np.median(out_wl_dict['a'])
pdata['I'] = np.median(out_wl_dict['inc'])

sys_model = utils.get_sys_model(out_wl_dict)
lc_model = utils.get_lc_model(out_wl_dict,pdata)
times_interp,lc_model_interp = utils.get_lc_model(out_wl_dict,pdata,interpolate=True)
plt.plot((out_wl_dict['times']-np.median(out_wl_dict['t0']))*24,(out_wl_dict['target']/sys_model)+i*0.01,'.')
plt.plot((times_interp-np.median(out_wl_dict['t0']))*24,lc_model_interp+i*0.01,color='black')
plt.plot((out_wl_dict['times']-np.median(out_wl_dict['t0']))*24+5,(out_wl_dict['target']/sys_model)-lc_model+i*0.01+1.,'.')
plt.plot((out_wl_dict['times']-np.median(out_wl_dict['t0']))*24+5,np.ones(len(lc_model))+i*0.01,'-',color='black')
plt.xlabel('Time from mid-transit (hours)')
plt.ylabel('Relative flux + constant')
plt.title('White light curve')
plt.show()

wbins = np.zeros(len(in_data['wbins']))
wbins_err = np.zeros(len(in_data['wbins']))
presults = np.zeros(len(wbins))
presults_err = np.zeros(len(wbins))
for i in range(len(files)):
    fin = open(files[i],'rb')
    out = pickle.load(fin)
    fin.close()
    sys_model = utils.get_sys_model(out)
    lc_model = utils.get_lc_model(out,pdata)
    times_interp,lc_model_interp = utils.get_lc_model(out,pdata,interpolate=True)
    plt.plot((out['times']-np.median(out_wl_dict['t0']))*24,(out['target']/sys_model)+i*0.02,'o')
    plt.plot((times_interp-np.median(out_wl_dict['t0']))*24,lc_model_interp+i*0.02,color='black')
    plt.plot((out['times']-np.median(out_wl_dict['t0']))*24+5,(out['target']/sys_model)-lc_model+i*0.02+1.,'o')
    plt.plot((out['times']-np.median(out_wl_dict['t0']))*24+5,np.ones(len(lc_model))+i*0.02,'-',color='black')
    presults[i] = np.median(out['p'])
    presults_err[i] = np.sqrt(np.var(out['p']))
    wbins[i] = np.mean(in_data['wbins'][i])
    wbins_err[i] = np.max(in_data['wbins'][i])-wbins[i]

plt.xlabel('Time from mid-transit (hours)')
plt.ylabel('Relative flux + constant')
plt.show()

plt.errorbar(wbins,presults,xerr=wbins_err,yerr=presults_err,fmt='o')
plt.xlabel('Wavelength (A)')
plt.ylabel('$R_p/R_*$')
plt.show()
