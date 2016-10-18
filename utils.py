# -*- coding: utf-8 -*-
from scipy.signal import medfilt
import batman
import numpy as np
import pickle
import os

def planet_data(pname):
    """
    Given a planet name, this function returns all the basic data for the planet
    from exoplanets.org stored in a dictionary, which is saved for future reference.
    Some useful keys:

             TT:        Time of transit center
             PER:       Period of the planet
             MSINI:     Minimum mass of the planet
             AR:        Semi-major eaxis over stellar radii
             DEPTH:     Transit depth (Rp/Rs)**2
             RR:        Planet-star radius ratio
             T14:       Transit duration
             I:         Inclination of the orbit

    If you add "UPPER" or "LOWER" next to the names defined above, you get the upper and lower 
    errors on those variables. For example, PERUPPER gives you the upper error on the period.
    """
    # Create folder where we will save the data:
    if not os.path.exists('planet_data'):
        os.mkdir('planet_data')
    # See if query has been done already for this planet:
    if os.path.exists('planet_data/'+pname+'.pkl'):
        fin = open('planet_data/'+pname+'.pkl','rb')
        data = pickle.load(fin)
        fin.close()
        return data
    def extract_data(fname):
        datadict = {}
        fin = open(fname,'r')
        while True:
            line = fin.readline()
            if line != '':
                if 'PRELOADED_DATA' in line:
                    data = line.split('"columns":{')[1]
                    data = data.split('},"numberOfRows')[0]
                    data = data.split(',')
                    for i in range(len(data)):
                            name,val = data[i].split('":[')
                            name = name[1:]
                            if val[0] == '"':
                                val = val[1:-2]
                            elif 'null' in val:
                                val = None
                            else:
                                val = np.double(val[:-1])
                            datadict[name] = val
                    break
            else:
                break
        fin.close()
        os.system('rm '+fname)
        return datadict
    name,number = pname.split('-')
    letter = number[-1:]
    number = number[:-1]
    # Get data from exoplanets.org:
    os.system('wget http://exoplanets.org/detail/'+name+'-'+number+'_'+letter)
    # Extract info from saved webpage:
    data = extract_data(name+'-'+number+'_'+letter)
    # Save query:
    fout = open('planet_data/'+pname+'.pkl','wb')
    pickle.dump(data,fout)
    fout.close()
    return data

def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def check_lcs(olc,clc):
    # First, check nans in target lightcurve:
    idx_not_nans_olc = np.where(~np.isnan(olc))[0]
    olc = olc[idx_not_nans_olc]
    clc = clc[idx_not_nans_olc,:]
    for j in range(clc.shape[1]):
        # Do the same for comparison stars:
        idx_not_nans = np.where(~np.isnan(clc[:,j]))[0]
        olc = olc[idx_not_nans]
        clc = clc[idx_not_nans,:]
    # Now that all nans are done, check zero counts on comparison stars. However, first
    # compute rms of the target lightcurve by the median-filter trick:
    #mfilt_olc = medfilt(olc/np.median(olc),21)
    #olc_sigma = get_sigma(mfilt_olc-olc/np.median(olc))
    idx = []
    for j in range(clc.shape[1]):
        idx_zero_counts = np.where(clc[:,j] == 0.0)[0]
        # If not all the counts are zero, save idx if and only if
        # the rms of the lightcurve is better than 5-sigma the one 
        # from the target (otherwise, you just introduce noise):
        if len(idx_zero_counts) != len(olc):
            idx.append(j)
    print '----'
    return olc,clc[:,idx]

def get_phases(t,P,t0):
    phase = ((t - np.median(t0))/np.median(P)) % 1
    ii = np.where(phase>=0.5)[0]
    phase[ii] = phase[ii]-1.0
    return phase

def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data,axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

def classic_PCA(Input_Data, standarize = True):
    """
    classic_PCA function

    Description

    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)

from sklearn import linear_model
def linear_fit(X,y):
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(X.transpose(),y)
    return regr.coef_
import matplotlib.pyplot as plt
def get_medfilt_outlier(data,window=21,nsigma = 5):
    mfilt = medfilt(data,window)
    plt.plot(data,'.')
    plt.plot(mfilt,'-')
    plt.show()
    residuals = data-mfilt
    sigma = get_sigma(residuals)
    idx = np.array(range(len(data)))
    print idx
    print window/2
    print len(idx)-window/2
    idx_good = np.where((np.abs(residuals)<nsigma*sigma)|((idx<window/2)|(idx>len(idx)-window/2)))[0]
    return idx_good

def simple_fit_lcs(times,olc,clc,pdata,Xin,plot=False,white_light=False, ld_law = 'quadratic'):
    target,comparisons = check_lcs(olc,clc)
    if len(comparisons.shape) == 2:
        super_comparison = np.sum(comparisons,axis=1)
    else:
        super_comparison = comparisons
    target = target/super_comparison
    idx = np.where((times<pdata['TT']-(((pdata['T14']/2.)+0.0025))) | \
                   (times>pdata['TT']+(((pdata['T14']/2.)+0.0025))))[0]
    X = np.vstack((np.ones(len(times)),Xin))
    coeffs = linear_fit(X[:,idx],target[idx])
    if plot:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        model = 0.0
        for i in range(X.shape[0]):
            model = model + coeffs[i]*X[i,:]
        plt.plot(times,target,'.')
        plt.plot(times[idx],target[idx],'ro')
        plt.plot(times,model)
        plt.show()
        plt.plot(times,target/model,'.-')
        plt.show()
    mcmc_results = fit_mcmc(times,target,X,coeffs,pdata,white_light=white_light, ld_law = ld_law)
    return mcmc_results

def fit_lcs(times,olc,clc,pdata,Xin=None,plot=False,white_light=False, ld_law = 'quadratic'):
    if plot:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
    # First, check lcs:
    target,comparisons = check_lcs(olc,clc)

    # Now, get SVD of the comparison stars:
    V,eigenvals,PC = classic_PCA(comparisons.transpose())
    if plot:
        plt.plot(eigenvals)
        plt.show()
        for i in range(comparisons.shape[1]):
            plt.plot(times,comparisons[:,i]/np.median(comparisons[:,i]),color='red')
        plt.plot(times,target/np.median(target),color='black')
        plt.show()

    # Get out-of-transit points:
    idx = np.where((times<pdata['TT']-(((pdata['T14']/2.)+0.005))) | \
                   (times>pdata['TT']+(((pdata['T14']/2.)+0.005))))[0]

    print idx
    # Select "best" model based on BIC of out-of-transit data:
    idx_pcs = range(1,PC.shape[0])
    all_idx = []
    all_coeffs = []
    all_bic = []
    n = np.double(len(idx))
    if Xin is not None:
        idx_x_ext = range(0,Xin.shape[0])
    else:
        idx_x_ext = [0]
    for i_pc in idx_pcs:
            c_PC = PC[:i_pc,idx]
            for i_x_ext in idx_x_ext:
                c_X = np.ones(len(idx))
                if i_x_ext != 0:
                    c_Xin = Xin[:i_x_ext,idx]
                    c_X = np.vstack((c_X,c_PC))
                    c_X = np.vstack((c_X,c_Xin))
                else:
                    c_X = np.vstack((c_X,c_PC))
                all_idx.append([i_pc,i_x_ext])
                coeffs = linear_fit(c_X,target[idx])
                all_coeffs.append(coeffs)
                model = 0.0
                for i in range(c_X.shape[0]):
                    model = model + coeffs[i]*c_X[i,:]
                rss = np.mean((model-target[idx])**2)
                all_bic.append(n*np.log(rss/n) + (1+i_pc + i_x_ext)*np.log(n))

    # Get the minimum BIC:
    min_bic = np.min(all_bic)
    # Check which of all the models has a BIC difference lower than 2, i.e., 
    # which models are indistinguishable from the data:
    idx_low_bics = np.where(np.abs(np.array(all_bic)-min_bic)<2)[0]
    # From all of them, define the "best" as the one with the lower complexity (modelled 
    # in this case by the number of parameters). If several models have the same complexity, 
    # then select the one which gives the lowest BIC amongst them:
    min_nparams = np.max(idx_pcs) + np.max(idx_x_ext)
    min_bic = np.max(all_bic)
    if len(idx_low_bics) == 1:
        min_nparams = all_idx[idx_low_bics[0]][0] + all_idx[idx_low_bics[0]][1]
        min_idx = idx_low_bics[0]
        min_bic = all_bic[idx_low_bics[0]]
    else:
        for i in idx_low_bics:
            nparams = all_idx[i][0] + all_idx[i][1]
            if nparams < min_nparams and all_bic[i]<min_bic:
                min_nparams = nparams
                min_idx = i
                min_bic = all_bic[i]
    print '\t Selected model has ',all_idx[min_idx][0],'PCs and ',all_idx[min_idx][1],'external parameters.'
    print '\t The BIC of the model is ',min_bic
    print '\t All BICs considered:'
    for i in range(len(all_bic)):
        print '\t + BIC:',all_bic[i],'npcs:',all_idx[i][0],'ext params:',all_idx[i][1]

    # Perform linear fit:
    X = np.ones(len(idx))
    Xfull = np.ones(len(times))
    X = np.vstack((X,PC[:all_idx[min_idx][0],idx]))
    Xfull = np.vstack((Xfull,PC[:all_idx[min_idx][0],:]))
    if all_idx[min_idx][1] != 0:
       X = np.vstack((X,Xin[:all_idx[min_idx][1],idx]))
       Xfull = np.vstack((Xfull,Xin[:all_idx[min_idx][1],:]))
    coeffs = linear_fit(X,target[idx])
    model = 0.0
    for i in range(Xfull.shape[0]):
        model = model + coeffs[i]*Xfull[i,:]
    #residuals = target-model
    #mfilt = medfilt(residuals,21)
    #sigma = get_sigma(residuals-mfilt)
    #idx_good = np.where(np.abs(residuals-mfilt)<3*sigma)[0]
    #times = times[idx_good]
    #target = target[idx_good]
    #model = model[idx_good]
    #Xfull = Xfull[:,idx_good]
    #idx = np.where((times<pdata['TT']-(((pdata['T14']/2.)+0.005))) | \
    #               (times>pdata['TT']+(((pdata['T14']/2.)+0.005))))[0]
    if plot:
        plt.plot((times-times[0])*24.,target,'.')
        plt.plot((times-times[0])*24.,model,'-')
        plt.show()
        plt.plot((times-times[0])*24.,target/model,'.-') 
        plt.plot(((times-times[0])*24.)[idx],(target/model)[idx],'o')
        plt.ylim([0.975,1.005])
        plt.show()

    # Now, perform MCMC. If white-lightcurve, fit everything. If not, fit LDs and Rp/R* only:
    mcmc_results = fit_mcmc(times,target,Xfull,coeffs,pdata,white_light=white_light, ld_law = ld_law)
    mcmc_results['npcs'] = all_idx[min_idx][0]
    return mcmc_results

def convert_ld_coeffs(ld_law, coeff1, coeff2):
    if ld_law == 'quadratic':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2.*(coeff1+coeff2))
    elif ld_law=='squareroot':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff2/(2.*(coeff1+coeff2))
    elif ld_law=='logarithmic':
        q1 = (1-coeff2)**2
        q2 = (1.-coeff1)/(1.-coeff2)
    return q1,q2

def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    return coeff1,coeff2

def init_batman(t,law):
    """
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

import emcee
import scipy.optimize as op
def fit_mcmc(times,target,X,coeffs,pdata,white_light = False,ld_law = 'quadratic',nwalkers = 500, nburnin = 1000,njumps = 1000):
    # Initialize useful parameters:
    nsystematic = X.shape[0]
    n_data_transit = X.shape[1]
    log2pi = np.log(2.*np.pi)
    # Initialize transit model:
    params,m = init_batman(times,law=ld_law)
    if pdata['ECC'] is None:
       pdata['ECC'] = 0.
    if pdata['OM'] is None:
       pdata['OM'] = 90.

    def lnlike_wl_transit(theta):
        # Extract transit + noise parameters:
        t0,a,p,inc,sigma_w,q1,q2 = theta[:7]
        # Extract coefficients of the systematics:
        sys_coeffs = theta[7:]
        # Build LC:
        coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.t0 = t0
        params.per = pdata['PER']
        params.rp = p
        params.a = a
        params.inc = inc
        params.ecc = pdata['ECC']
        params.w = pdata['OM']
        params.u = [coeff1,coeff2]
        lc_model = m.light_curve(params)
        # Build systematics model:
        sys_model = 0.0
        for i in range(nsystematic):
            sys_model = sys_model + sys_coeffs[i]*X[i,:]
        residuals = ((target/sys_model)-lc_model)*1e6
        tau = 1.0/sigma_w**2
        log_like = -0.5*(n_data_transit*log2pi+np.sum(np.log(1./tau)+tau*(residuals**2)))
        return log_like

    def lnlike_transit(theta):
        # Extract transit + noise parameters:
        p,sigma_w,q1,q2 = theta[:4]
        # Extract coefficients of the systematics:
        sys_coeffs = theta[4:]
        # Build LC:
        coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.t0 = pdata['TT']
        params.per = pdata['PER']
        params.rp = p
        params.a = pdata['AR']
        params.inc = pdata['I']
        params.ecc = pdata['ECC']
        params.w = pdata['OM']
        params.u = [coeff1,coeff2]
        lc_model = m.light_curve(params)
        # Build systematics model:
        sys_model = 0.0
        for i in range(nsystematic):
            sys_model = sys_model + sys_coeffs[i]*X[i,:]
        residuals = ((target/sys_model)-lc_model)*1e6
        tau = 1.0/sigma_w**2
        log_like = -0.5*(n_data_transit*log2pi+np.sum(np.log(1./tau)+tau*(residuals**2)))
        return log_like

    def normal_prior(val,mu,sigma):
        return np.log(1./np.sqrt(2.*np.pi*(sigma**2)))-\
                      0.5*(((mu-val)**2/(sigma**2)))

    def jeffreys_prior(val,lower_val,upper_val):
        return np.log(1) - np.log(val*np.log(upper_val/lower_val))

    def uniform_prior(val,a,b):
        return np.log(1./(b-a))

    def ln_wl_prior(theta):
        # Read in the values of the parameter vector and update values of the objects.
        # For each one, if everything is ok, get the total prior, which is the sum 
        # of the independant priors for each parameter:
        total_prior = 0.0
        # Extract transit + noise parameters:
        t0,a,p,inc,sigma_w,q1,q2 = theta[:7]
        # Extract coefficients of the systematics:
        sys_coeffs = theta[7:]
        # Check all vals are whitin the prior boudnaries:
        if sigma_w<10 or sigma_w>3000:
           return -np.inf
        # For q1 and q2:
        if q1<0 or q1>1:
           return -np.inf
        if q2<0 or q2>1:
           return -np.inf
        for i in range(nsystematic):
            if coeffs[i]>0:
                if sys_coeffs[i]<coeffs[i]*0.1 or sys_coeffs[i]>10.*coeffs[i]:
                    return -np.inf
            else:
                if sys_coeffs[i]>coeffs[i]*0.1 or sys_coeffs[i]<10.*coeffs[i]:
                    return -np.inf
        # If they are, generate priors for each planetary parameter. First, for t0:
        total_prior = total_prior + normal_prior(t0,pdata['TT'],0.1)
        # Now for a:
        total_prior = total_prior + normal_prior(a,pdata['AR'],np.max(pdata['ARUPPER'],pdata['ARLOWER']))
        # For p:
        total_prior = total_prior + normal_prior(p,np.sqrt(pdata['DEPTH']),0.01)
        # For inc:
        total_prior = total_prior + normal_prior(p,np.sqrt(pdata['I']),np.max(pdata['IUPPER'],pdata['ILOWER']))
        # For sigma_w
        total_prior = total_prior + jeffreys_prior(sigma_w,10.,3000.)
        # For q1 and q2:
        total_prior = total_prior + uniform_prior(q1,0.,1.)
        total_prior = total_prior + uniform_prior(q2,0.,1.)
        # And finally, for the coefficients of the systematics:
        for i in range(nsystematic):
            if coeffs[i]>0:
                total_prior = total_prior + uniform_prior(sys_coeffs[i],coeffs[i]*0.1,10.*coeffs[i])
            else:
                total_prior = total_prior + uniform_prior(sys_coeffs[i],coeffs[i]*10.,0.1*coeffs[i])
        return total_prior

    def ln_prior(theta):
        # Read in the values of the parameter vector and update values of the objects.
        # For each one, if everything is ok, get the total prior, which is the sum 
        # of the independant priors for each parameter:
        total_prior = 0.0
        # Extract transit + noise parameters:
        p,sigma_w,q1,q2 = theta[:4]
        # Extract coefficients of the systematics:
        sys_coeffs = theta[4:]
        # Check all vals are whitin the prior boudnaries:
        if sigma_w<10 or sigma_w>3000:
           return -np.inf
        # For q1 and q2:
        if q1<0 or q1>1:
           return -np.inf
        if q2<0 or q2>1:
           return -np.inf
        for i in range(nsystematic):
            if coeffs[i]>0:
                if sys_coeffs[i]<coeffs[i]*0.1 or sys_coeffs[i]>10.*coeffs[i]:
                    return -np.inf
            else:
                if sys_coeffs[i]>coeffs[i]*0.1 or sys_coeffs[i]<10.*coeffs[i]:
                    return -np.inf
        # For p:
        total_prior = total_prior + normal_prior(p,np.sqrt(pdata['DEPTH']),0.01)
        # For sigma_w
        total_prior = total_prior + jeffreys_prior(sigma_w,10.,3000.)
        # For q1 and q2:
        total_prior = total_prior + uniform_prior(q1,0.,1.)
        total_prior = total_prior + uniform_prior(q2,0.,1.)
        # And finally, for the coefficients of the systematics:
        for i in range(nsystematic):
            if coeffs[i]>0:
                total_prior = total_prior + uniform_prior(sys_coeffs[i],coeffs[i]*0.1,10.*coeffs[i])
            else:
                total_prior = total_prior + uniform_prior(sys_coeffs[i],coeffs[i]*10.,0.1*coeffs[i])
        return total_prior

    def lnprob_wl_transit(theta):
        lp = ln_wl_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_wl_transit(theta)       

    def lnprob_transit(theta):
        lp = ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_transit(theta)  

    # Check if white-light mode is on. If it is, fit t0,a,inc, etc. For each case, extract 
    # input theta
    if white_light:
        lnprob = lnprob_wl_transit
        theta_0 = [pdata['TT'],pdata['AR'],np.sqrt(pdata['DEPTH']),pdata['I'],500.,0.5,0.5]
        for i in range(len(coeffs)):
            theta_0.append(coeffs[i])
    else:
        lnprob = lnprob_transit
        theta_0 = [np.sqrt(pdata['DEPTH']),500.,0.5,0.5]
        for i in range(len(coeffs)):
            theta_0.append(coeffs[i])

    # Start at the maximum likelihood value:
    nll = lambda *args: -lnprob(*args)

    # Get ML estimate:
    result = op.minimize(nll, theta_0)
    theta_ml = result["x"]

    # Now define parameters for emcee:
    ndim = len(theta_ml)
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    # Run the MCMC:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, njumps+nburnin)

    # Save results:
    if white_light:
        out_dict = {}
        out_dict['ld_law'] = ld_law
        params = ['t0','a','p','inc','sigma_w','q1','q2']
        for i in range(len(coeffs)):
            params.append('coeffs'+str(i))
        for i in range(len(params)):
            out_dict[params[i]] = np.array([])
            for walker in range(nwalkers):
                out_dict[params[i]] = np.append(out_dict[params[i]],sampler.chain[walker,nburnin:,i])
        out_dict['X'] = X
        out_dict['times'] = times
        out_dict['target'] = target
        return out_dict
    else:
        out_dict = {}
        out_dict['ld_law'] = ld_law
        params = ['p','sigma_w','q1','q2']
        for i in range(len(coeffs)):
            params.append('coeffs'+str(i))
        for i in range(len(params)):
            out_dict[params[i]] = np.array([])
            for walker in range(nwalkers):
                out_dict[params[i]] = np.append(out_dict[params[i]],sampler.chain[walker,nburnin:,i])
        out_dict['X'] = X
        out_dict['times'] = times
        out_dict['target'] = target
        return out_dict

def get_sys_model(out_dict):
    model = 0.0
    for i in range(out_dict['X'].shape[0]):
        model = model + np.median(out_dict['coeffs'+str(i)])*out_dict['X'][i,:]
    return model

def get_lc_model(out_dict,pdata,interpolate=False):
    if interpolate:
        times = np.arange(np.min(out_dict['times']),np.max(out_dict['times']),0.001)
        params,m = init_batman(times,law=out_dict['ld_law'])
    else:
        params,m = init_batman(out_dict['times'],law=out_dict['ld_law'])
    coeff1,coeff2 = reverse_ld_coeffs(out_dict['ld_law'], np.median(out_dict['q1']), np.median(out_dict['q2']))
    params.t0 = pdata['TT']
    params.per = pdata['PER']
    params.rp = np.median(out_dict['p'])
    params.a = pdata['AR']
    params.inc = pdata['I']
    params.ecc = pdata['ECC']
    params.w = pdata['OM']
    params.u = [coeff1,coeff2]
    lc_model = m.light_curve(params)
    if interpolate:
        return times,lc_model
    return lc_model
