import numpy as np
import extended_model as em
import scipy.stats as ss
import pandas as pd
import random
import normalization as norm
# set seed of rng
np.random.seed(13)
random.seed(13)


def sample_sweat_volumes(nr):
    '''
    Sampling sweat volumes from a log-normal distribution. Mean and std are calculated from the sweat volumes estimated in https://doi.org/10.1038/s41467-021-26245-4.
    -
    Input
    nr        Int. Number of sweat volumes to sample.
    -
    Output
    sampels   List of sampled sweat volumes
    '''
    
    sv_distribution = ss.lognorm(s=0.492,loc=-0.521,scale=1.797)
    # sample new values
    samples = []
    for i in range(nr):
        sample = np.zeros(1)
        while sample < 0.05 or sample > 4:
            sample = sv_distribution.rvs()
        samples.append(sample)
    return samples

def generate_sweat_volumes(n_replicates,n_metabolites,n_timepoints):
    '''
    Samples sweat volumes and returns them as vector and tensor.
    -
    Input:
    n_replicates    Int. Number of synthetic data replicates.
    n_metabolites   Int. Number of measured metabolites.
    n_timepoints    Int. Number of measured time points.
    -
    Output
    sv_tensor       numpy.ndarray of shape (n_metabolites, n_timepoints) of sampled sweat volumes.
                    Sweat volumes are duplicated along the n_metabolites axis.
    sv_vector       numpy.ndarray of shape (n_timepoints) of sampled sweat volumes.
    '''
    
    sv_tensor = []
    sv_vector = []
    for i in range(n_replicates):
        tmp_sv_vector    = sample_sweat_volumes(n_timepoints)
        tmp_sv_tensor    = np.tile(tmp_sv_vector,n_metabolites).reshape(n_metabolites,n_timepoints)
        sv_tensor.append(tmp_sv_tensor)
        sv_vector.append(tmp_sv_vector)
    
    return np.array(sv_tensor), np.array(sv_vector)

def generate_experimental_errors(n_replicates,n_metabolites,n_timepoints,error_sigma):
    '''
    Samples sweat volumes and experimental error values and returns them a lists.
    -
    Input:
    n_replicates    Int. Number of synthetic data replicates.
    n_metabolites   Int. Number of measured metabolites.
    n_timepoints    Int. Number of measured time points.
    error_sigma     Float of relative error SD of the synthetic data.
    -
    Output
    e_tensor        Numpy.ndarray of shape (n_metabolites, n_timepoints).
    '''
    
    e_tensor    = []
    for i in range(n_replicates):
        tmp_e_tensor     = 0
        while np.min(tmp_e_tensor) <= 0:
            # In contrast to the sweat volume the error is not equal for every time point.
            tmp_e_tensor = np.random.normal(1,error_sigma,(n_metabolites,n_timepoints))
        e_tensor.append(tmp_e_tensor)
    return np.array(e_tensor)

def generate_random_kinetic_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite):
    '''
    Samples kinetic constants between defined bounds, appends them to the kinetic constants of the toy model 
    and calculates concentration time series of synthetic metabolites.
    -
    Input
    n_known_metabolites    Int. Number of metabolites in the toy model, in this study 4.
    n_metabolites          Int. Total number of metabolites for which measured data needs to be generated.
    toy_parameters         Numpy.ndarray of shape (5, n_known_parameters) with parameters of basic toy model.
    timepoints             Numpy.ndarray of shape (n_timepoints) of time points.
    bounds_per_metabolite  Numpy.ndarray of shape (5) of upper bounds of kinetic constants for one metabolite.
                           Kinetic parameters are sampled between 0 and bounds_per_metabolite for n_metabolites
                           - n_known_metabolites.
    - 
    Output
    c_tensor               Numpy.ndarray of shape (n_metabolites, n_timepoints)
    '''
    
    n_timepoints = len(timepoints)
    # sample new parameters
    new_parameters = np.swapaxes(np.random.uniform(np.zeros(5),
                                                   bounds_per_metabolite,
                                                   (n_metabolites-n_known_metabolites,5)
                                                  ),1,0)
    # reshape the toy parameters
    toy_parameters = np.swapaxes(toy_parameters,1,0)
    # append toy and sampled parameters
    tot_parameters = np.hstack([toy_parameters,new_parameters])
    # create time tensor for calculation of concentration time series
    time_tensor  = np.tile(timepoints,tot_parameters.shape[1]).reshape(tot_parameters.shape[1],-1)
    # create a parameter tensor for calculation of concentration time series
    tot_parameter_tensor = np.repeat([tot_parameters[0::5],
                                      tot_parameters[1::5],
                                      tot_parameters[2::5],
                                      tot_parameters[3::5],
                                      tot_parameters[4::5]],
                                     n_timepoints).reshape(-1,tot_parameters.shape[1],n_timepoints)
    # calculate concentration time series
    c_tensor = em.bateman(time_tensor,tot_parameter_tensor)
    return c_tensor

def generate_completely_random_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite):
    '''
    Samples a mean and standard deviation for each metabolite from a distribution following measured values and
    then uses the sampeled mean and SD to sample concentration values from a log-normal distribution. The
    sampled concentration time series are then concatenated to the toy model concentration time series.
    -
    Input
    n_known_metabolites    Int. Number of metabolites in the toy model, in this study 4.
    n_metabolites          Int. Total number of metabolites for which measured data needs to be generated.
    toy_parameters         Numpy.ndarray of shape (5, n_known_parameters) with parameters of basic toy model.
    timepoints             Numpy.ndarray of shape (n_timepoints) of time points.
    bounds_per_metabolite  Numpy.ndarray of shape (5) of upper bounds of kinetic constants for one metabolite.
                           Kinetic parameters are sampled between 0 and bounds_per_metabolite for n_metabolites
                           - n_known_metabolites. NOT USED. Just present for parity issues.
    - 
    Output
    c_tensor               Numpy.ndarray of shape (n_metabolites, n_timepoints)
    '''
    
    n_timepoints = len(timepoints)
    mean_distribution = ss.lognorm(s=1.820,loc=0.000143,scale=0.0875)
    std_distribution  = ss.lognorm(s=1.826,loc=0.000134,scale=0.0753)
    # sample random values - no kinetics
    random_c = []
    for i in range(n_metabolites-n_known_metabolites):
        tmp_mean = 0
        while tmp_mean <= 0:
            # sampling from a distribution that describes the 
            # means of all features in the finger sweat data set
            tmp_mean = mean_distribution.rvs()
        tmp_std = 0
        while tmp_std <= 0:
            # sampling from a distribution that describes the 
            # standard deviation of all features in the finger sweat data set
            tmp_std  = std_distribution.rvs()
        # scaling mu and s to their correspoing versions for the np.random.lognormal function
        mu  = 1
        s   = tmp_std
        mu_ = np.log(mu**2*np.sqrt(1/(s**2+mu)))
        s_  = np.sqrt(np.log(s**2+1))
        tmp_c = np.random.lognormal(mu_,s_,len(timepoints))*tmp_mean
        random_c.append(tmp_c)
    random_c_tensor = np.array(random_c)
    
    # reshape the toy parameters
    toy_parameters = np.swapaxes(toy_parameters,1,0)
    # append toy and sampled parameters
    tot_parameters = np.hstack([toy_parameters])
    # create time tensor for calculation of concentration time series
    time_tensor  = np.tile(timepoints,tot_parameters.shape[1]).reshape(tot_parameters.shape[1],-1)
    # create a parameter tensor for calculation of concentration time series
    tot_parameter_tensor = np.repeat([tot_parameters[0::5],
                                      tot_parameters[1::5],
                                      tot_parameters[2::5],
                                      tot_parameters[3::5],
                                      tot_parameters[4::5]],
                                     n_timepoints).reshape(-1,tot_parameters.shape[1],n_timepoints)
    # calculate concentration time series
    known_c_tensor = em.bateman(time_tensor,tot_parameter_tensor)
    
    # combine known kinetic and random number concentrations
    if n_known_metabolites != n_metabolites:
        c_tensor = np.vstack([known_c_tensor,random_c_tensor])
    else:
        c_tensor = known_c_tensor

    return c_tensor

def generate_random_from_real_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite):
    '''
    Samples a concentration time series from PQ normalized real finger sweat data. The
    sampled concentration time series are then concatenated to the toy model concentration time series.
    -
    Input
    n_known_metabolites    Int. Number of metabolites in the toy model, in this study 4.
    n_metabolites          Int. Total number of metabolites for which measured data needs to be generated.
    toy_parameters         Numpy.ndarray of shape (5, n_known_parameters) with parameters of basic toy model.
    timepoints             Numpy.ndarray of shape (n_timepoints) of time points.
    bounds_per_metabolite  Numpy.ndarray of shape (5) of upper bounds of kinetic constants for one metabolite.
                           Kinetic parameters are sampled between 0 and bounds_per_metabolite for n_metabolites
                           - n_known_metabolites. NOT USED. Just present for parity issues.
    - 
    Output
    c_tensor               Numpy.ndarray of shape (n_metabolites, n_timepoints)
    '''
    
    n_timepoints = len(timepoints)
    
    df = pd.read_csv('raw_data/imputed_untargeted.csv',index_col=0)
    # all donors with 20 timepoints
    donors = ['Donor_20','Donor_21','Donor_22','Donor_27','Donor_28','Donor_29','Donor_30','Donor_31','Donor_32','Donor_34','Donor_35','Donor_36','Donor_37','Donor_38','Donor_39','Donor_40','Donor_41','Donor_42','Donor_43','Donor_44','Donor_45','Donor_46','Donor_47']
    replicates = [1,2]
    tmp_donor = random.choice(donors)
    tmp_rep = random.choice(replicates)
    tmp = df[(df['donor']==tmp_donor)&(df['replicate']==tmp_rep)]
    pqn = norm.calculate_pqn(tmp.iloc[:,4:].values.T)
    norm_tmp = tmp.iloc[:,4:]/np.repeat([pqn],df.shape[1]-4,axis=0).T
    sampled = norm_tmp.loc[:,random.sample(list(norm_tmp.columns),n_metabolites-n_known_metabolites)]
    random_c_tensor = sampled.T.values
    
    # reshape the toy parameters
    toy_parameters = np.swapaxes(toy_parameters,1,0)
    # append toy and sampled parameters
    tot_parameters = np.hstack([toy_parameters])
    # create time tensor for calculation of concentration time series
    time_tensor  = np.tile(timepoints,tot_parameters.shape[1]).reshape(tot_parameters.shape[1],-1)
    # create a parameter tensor for calculation of concentration time series
    tot_parameter_tensor = np.repeat([tot_parameters[0::5],
                                      tot_parameters[1::5],
                                      tot_parameters[2::5],
                                      tot_parameters[3::5],
                                      tot_parameters[4::5]],
                                     n_timepoints).reshape(-1,tot_parameters.shape[1],n_timepoints)
    # calculate concentration time series
    known_c_tensor = em.bateman(time_tensor,tot_parameter_tensor)
    
    # combine known kinetic and random number concentrations
    if n_known_metabolites != n_metabolites:
        c_tensor = np.vstack([known_c_tensor,random_c_tensor])
    else:
        c_tensor = known_c_tensor

    return c_tensor