import numpy as np
import extended_model as em

def calculate_pqn(m_tensor):
    '''
    Calculate normalization factor according to PQN as described by Filzmoser et al., 2014.
    -
    Input:
    m_tensor     Numpy array of shape (n_metabolites * n_timepoints) of measured data.
    -
    Output:
    pqn          Numpy array of shape (n_timepoints) with normalization factors.
    '''
    
    # implemented according to http://dx.doi.org/10.1016/j.chroma.2014.08.050
    ref = np.median(m_tensor,axis=1) 
    div = np.divide(m_tensor,ref[:,None])
    pqn = np.median(div,axis=0)
    return pqn

def calculate_pkm(m_tensor,lb,ub,timepoints,n_metabolites,n_cpu,n_replicates,loss_name,trans_fun,lambda_):
    '''
    Calculate normalization factor according to an PKM model.
    -
    m_tensor         numpy.ndarray of shape (n_metabolites * n_timepoints) of measured data.
    lb               numpy.ndarray of shape (n_metabolites * 5 + n_timepoints) of lower bounds of parameters for model fitting.
    ub               numpy.ndarray of shape (n_metabolites * 5 + n_timepoints) of upper bounds of parameters for model fitting.
    timepoints       numpy.ndarray of shape (n_timepoints) of time points.
    n_metabolites    Int. Number of metabolites to fit. 
    n_cpu            Int. Number of CPUs used in multiprocessing.
    n_replicates     Int. Number of Monte Carlo replicates used for optimization.
    loss_name        Str. Loss used for optimization.
    trans_fun        Str. Trans_fun argument of em.PKM_model
    lambda_          Float. Weighting value for loss calculation.
    -
    Output:
    sweat_volumes    numpy.ndarray of shape (n_timepoints) with normalization factors.
    model            Optimized em.PKM_model.
    '''
    
    # creating model and setting properties
    model = em.PKM_model(timepoints,n_metabolites,pkm_fun='bateman',trans_fun=trans_fun)
    model.set_fit_bounds(lb,ub)
    model.set_measured_data(m_tensor.flatten('F'))
    model.set_loss_function(loss_name)
    # the weighting of error residuals is calculated over the sigma array parsed to the underlying scipy.optimize.curve_fit function.
    sigma_pkm = 1/lambda_
    model.set_sigma(np.ones(model.n_timepoints*model.n_metabolites)*sigma_pkm)
    # optimization of the model
    # out is a nd.array of shape ((n_parameters + 1) * n_replicates) with optimized parameters + loss value for every MC replicate.
    # as we are only interested in the best solution out is not used further.
    out = model.optimize_monte_carlo(n_replicates=n_replicates,n_cpu=n_cpu)
    return model.get_sweat_volumes(), model

def calculate_mix(m_tensor,sv_pqn,lb,ub,timepoints,n_metabolites,n_cpu,n_replicates,loss_name,trans_fun,scale_fun,lambda_):
    '''
    Calculate normalization factor according to a MIX model.
    -
    m_tensor         numpy.ndarray of shape (n_metabolites * n_timepoints) of measured data.
    sv_pqn           numpy.ndarray of shape (n_timepoints) of normalization factors as calculated from PQN.
    lb               numpy.ndarray of shape (n_metabolites * 5 + n_timepoints) of lower bounds of parameters for model fitting.
    ub               numpy.ndarray of shape (n_metabolites * 5 + n_timepoints) of upper bounds of parameters for model fitting.
    timepoints       numpy.ndarray of shape (n_timepoints) of time points.
    n_metabolites    Int. Number of metabolites to fit. 
    n_cpu            Int. Number of CPUs used in multiprocessing.
    n_replicates     Int. Number of Monte Carlo replicates used for optimization.
    loss_name        Str. Loss used for optimization.
    trans_fun        Str. Trans_fun argument of em.MIX_model
    scale_fun        Str. Scale_fun argument of em.MIX_model
    lambda_          Float. Weighting value for loss calculation.
    -
    Output:
    sweat_volumes    numpy.ndarray of shape (n_timepoints) with normalization factors.
    model            Optimized em.MIX_model.
    '''
    
    # creating model and setting properties
    model = em.MIX_model(timepoints,n_metabolites,scale_fun=scale_fun,pkm_fun='bateman',trans_fun=trans_fun)
    # as the MIX model has one additional parameter the bounds have to be appended.
    model.set_fit_bounds(lb,ub)
    model.set_measured_data(m_tensor.flatten('F'),sv_pqn)
    model.set_loss_function(loss_name)
    # the weighting of error residuals for EM and PQN part of the MIX model is calculated over the sigma array parsed to the underlying scipy.optimize.curve_fit function.
    sigma_pkm = 1/lambda_
    sigma_pqn= 1/(1-lambda_)
    model.set_sigma(np.append(np.ones(model.n_timepoints*model.n_metabolites)*sigma_pkm,np.ones(model.n_timepoints)*sigma_pqn))
    # optimization of the model
    # out is a nd.array of shape ((n_parameters + 1) * n_replicates) with optimized parameters + loss value for every MC replicate.
    # as we are only interested in the best solution out is not used further.
    out = model.optimize_monte_carlo(n_replicates=n_replicates,n_cpu=n_cpu)
    return model.get_sweat_volumes(), model