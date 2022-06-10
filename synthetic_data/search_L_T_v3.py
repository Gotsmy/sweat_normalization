#!/home/users/mgotsmy/envs/robust_loss/bin/python

import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import pandas as pd
import sys
import extended_model as em
import synthetic_data_generation as sdg
import pickle
import time

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

def calculate_mix(m_tensor,sv_pqn,lb,ub,timepoints,n_metabolites,n_cpu,n_replicates,loss_name,lambda_,log):
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
    lambda_          Float. Weighting value for loss calculation.
    -
    Output:
    sweat_volumes    numpy.ndarray of shape (n_timepoints) with normalization factors.
    model            Optimized em.MIX_model.
    '''
    
    # creating model and setting properties
    model = em.MIX_model(timepoints,n_metabolites,scaler='standard',log=log)
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


if __name__ == "__main__":
    # define basic toy model parameters
    toy_parameters = np.array([[2,.1,1,0,.1],
                                [2,.1,2,0,.1],
                                [2,.1,3,0,.1],
                                [2,.1,.5,0,.1]])
    # define time points of toy model
    timepoints = np.linspace(0,15,20)
    # set seed of rng
    np.random.seed(13)

    ## BEGIN INPUT PARAMETERS ##
    bounds_per_metabolite  = [3,3,5,15,3]
    for error_sigma in [.2]:
        for n_metabolites in [10,4,60,20,40]:
            print('error sigma  ',error_sigma)
            print('n_metabolites',n_metabolites)
            n_replicates        = 100
            n_timepoints        = len(timepoints)
            n_known_metabolites = 4
            n_cpu               = 40
            n_mc_replicates     = 100
            full_lambda         = 1/(n_metabolites+1)
            mini_lambda         = 1/(n_known_metabolites+1)
    ## END INPUT PARAMETERS ##

            # sample sv and e
            sv_t_list, sv_v_list = sdg.generate_sweat_volumes(n_replicates,n_metabolites,n_timepoints)
            e_list               = sdg.generate_experimental_errors(n_replicates,n_metabolites,n_timepoints,error_sigma)
            
            results_time  = {'PQN':[],'MIX_abs_log':[],'MIX_max_log':[],'MIX_max':[],'MIX_abs':[]}
            results_sv    = {'PQN':[],'MIX_abs_log':[],'MIX_max_log':[],'MIX_max':[],'MIX_abs':[],'TRUE':[]}
            results_model = {'MIX_abs_log':[],'MIX_max_log':[],'MIX_max':[],'MIX_abs':[]}
            raw_values    = {'C':[],'SV':[],'M':[]}
            for n_replicate in range(n_replicates):
                print('n_replicate',n_replicate)
                
                ## DATA GENERATION
                sv_tensor = sv_t_list[n_replicate]
                sv_vector = sv_v_list[n_replicate]
                e_tensor  = e_list[n_replicate]
                # get concentration values
                c_tensor = sdg.generate_random_from_real_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite)
                # calculate M_tilde
                m_tensor     = c_tensor * sv_tensor * e_tensor
                # scaling can be done for a faster convergence. Since over all time points is the same this does not affect normalization.
                if n_metabolites != n_known_metabolites:
                    for i in range(4,n_metabolites):
                        m_tensor[i] = m_tensor[i]/np.max(m_tensor[i])
                        
                ## CREATE BOUNDS FOR THE MODEL
                # full model
                full_lb = np.concatenate((np.zeros(5*n_metabolites),np.ones(n_timepoints)*.05))
                full_ub = np.concatenate((bounds_per_metabolite*n_metabolites,np.ones(n_timepoints)*4))
                # the bounds for known parameteres are set to the true values + precision of the optimization function.
                for p in [2,3,4]:
                    full_lb[:n_known_metabolites*5][p::5] = toy_parameters[:,p]
                    full_ub[:n_known_metabolites*5][p::5] = toy_parameters[:,p]+10e-8
                # mini model
                mini_lb = np.concatenate((np.zeros(5*n_known_metabolites),np.ones(n_timepoints)*.05))
                mini_ub = np.concatenate((bounds_per_metabolite*n_known_metabolites,np.ones(n_timepoints)*4))
                # the bounds for known parameteres are set to the true values + precision of the optimization function.
                for p in [2,3,4]:
                    mini_lb[:n_known_metabolites*5][p::5] = toy_parameters[:,p]
                    mini_ub[:n_known_metabolites*5][p::5] = toy_parameters[:,p]+10e-8
                
                ## NORMALIZE FOR SWEAT VOLUME
                t1 = time.time()
                sv_pqn                      = calculate_pqn(m_tensor)
                t2 = time.time()
                loss_name                   = 'cauchy_loss'
                log                         = True
                sv_pqn                      = calculate_pqn(m_tensor)
                sv1, model1 = calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,mini_lambda,log)
                t3 = time.time()
                loss_name                   = 'max_cauchy_loss'
                log                         = True
                sv_pqn                      = calculate_pqn(m_tensor)
                sv2, model2 = calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,mini_lambda,log)
                t4 = time.time()
                loss_name                   = 'max_cauchy_loss'
                log                         = False
                sv_pqn                      = calculate_pqn(m_tensor)
                sv3, model3 = calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,mini_lambda,log)
                t5 = time.time()
                loss_name                   = 'cauchy_loss'
                log                         = False
                sv_pqn                      = calculate_pqn(m_tensor)
                sv4, model4 = calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,mini_lambda,log)
                t6 = time.time()
                
                results_time['PQN'].append(t2-t1)
                results_time['MIX_abs_log'].append(t3-t2)
                results_time['MIX_max_log'].append(t4-t3)
                results_time['MIX_max'    ].append(t5-t4)
                results_time['MIX_abs'    ].append(t6-t5)
                
                results_sv['PQN'].append(sv_pqn)
                results_sv['MIX_abs_log'].append(sv1)
                results_sv['MIX_max_log'].append(sv2)
                results_sv['MIX_max'    ].append(sv3)
                results_sv['MIX_abs'    ].append(sv4)
                results_sv['TRUE'].append(sv_vector)
                
                results_model['MIX_abs_log'].append(model1)
                results_model['MIX_max_log'].append(model2)
                results_model['MIX_max'    ].append(model3)
                results_model['MIX_abs'    ].append(model4)
            
                raw_values['C'].append(c_tensor)
                raw_values['SV'].append(sv_vector)
                raw_values['M'].append(m_tensor)
                
            # pickle results
            to_pickle = [results_time,results_sv,results_model,raw_values]
            with open(f'L_T_results/v3_e_{error_sigma}_n_{n_metabolites}.pkl','wb') as file:
                    pickle.dump(to_pickle,file)

    print('done')
