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

def calculate_em(m_tensor,lb,ub,timepoints,n_metabolites,n_cpu,n_replicates,loss_name,lambda_):
    '''
    Calculate normalization factor according to an EM model.
    -
    m_tensor         numpy.ndarray of shape (n_metabolites * n_timepoints) of measured data.
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
    model            Optimized em.extended_model.
    '''
    
    # creating model and setting properties
    model = em.extended_model(timepoints,n_metabolites)
    model.set_fit_bounds(lb,ub)
    model.set_measured_data(m_tensor.flatten('F'))
    model.set_loss_function(loss_name)
    # the weighting of error residuals is calculated over the sigma array parsed to the underlying scipy.optimize.curve_fit function.
    sigma_em = 1/lambda_
    model.set_sigma(np.ones(model.n_timepoints*model.n_metabolites)*sigma_em)
    # optimization of the model
    # out is a nd.array of shape ((n_parameters + 1) * n_replicates) with optimized parameters + loss value for every MC replicate.
    # as we are only interested in the best solution out is not used further.
    out = model.optimize_monte_carlo(n_replicates=n_replicates,n_cpu=n_cpu)
    return model.get_sweat_volumes(), model

def calculate_mix(m_tensor,sv_pqn,lb,ub,timepoints,n_metabolites,n_cpu,n_replicates,loss_name,lambda_):
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
    model            Optimized em.extended_model.
    '''
    
    # creating model and setting properties
    model = em.extended_mix_model(timepoints,n_metabolites)
    # as the MIX model has one additional parameter the bounds have to be appended.
    model.set_fit_bounds(np.concatenate([[.1],lb]),
                           np.concatenate([[10],ub]))
    model.set_measured_data(m_tensor.flatten('F'),sv_pqn)
    model.set_loss_function(loss_name)
    # the weighting of error residuals for EM and PQN part of the MIX model is calculated over the sigma array parsed to the underlying scipy.optimize.curve_fit function.
    sigma_em = 1/lambda_
    sigma_pqn= 1/(1-lambda_)
    model.set_sigma(np.append(np.ones(model.n_timepoints*model.n_metabolites)*sigma_em,np.ones(model.n_timepoints)*sigma_pqn))
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
        for n_metabolites in [6,60,10,20,40]:
            print('error sigma  ',error_sigma)
            print('n_metabolites',n_metabolites)
            n_replicates        = 2
            n_timepoints        = len(timepoints)
            n_known_metabolites = 4
            n_cpu               = 200
            n_mc_replicates     = 100
            loss_name           = 'max_cauchy_loss'
            # calculate lambda
            lambda_             = 1/(n_metabolites+1)
    ## END INPUT PARAMETERS ##

            # sample sv and e
            sv_t_list, sv_v_list = sdg.generate_sweat_volumes(n_replicates,n_metabolites,n_timepoints)
            e_list               = sdg.generate_experimental_errors(n_replicates,n_metabolites,n_timepoints,error_sigma)
            
            results_time  = {'PQN':[],'EM_full':[],'EM_mini':[],'MIX_full':[],'MIX_mini':[]}
            results_sv    = {'PQN':[],'EM_full':[],'EM_mini':[],'MIX_full':[],'MIX_mini':[],'TRUE':[]}
            results_model = {'EM_full':[],'EM_mini':[],'MIX_full':[],'MIX_mini':[]}
            raw_values    = {'C':[],'SV':[],'M':[]}
            for n_replicate in range(n_replicates):
                print('n_replicate',n_replicate)
                
                ## DATA GENERATION
                sv_tensor = sv_t_list[n_replicate]
                sv_vector = sv_v_list[n_replicate]
                e_tensor  = e_list[n_replicate]
                # get concentration values
                c_tensor = sdg.generate_completely_random_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite)
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
                sv_em_full, em_full_model   = calculate_em(m_tensor,
                                                           full_lb,full_ub,timepoints,n_metabolites,
                                                           n_cpu,n_mc_replicates,loss_name,lambda_)
                t3 = time.time()
                sv_mix_full, mix_full_model = calculate_mix(m_tensor,sv_pqn,
                                                            full_lb,full_ub,timepoints,n_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,lambda_)
                t4 = time.time()
                sv_em_mini, em_mini_model   = calculate_em(m_tensor[:4,:],
                                                           mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                           n_cpu,n_mc_replicates,loss_name,lambda_)
                t5 = time.time()
                sv_mix_mini, mix_mini_model = calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,lambda_)
                t6 = time.time()
                
                results_time['PQN'].append(t2-t1)
                results_time['EM_full'].append(t3-t2)
                results_time['MIX_full'].append(t4-t3)
                results_time['EM_mini'].append(t5-t4)
                results_time['MIX_mini'].append(t6-t5)
                
                results_sv['PQN'].append(sv_pqn)
                results_sv['EM_full'].append(sv_em_full)
                results_sv['MIX_full'].append(sv_mix_full)
                results_sv['EM_mini'].append(sv_em_mini)
                results_sv['MIX_mini'].append(sv_mix_mini)
                results_sv['TRUE'].append(sv_vector)
                
                results_model['EM_full'].append(em_full_model)
                results_model['MIX_full'].append(mix_full_model)
                results_model['EM_mini'].append(em_mini_model)
                results_model['MIX_mini'].append(mix_mini_model)
            
                raw_values['C'].append(c_tensor)
                raw_values['SV'].append(sv_vector)
                raw_values['M'].append(m_tensor)

            # pickle results
            to_pickle = [results_time,results_sv,results_model,raw_values]
            with open(f'simulation_results/v2_e_{error_sigma}_n_{n_metabolites}.pkl','wb') as file:
                    pickle.dump(to_pickle,file)

    print('done')
