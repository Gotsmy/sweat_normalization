#!/home/users/mgotsmy/envs/robust_loss/bin/python

import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import pandas as pd
import sys
import extended_model as em
import synthetic_data_generation as sdg
import normalization as norm
import pickle
import time

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
                sv_pqn                      = norm.calculate_pqn(m_tensor)
                t2 = time.time()
                sv_em_full, em_full_model   = norm.calculate_em(m_tensor,
                                                           full_lb,full_ub,timepoints,n_metabolites,
                                                           n_cpu,n_mc_replicates,loss_name,lambda_)
                t3 = time.time()
                sv_mix_full, mix_full_model = norm.calculate_mix(m_tensor,sv_pqn,
                                                            full_lb,full_ub,timepoints,n_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,lambda_)
                t4 = time.time()
                sv_em_mini, em_mini_model   = norm.calculate_em(m_tensor[:4,:],
                                                           mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                           n_cpu,n_mc_replicates,loss_name,lambda_)
                t5 = time.time()
                sv_mix_mini, mix_mini_model = norm.calculate_mix(m_tensor[:4,:],sv_pqn,
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
            with open(f'simulation_results/v3_e_{error_sigma}_n_{n_metabolites}.pkl','wb') as file:
                    pickle.dump(to_pickle,file)

    print('done')
