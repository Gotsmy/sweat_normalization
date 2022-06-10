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
        for noise_fraction in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]:
            n_metabolites = 60
            print('error sigma  ',error_sigma)
            print('f_n',noise_fraction)
            n_replicates        = 100
            n_timepoints        = len(timepoints)
            n_known_metabolites = 4
            n_cpu               = 100
            n_mc_replicates     = 100
            full_lambda         = 1/(n_metabolites+1)
            mini_lambda         = 1/(n_known_metabolites+1)
    ## END INPUT PARAMETERS ##

            # sample sv and e
            sv_t_list, sv_v_list = sdg.generate_sweat_volumes(n_replicates,n_metabolites,n_timepoints)
            e_list               = sdg.generate_experimental_errors(n_replicates,n_metabolites,n_timepoints,error_sigma)
            
            results_time  = {'PQN':[],'PKM_full':[],'PKM_mini':[],'MIX_full':[],'MIX_mini':[]}
            results_sv    = {'PQN':[],'PKM_full':[],'PKM_mini':[],'MIX_full':[],'MIX_mini':[],'TRUE':[]}
            results_model = {'PKM_full':[],'PKM_mini':[],'MIX_full':[],'MIX_mini':[]}
            raw_values    = {'C':[],'SV':[],'M':[]}
            for n_replicate in range(n_replicates):
                print('n_replicate',n_replicate)
                
                ## DATA GENERATION
                sv_tensor = sv_t_list[n_replicate]
                sv_vector = sv_v_list[n_replicate]
                e_tensor  = e_list[n_replicate]
                # get concentration values
                c_tensor = sdg.generate_random_from_real_data(n_known_metabolites,n_metabolites,toy_parameters,timepoints,bounds_per_metabolite)
                # calculate how many metabolites are noisy:
                n_noisy_metabolites = round(n_metabolites*noise_fraction)
                n_real_metabolites = n_metabolites - n_noisy_metabolites
                # calculate M_tilde
                m_tensor_real = c_tensor[:n_real_metabolites,:] * sv_tensor[:n_real_metabolites,:] * e_tensor[:n_real_metabolites,:]
                m_tensor_noisy= c_tensor[n_real_metabolites:,:] * e_tensor[n_real_metabolites:,:]
                m_tensor = np.vstack([m_tensor_real,m_tensor_noisy])
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
                loss_name = 'max_cauchy_loss' 
                sv_pkm_full, pkm_full_model = sv_pqn, None # not used!
                t3 = time.time()
                loss_name = 'cauchy_loss'
                sv_pqn                      = norm.calculate_pqn(m_tensor)
                sv_mix_full, mix_full_model = sv_pqn, None # not used!
                t4 = time.time()
                loss_name = 'max_cauchy_loss'
                sv_pkm_mini, pkm_mini_model = sv_pqn, None # not used!
                t5 = time.time()
                loss_name = 'cauchy_loss'
                sv_pqn                      = norm.calculate_pqn(m_tensor)
                sv_mix_mini, mix_mini_model = norm.calculate_mix(m_tensor[:4,:],sv_pqn,
                                                            mini_lb,mini_ub,timepoints,n_known_metabolites,
                                                            n_cpu,n_mc_replicates,loss_name,'log10',
                                                            'standard',mini_lambda)
                t6 = time.time()
                
                results_time['PQN'].append(t2-t1)
                results_time['PKM_full'].append(t3-t2)
                results_time['MIX_full'].append(t4-t3)
                results_time['PKM_mini'].append(t5-t4)
                results_time['MIX_mini'].append(t6-t5)
                
                results_sv['PQN'].append(sv_pqn)
                results_sv['PKM_full'].append(sv_pkm_full)
                results_sv['MIX_full'].append(sv_mix_full)
                results_sv['PKM_mini'].append(sv_pkm_mini)
                results_sv['MIX_mini'].append(sv_mix_mini)
                results_sv['TRUE'].append(sv_vector)
                
                results_model['PKM_full'].append(pkm_full_model)
                results_model['MIX_full'].append(mix_full_model)
                results_model['PKM_mini'].append(pkm_mini_model)
                results_model['MIX_mini'].append(mix_mini_model)
            
                raw_values['C'].append(c_tensor)
                raw_values['SV'].append(sv_vector)
                raw_values['M'].append(m_tensor)
                
            # pickle results
            to_pickle = [results_time,results_sv,results_model,raw_values]
            with open(f'comparison_with_noise_results/v3_fn_{noise_fraction}.pkl','wb') as file:
                    pickle.dump(to_pickle,file)

    print('done')
