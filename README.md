# Probabilistic quotient's work \& pharmacokinetics' contribution: countering size effect in metabolic time series measurements

Creation and comparison of PQN, PKM, and MIX models for size effect normalization. 
A preprint of the manuscript is available on bioRxiv [![DOI:10.1101/2022.01.17.476591](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1101/2022.01.17.476591).

## Prerequisites
Python 3.7 and packages listed in requirements.txt.
```
pip install -r requirements.txt
```

## Synthetic Data Simulations
Classes of PKM and MIX model are located in ```synthetic_data/extended_model.py```.
Functions for synthetic data generation are located in ```synthetic_data/synthetic_data_generation.py```.
Functions for model initialization and optimization are located in ```synthetic_data/normalization.py```.

### Normalization Model Comparison
Scripts used to run simulatons v1-v3 (Figures 4-7, Supplementary Figures S8-S10) are are located in ```synthetic_data/run_simulation_v*.py```.
Results of the simulations are located in ```synthetic_data/simulation_results/*.pkl```.

To test a (shortened) simulation run you can execute ```synthetic_data/run_simulation_example.py```.
Results of test simulation is located in ```synthetic_data/simulation_results/example_simulation.pkl```.


### PQN on Noisy Data
An investigation on the performance of PQN on noisy data (Figure 8) is located in ```synthetic_data/Noisy_PQN.ipynb```.
Results of the simulation performed there is stored in ```synthetic_data/other_results/noisy_pqn.pkl```.

### Further Analysis
The script to test the difference of PQN and MIX_mini in performance on noisy data (Figure 9) is located in ```synthetic_data/comparison_with_noise.py```.
The results are located in ```synthetic_data/other_results/comparison_with_noise_results/*```.

The script to test the lambda hyperparameter (Supplementary Figure S3) is located in ```synthetic_data/search_lambda_v3.py```.
The results are located in ```synthetic_data/other_results/lambda_results/*```.

The script to test different loss function and transformation function combinations (Supplementary Figure S3) is located in ```synthetic_data/search_L_T_v3.py```.
The results are located in ```synthetic_data/other_results/L_T_results/*```.

An investigation of the influence on the weighting parameter lambda on the variance of fitted sweat volumes (Supplementary Figure S11) is done in ```synthetic_data/Lambda_Variance.ipynb```. 
Results of the simulations are stored in ```synthetic_data/other_results/lambda_variance*.pkl```.

## Real Finger Sweat Data Simulations
Jupyter Notebooks that run real data simulations are located at ```real_data/Brunmair_2021/PKM_Sub_2.ipynb``` and ```real_data/Brunmair_2021/MIX_Sub_2.ipynb``` for PKM and MIX respectively. 
The results of these simuations are located in ```real_data/Brunmair_2021/PKM_sub_2/*``` and ```real_data/Brunmair_2021/MIX_sub_2*``` for PKM_mininimal and MIX_minimal respectively.
The script for data preprocessing is located in ```real_data/Brunmair_2021/Preprocessing.ipynb```.

## Real Blood Plasma Data Simulations
The Jupyter Notebook that runs real blood plasma data simulations is located at ```real_data/Panitchpakdi_2021/DPH_Sub_2_models.ipynb```.
The results of these simulations are located in ```real_data/Panitchpakdi_2021/Sub_2_plasma.pkl```.
The script for data preprocessing is located in ```real_data/Panitchpakdi_2021/Preprocessing.ipynb```.

##  Figures from the Manuscript
A Jupyter Notebook that replicates all figures used in the manuscript is located at ```/Figures.ipynb```.

## Licensing
All original code is licensed under the GNU GPL version 3.