# Probabilistic quotient's work \& pharmacokinetics' contribution: countering size effect in metabolic time series measurements

Creation and comparison of models for finger sweat normalization. A preprint of the manuscript is available on bioRxiv [![DOI:10.1101/2022.01.17.476591](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1101/2022.01.17.476591).

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
Scripts used to run simulatons v1-v3 are are located in ```synthetic_data/run_simulation_v*.py```.
Results of the simulations are located in ```synthetic_data/simulation_results/*.pkl```.

To test a (shortened) simulation run you can execute ```synthetic_data/run_simulation_example.py```.
Results of test simulation is located in ```synthetic_data/example_simulation.pkl```.

### PQN on Noisy Data
An investigation on the performance of PQN on noisy data is located in ```synthetic_data/Noisy_PQN.ipynb```.
Results of the simulation performed there is stored in ```synthetic_data/noisy_pqn.pkl```.

## Real Finger Sweat Data Simulations

Jupyter Notebooks that run real data simulations are located at ```real_data/PKM_Sub_2.ipynb``` and ```real_data/MIX_Sub_2.ipynb``` for PKM and MIX respectively. 
The results of these simuations are located in ```real_data/PKM_sub_2/*``` and ```real_data/MIX_sub_2``` for PKM_mininimal and MIX_minimal respectively.

##  Figures from the Manuscript
A Jupyter Notebook that replicates Figures used in the manuscript is located in the base (```Figures.ipynb```).
