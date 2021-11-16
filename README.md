# A Mixed Model for Size Effect Normalization in Biofluid Concentration Time Series Measurements

Creation and comparison of models for finger sweat normalization.

## Synthetic Data Simulations
Classes of EM and MIX model are located in ```synthetic_data/extended_model.py```.
Functions for synthetic data generation are located in ```synthetic_data/synthetic_data_generation.py```.
Functions for model initialization and optimization are located in ```synthetic_data/normalization.py```.

Scripts used to run simulatons v1-v3 are are located in ```synthetic_data/run_simulation_v*.py```.
To test a (shortened) simulation run you can execute ```synthetic_data/run_simulation_example.py```.

Results of the simulations are located in ```synthetic_data/simulation_results/*```.

A Jupyter Notebook that replicates Figures used in the manuscript is located at ```synthetic_data/Simulation_Analysis.ipynb```.

## Real Finger Sweat Data Simulations

Jupyter Notebooks that run real data simulations are located at ```real_data/EM_Sub_2.ipynb``` and ```real_data/MIX_Sub_2.ipynb```. 
The results of these simuations are located in ```real_data/EM_sub_2/*```, ```real_data/MIX_sub_2_n_EM```, and ```real_data/MIX_sub_2_n_PQN/*``` for EM_mininimal, equal weighting MIX_minimal, and metabolite-wise weighted MIX_minimal respectively.

A Jupyter Notebook that replicates Figures used in the manuscript is located at ```real_data/Real_Data_Analysis.ipynb```.