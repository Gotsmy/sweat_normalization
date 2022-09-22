# Probabilistic quotient's work \& pharmacokinetics' contribution: countering size effect in metabolic time series measurements

Creation and comparison of PQN, PKM, and MIX models for size effect normalization.
For all details, read our research paper published at BMC Bioinformatics [![10.1186/s12859-022-04918-1](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1186/s12859-022-04918-1).

<p align="center">
    <img src="graphical_abstract.png" alt="graphical abstract" width="50%"/>
</p>

## 1. Size Effect Normalization Python Package

### 1.1 Prerequisites and Installation

The ```size_effect_normalization``` package requires Python 3.7 and packages listed in ```requirements.txt```. You can install the ```size_effect_normalization``` package via ```setup.py```.
```
pip install -r requirements.txt
python setup.py install
```

### 1.2 Testing Installation
To test if the installation was successful you can execute a (shortened) simulation run.
```
python tests/run_simulation_example.py
```

### 1.3 Tutorial
A tutorial for PKM<sub>minimal</sub> and MIX<sub>minimal</sub> normalization is available as Jupyter Notebook in ```docs/source/data/Tutorial.ipynb```.

### 1.4 Documentation
A "Read the Docs" style documentation of the ```size_effect_normalization``` package can be accessed <a href="https://homepage.univie.ac.at/mathias.gotsmy/size_effect_normalization/index.html" title="documentation link">here</a>.

## 2. Synthetic Data Simulations

### 2.1 Normalization Model Comparison
Scripts used to run simulatons v1-v3 (Figures 4-7, Supplementary Figures S8-S10) are are located in ```synthetic_data/run_simulation_v*.py```.
Results of the simulations are located in ```synthetic_data/simulation_results/*.pkl```.

### 2.2 PQN on Noisy Data
An investigation on the performance of PQN on noisy data (Figure 8) is located in ```synthetic_data/Noisy_PQN.ipynb```.
Results of the simulation performed there is stored in ```synthetic_data/other_results/noisy_pqn.pkl```.

### 2.3 Further Analysis
The script to test the difference of PQN and MIX<sub>minimal</sub> in performance on noisy data (Figure 9) is located in ```synthetic_data/comparison_with_noise.py```.
The results are located in ```synthetic_data/other_results/comparison_with_noise_results/*```.

The script to test the lambda hyperparameter (Supplementary Figure S3) is located in ```synthetic_data/search_lambda_v3.py```.
The results are located in ```synthetic_data/other_results/lambda_results/*```.

The script to test different loss function and transformation function combinations (Supplementary Figure S3) is located in ```synthetic_data/search_L_T_v3.py```.
The results are located in ```synthetic_data/other_results/L_T_results/*```.

An investigation of the influence on the weighting parameter lambda on the variance of fitted sweat volumes (Supplementary Figure S11) is done in ```synthetic_data/Lambda_Variance.ipynb```.
Results of the simulations are stored in ```synthetic_data/other_results/lambda_variance*.pkl```.

## 3. Real Data Simulations

### 3.1 Finger Sweat
Jupyter Notebooks that run real data simulations are located at ```real_data/Brunmair_2021/PKM_Sub_2.ipynb``` and ```real_data/Brunmair_2021/MIX_Sub_2.ipynb``` for PKM and MIX respectively.
The results of these simuations are located in ```real_data/Brunmair_2021/PKM_sub_2/*``` and ```real_data/Brunmair_2021/MIX_sub_2/*``` for PKM<sub>minimal</sub> and MIX<sub>minimal</sub> respectively.
The script for data preprocessing is located in ```real_data/Brunmair_2021/Preprocessing.ipynb```.

### 3.2 Blood Plasma
The Jupyter Notebook that runs real blood plasma data simulations is located at ```real_data/Panitchpakdi_2021/DPH_Sub_2_models.ipynb```.
The results of these simulations are located in ```real_data/Panitchpakdi_2021/Sub_2_plasma.pkl```.
The script for data preprocessing is located in ```real_data/Panitchpakdi_2021/Preprocessing.ipynb```.

##  5. Figures from the Manuscript
A Jupyter Notebook that replicates all figures used in the manuscript is located at ```/Figures.ipynb```.

## 6. Licensing
All original code is licensed under the GNU GPL version 3.
