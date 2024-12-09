# PhyloCNN

This repository contains scripts, notebooks, and tools to perform simulations, encoding, model selection, parameter estimation, and posterior distribution analyses using PhyloCNN.

# Article
Perez M.F. and Gascuel O.PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies.

## **Installation**
To set up the required Python environment, use the following command:
```bash
conda env create -f environment.yml
conda activate phylocnn
```

## **Scripts and Notebooks**

### **Simulations**
1. **Birth-Death Model Simulations** (Python)
    - `generate_parameters.py`: Generate input parameters for BD, BDEI, and BDSS models.
    - Command Examples:
      ```bash
      python generate_parameters.py -m BD -r 1,5 -i 1,10 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BD.txt
      ```
   - [Simulators from (Voznica et al. 2022)](https://github.com/evolbioinfo/phylodeep/tree/main/simulators/bd_models): Simulate trees using BD, BDEI, or BDSS parameters.
    - Command Examples:
      ```bash
      python TreeGen_BD_refactored.py parameters_BD.txt <max_time=500> > BD_trees.nwk
      ```

2. **BiSSE Model Simulations** (R + Python)
    - Generate parameters with `generate_parameters.py` (Python).
    - Simulate trees with `BiSSE_simulator.R` (R).
    - Command Examples:
      ```bash
      python generate_parameters.py -m BISSE -l0 0.01,1.0 -t 0,1 -l1 0.1,1.0 -q 0.01,0.1 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BiSSE.txt
      Rscript BiSSE_simulator.R parameters_BiSSE.txt <indice=1> <seed_base=12345> <step=10> <nb_retrials=100> BiSSE_trees.nwk BiSSE_stats.txt BiSSE_params.txt
      ```

### **Encoding**
1. **Phylogenies Encoding** (Python)
    - `PhyloCNN_Encoding_PhyloDyn.py`: Encode BD, BDEI, and BDSS trees.
    - `PhyloCNN_Encoding_BiSSE.py`: Encode BiSSE trees.
    - Command Examples:
      ```bash
      python PhyloCNN_Encoding_PhyloDyn.py -t BD_trees.nwk -o Encoded_trees_BD.csv
      python PhyloCNN_Encoding_BiSSE.py -t BiSSE_trees.nwk -o Encoded_trees_BiSSE.csv
      ```

### **Preprocessing, Training, and Predictions**
1. **Preprocessing and Training** (Jupyter Notebooks)
    - `PhyloCNN_Train_PhyDyn_ModelSelection.ipynb`: Model selection for BD, BDEI, BDSS.
    - `PhyloCNN_Train_BD.ipynb`: Parameter estimation for BD model.
    - `PhyloCNN_Train_BDEI.ipynb`: Parameter estimation for BDEI model.
    - `PhyloCNN_Train_BDSS.ipynb`: Parameter estimation for BDSS model.
    - `PhyloCNN_Train_BiSSE.ipynb`: Parameter estimation for BiSSE model.

2. **Confidence Intervals and Posterior Distributions**:
    - `CI_HIV.ipynb`: Compute confidence intervals for HIV dataset.
    - `CI_primates.ipynb`: Compute confidence intervals for primates dataset.

### **Statistical Evaluation**
- PCA and Summary Statistics:
    - `BiSSE_SumStats.ipynb`: Extract summary statistics from trees simulated under BiSSE.
    - `PCA_HIV.ipynb`: Compute confidence intervals for HIV dataset.
    - `PCA_primates.ipynb`: Compute confidence intervals for primates dataset.

---
