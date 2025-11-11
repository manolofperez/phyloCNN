# PhyloCNN

This repository contains scripts and notebooks to perform simulations, encoding, model selection, parameter estimation, and posterior distribution analyses using PhyloCNN.

# Article
Perez M.F. and Gascuel O.PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies.

## **Installation**
To set up the required Python environment (using the file `environment.yml`), use the following command:
```bash
conda env create -f environment.yml
conda activate phylocnn
```

## **Scripts and Notebooks**

### **Simulations**
1. **Phylodynamics Birth-Death Model Simulations** (Python)

    - `generate_parameters.py`: Generate input parameters for BD, BDEI, and BDSS models.
    - Command Examples:
      ```bash
      python generate_parameters.py -m BD_PhyDyn -r 1,5 -i 1,10 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BD.txt
      ```
      For BD model, where -m=model; -r=*R*<sub>0</sub>; -i=1/γ; -s=tree size; -p=sampling probability; -n=number of samples; -o: output file

      ```bash
      python generate_parameters.py -m BDEI -r 1,5 -i 1,10 -e 0.2,5 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BDEI.txt
      ```
      For BDEI model, where -m=model; -r=*R*<sub>0</sub>; -i=1/γ; -e=incubation factor (ε/γ); -s=tree size; -n=number of samples; -p=sampling probability; -o: output file
      
      ```bash
      python generate_parameters.py -m BDSS -r 1,5 -i 1,10 -x 3,10 -f 0.05,0.2 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BDSS.txt
      ```
      For BDSS model, where -m=model; -r=*R*<sub>0</sub>; -i=1/γ; -x=*X*<sub>SS</sub> ; -f=*f*<sub>SS</sub>; -s=tree size; -p=sampling probability; -n=number of samples; -o: output file


    - The output from `generate_parameters.py` should then be used with the [simulators from (Voznica et al. 2022)](https://github.com/evolbioinfo/phylodeep/tree/main/simulators/bd_models). 
    It requires the simulator to be called along with the parameter file generated in the previous step (e.g., parameters_BD.txt) and the maximum simulation time (with a default of 500; [Voznica et al., 2022](https://github.com/evolbioinfo/phylodeep/tree/main/simulators/bd_models)): Simulate trees using BD, BDEI, or BDSS parameters.
    - Command Examples:
      ```bash
      python TreeGen_BD_refactored.py parameters_BD.txt <max_time=500> > BD_trees.nwk
      ```

2. **Diversification Birth-Death Models Simulations** (R + Python)
    - `generate_parameters.py` (Python): Generate input parameters for BD, and BiSSE models.
    - Command Examples for BD:
      ```bash
      python generate_parameters.py -m BD_div -l0 0.01,1.0 -t 0,1 -l1 0.1,1.0 -q 0.01,0.1 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BD_div.txt
      ```
      For BD model, where -m=model; -l =λ; -t=τ; -s =tree size; -n = number of samples; -p = sampling probability; -o = output file

    - The output from `generate_parameters.py` should then be used with the [simulator from (Lambert et al. 2023)](https://github.com/JakubVoz/deeptimelearning/tree/main/simulators/BD). 
    It requires the simulator to be called along with the parameter file generated in the previous step (e.g., parameters_BD.txt) and the maximum simulation time (with a default of 500).
      ```bash
      python BD_simulator.py parameters_BD_div.txt <max_time=500> > BD_trees.nwk
      ```

      - Command Examples for BISSE:      
      ```bash
      python generate_parameters.py -m BISSE -l0 0.01,1.0 -t 0,1 -l1 0.1,1.0 -q 0.01,0.1 -s 200,500 -p 0.01,1 -n 10000 -o parameters_BiSSE.txt
      ```
      For BiSSE model, where -m=model; -l0 =λ<sub>0</sub>; -t=τ; -l1=ratio between λ<sub>1</sub> and λ<sub>0</sub>; -q=ratio between *q* (= *q*<sub>01</sub> = *q*<sub>10</sub>) and λ<sub>0</sub>; -s =tree size; -n = number of samples; -p = sampling probability; -o = output file

    - Use the output to simulate trees with `BiSSE_simulator.R` (R) [from (Lambert et al. 2023)](https://github.com/JakubVoz/deeptimelearning/tree/main/simulators/BiSSE).
    The values between <> are the ones we used for the parameters required by the script (indice, seed number, step, number of retrials, and output file names).
      ```
      Rscript BiSSE_simulator.R parameters_BiSSE.txt <indice=1> <seed_base=12345> <step=10> <nb_retrials=100> BiSSE_trees.nwk BiSSE_stats.txt BiSSE_params.txt
      ```

### **Encoding**
1. **Phylogenies Encoding** (Python)
    - `PhyloCNN_Encoding_PhyloDyn.py`: Encode BD, BDEI, BDSS and BD_div trees.
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

### **Model Adequacy**

1. **Posterior Sampling and Summary Statistics**
    - `SampleDistribution_kde.py`: Samples parameter values from the posterior distribution using gaussian Kernel Density Estimate (KDE).  
    - `BiSSE_SumStats.ipynb`: Extract summary statistics from trees simulated under BiSSE.
---
