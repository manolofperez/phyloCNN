# PhyloCNN - Model Adequacy

This repository contains notebooks to perform sanity checks (model adequacy) on prior and posterior results used with PhyloCNN.

# Article
Perez M.F. and Gascuel O.PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies.


## **Scripts and Notebooks**

### **Statistical Evaluation**
- PCA and Summary Statistics:
    - `BiSSE_SumStats.ipynb`: Extract summary statistics from trees simulated under BiSSE.
    - `PCA_HIV.ipynb`: Compute confidence intervals for HIV dataset.
    - `PCA_primates.ipynb`: Compute confidence intervals for primates dataset.

### **Results**
1. **HIV Dataset** (Jupyter Notebooks)
    | BDSS (HIV priors)                       | # SumStats outside Min-Max | SumStats names (according to [Saulnier et al., 2017](https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1005416.t003))<sup>1</sup>         |
    |--------------------------------------|----------------------------|------------------------------------------------------------|
    | Prior distribution       | 4                          | y_5-8                                                       |
    | PhyloCNN Posterior     | 18                         | max_L, mean_s_time, x_12-18, y_2-10                        |
    
    <sup>1</sup>max_L: maximal number of lineages; mean_s_time: mean time between two consecutive down steps (mean sampling time); x_1-20: x coordinates of the LTT plot; y_1-20: x coordinates of the LTT time plot.

2. **Primates Dataset** (Jupyter Notebooks)

---
