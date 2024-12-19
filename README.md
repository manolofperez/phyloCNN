# PhyloCNN - Model Adequacy

This repository contains notebooks to perform sanity checks (model adequacy) on prior and posterior results used with PhyloCNN.

# Article
Perez M.F. and Gascuel O.PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies.


## **Scripts and Notebooks**

### **Posterior Sampling and Summary Statistics**
  - `SampleDistribution_kde.py`: Samples parameter values from the posterior distribution using gaussian Kernel Density Estimate (KDE).  
  - `BiSSE_SumStats.ipynb`: Extract summary statistics (SumStats) from trees simulated under BiSSE.

### **PCA**
  - `PCA_HIV.ipynb`: Compute confidence intervals for HIV dataset.
  - `PCA_primates.ipynb`: Compute confidence intervals for primates dataset.

### **Results**
1. **HIV Dataset**
For the 10,000 simulations obtained using the prior and postrior distributions of parameters, we add the number of SumStats rejected, for which the minimum and maximum values obtained from the 10K new simulations generated from the posterior do not contain the value observed in the empirical HIV dataset. We also provide the names of all rejected SumStats (according to [Saulnier et al., 2017](https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1005416.t003)). It is important to note that all rejected SumStats are related to the LTT plot, which is expected due to non-random sampling and partner notification in the HIV dataset ([Zhukova and Gascuel, 2024](https://www.medrxiv.org/content/10.1101/2024.09.09.24313296v1)).

    | BDSS (HIV priors)                       | # SumStats outside Min-Max | SumStats names (according to [Saulnier et al., 2017](https://journals.plos.org/ploscompbiol/article/figure/image?size=large&id=10.1371/journal.pcbi.1005416.t003))<sup>1</sup>         |
    |--------------------------------------|----------------------------|------------------------------------------------------------|
    | Prior distribution       | 4                          | y_5-8                                                       |
    | PhyloCNN Posterior     | 18                         | max_L, mean_s_time, x_12-18, y_2-10                        |
    
    <sup>1</sup>max_L: maximal number of lineages; mean_s_time: mean time between two consecutive down steps (mean sampling time); x_1-20: x coordinates of the LTT plot; y_1-20: x coordinates of the LTT time plot.

2. **Primates Dataset**
    All SumStats of the empirical dataset where recovered within the interval of SumStats obtained from the 10,000 simulations (Prior and Posterior).

---
