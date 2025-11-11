# PhyloCNN - Trained Models

This repository contains the trained neural network models obtained with PhyloCNN.

# Article
Perez M.F. and Gascuel O. 2025. PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies. Systematic Biology.


## **Trained Models (.json is contains the model and .h5 contains the weights)**

### **Phylodynamics Simulated data**
- Model Classification:
    - `Trained_2Generation_PhyDyn.json` and `Trained_2Generation_PhyDyn.h5`: Trained model to classify phylodynamics models using 2-Generation context PhyloCNN.
    - `Trained_1Generation_PhyDyn.json` and `Trained_1Generation_PhyDyn.h5`: Trained model to classify phylodynamics model using 1-Generation context PhyloCNN.
    - `Trained_NoContext_PhyDyn.json` and `Trained_NoContext_PhyDyn.h5`: Trained model to classify phylodynamics model using No_context PhyloCNN.

- BD:
    - `Trained_2Generation_BD.json` and `Trained_2Generation_BD.h5`: Trained model to estimate parameters under BD using 2-Generation context PhyloCNN.
    - `Trained_1Generation_BD.json` and `Trained_1Generation_BD.h5`: Trained model to estimate parameters under BD using 1-Generation context PhyloCNN.
    - `Trained_NoContext_BD.json` and `Trained_NoContext_BD.h5`: Trained model to estimate parameters under BD using No_context PhyloCNN.

- BDSS:
    - `Trained_2Generation_BDEI.json` and `Trained_2Generation_BDEI.h5`: Trained model to estimate parameters under BDEI using 2-Generation context PhyloCNN.
    - `Trained_1Generation_BDEI.json` and `Trained_1Generation_BDEI.h5`: Trained model to estimate parameters under BDEI using 1-Generation context PhyloCNN.
    - `Trained_NoContext_BDEI.json` and `Trained_NoContext_BDEI.h5`: Trained model to estimate parameters under BDEI using No_context PhyloCNN.

- BDSS:
    - `Trained_2Generation_BDSS.json` and `Trained_2Generation_BDSS.h5`: Trained model to estimate parameters under BDSS using 2-Generation context PhyloCNN.
    - `Trained_1Generation_BDSS.json` and `Trained_1Generation_BDSS.h5`: Trained model to estimate parameters under BDSS using 1-Generation context PhyloCNN.
    - `Trained_NoContext_BDSS.json` and `Trained_NoContext_BDSS.h5`: Trained model to estimate parameters under BDSS using No_context PhyloCNN.

### **Phylodynamics HIV dataset**
- Model Classification:
    - `Trained_2Generation_PhyDyn_HIV.json` and `Trained_2Generation_PhyDyn_HIV.h5`: Trained model to classify phylodynamics models using 2-Generation context PhyloCNN using simulations with the HIV priors.

- BDSS:
    - `Trained_2Generation_BDSS_HIV.json` and `Trained_2Generation_BDSS_HIV.h5`: Trained model to estimate parameters under BDSS using 2-Generation context PhyloCNN using simulations with the HIV priors.

### **BiSSE Simulated data (also used on the empirical primates dataset)**

- BiSSE:
    - `Trained_2Generation_BiSSE.json` and `Trained_2Generation_BiSSE.h5`: Trained model to estimate parameters under BiSSE using 2-Generation context PhyloCNN.
    - `Trained_1Generation_BiSSE.json` and `Trained_1Generation_BiSSE.h5`: Trained model to estimate parameters under BiSSE using 1-Generation context PhyloCNN.
    - `Trained_NoContext_BiSSE.json` and `Trained_NoContext_BiSSE.h5`: Trained model to estimate parameters under BiSSE using No_context PhyloCNN.

---
