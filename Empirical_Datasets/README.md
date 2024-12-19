# PhyloCNN - Empirical data

This folder contains the two empirical phylogenies analyzed in newick format.

# Article
Perez M.F. and Gascuel O.PhyloCNN: Improving tree representation and neural network architecture for deep learning from trees in phylodynamics and diversification studies.

1. ## **PhyloCNN Inputs**

    - `ZurichHIV-tree.trees` : 200 taxa, from (Rasmussen et al. PLOS CB, 2017), analyzed in [(Voznica et al. Nature Com 2022)](https://github.com/evolbioinfo/phylodeep/tree/main/test_tree_HIV_Zurich).

    - `Primates-traits-ultrametric-tree.nwk` : 260 taxa, from (Fabre et al. Mol Phyl Evol 2009) and (Gomez and Verdu Syst Biol 2012), analyzed in (Lambert et al. Syst Biol 2023). This tree is ultrametric (dated) and the trait values are included in the taxon names using [&&NHX-t_s=1 or 2]; 1 stands for mutualistic (= 0 in the manuscript) and 2 for antagonistic (= 1 in the manuscript). This file can be obtained from a phylogeny and a file containing traits by using the script `CombineTreeTraits.R`.

2. ## **Script**

    - `CombineTreeTraits.R` : R-script that combined a phylogeny with a file containing a list of species and their respective traits.

3. ## **Primates input files**

    - `primatesUltra.tre` : Phylogenetic tree with 260 taxa from (Fabre et al. Mol Phyl Evol 2009). Available at [(Lambert et al. Syst Biol 2023)](https://github.com/JakubVoz/deeptimelearning/tree/main/data/empirical/primates_data).

    - `data_traits.csv` : Species list with their respective trait data. Available at [(Lambert et al. Syst Biol 2023)](https://github.com/JakubVoz/deeptimelearning/tree/main/data/empirical/primates_data).
