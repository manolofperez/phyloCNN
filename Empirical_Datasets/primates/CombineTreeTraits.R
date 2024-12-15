phyUltra <- ape::read.tree(file = "./primatesUltra.tre")
traits_data <- read.csv("./data_traits.csv")
traits_data <- traits_data[-1]

traits_data_phylo <- traits_data[match(phyUltra$tip.label, traits_data$Species),]

# Replace state values: if state == 1, change to 2; if state == 0, change to 1
traits_data_phylo$state <- ifelse(traits_data_phylo$state == 1, 2, 
                                  ifelse(traits_data_phylo$state == 0, 1, traits_data_phylo$state))

phyUltra$tip.label <- paste(phyUltra$tip.label, 
                            "[&&NHX-t_s=",
                            traits_data_phylo$state,  # Replace "Trait" with the column name from your trait data
                            "]", sep="")

# Export the tree with the trait data to a Newick file
ape::write.tree(phyUltra, file = "./Primates-traits-ultrametric-tree.nwk")
