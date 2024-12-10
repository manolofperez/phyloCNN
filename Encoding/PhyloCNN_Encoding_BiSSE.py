#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import time
import argparse
from ete3 import Tree
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set a high recursion limit to handle large trees
sys.setrecursionlimit(100000)

# Constants
TARGET_AVG_BL = 1  # Target average branch length for rescaling

def assign_t_s_attributes(tree):
    """
    Assigns 't_s' attribute to each tip based on the tip's name.
    The 't_s' attribute represents the state (1 or 2) extracted from the tip name.
    """
    for leaf in tree.iter_leaves():
        if "&&NHX-t_s=1" in leaf.name:
            leaf.add_feature("t_s", 1)
        elif "&&NHX-t_s=2" in leaf.name:
            leaf.add_feature("t_s", 2)
        else:
            leaf.add_feature("t_s", 0)  # Default state if not specified

def add_t_s_to_nodes(tree):
    """
    Calculates the number of descendant tips with 't_s' equal to 1 for each node.
    Assigns this value to the node's 't_s' attribute.
    """
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.t_s = getattr(node, 't_s', 0)
        else:
            node.t_s = sum(child.t_s for child in node.children)

def add_number_of_leaves(tree):
    """
    Adds the 'leaves' attribute to each node, representing the number of descendant leaves.
    """
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.leaves = 1
        else:
            node.leaves = sum(child.leaves for child in node.children)

def add_dist_to_root(tree):
    """
    Adds 'dist_to_root' attribute to each node, representing the distance from the root to the node.
    """
    for node in tree.traverse("preorder"):
        if node.is_root():
            node.dist_to_root = 0
        else:
            node.dist_to_root = node.up.dist_to_root + node.dist

def add_dist_to_present(tree):
    """
    Adds 'dist_to_present' attribute to each node.
    For leaves, sets 'dist_to_present' to 0.
    For internal nodes, computes the distance from the node to the deepest tip.
    """
    max_depth = max(leaf.dist_to_root for leaf in tree.iter_leaves())
    for node in tree.traverse():
        if node.is_leaf():
            node.dist_to_present = 0
        else:
            node.dist_to_present = max_depth - node.dist_to_root

def add_distances_to_ancestors(tree):
    """
    Adds distances to ancestor, grandparent, and great-grandparent for each node.
    """
    for node in tree.traverse("levelorder"):
        # Initialize ancestor attributes
        node.dist_to_anc = node.dist_to_grand_anc = node.dist_to_great_grand_anc = -1

        # Ancestor (parent)
        if node.up:
            node.dist_to_anc = node.dist

            # Grandparent
            if node.up.up:
                node.dist_to_grand_anc = node.dist_to_root - node.up.up.dist_to_root

                # Great-grandparent
                if node.up.up.up:
                    node.dist_to_great_grand_anc = node.dist_to_root - node.up.up.up.dist_to_root

def add_distances_to_children(tree):
    """
    Adds distances and leaves to minor and major child for each node.
    Minor child: child with fewer leaves.
    Major child: child with more leaves.
    """
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.dist_to_minor_child = node.dist_to_major_child = -1
            node.minor_leaves = node.major_leaves = -1
        else:
            children = node.children
            # Sort children by number of leaves
            children.sort(key=lambda n: n.leaves)
            minor_child, major_child = children[0], children[-1]

            node.dist_to_minor_child = minor_child.dist
            node.dist_to_major_child = major_child.dist
            node.minor_leaves = minor_child.leaves
            node.major_leaves = major_child.leaves

def add_distances_to_grandchildren(tree):
    """
    Adds distances and leaves to minor and major grandchildren for each node's minor and major child.
    """
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            # Initialize attributes for leaves
            attributes = [
                'dist_to_minor_minor_grandchild', 'dist_to_minor_major_grandchild',
                'dist_to_major_minor_grandchild', 'dist_to_major_major_grandchild',
                'minor_minor_leaves', 'minor_major_leaves',
                'major_minor_leaves', 'major_major_leaves'
            ]
            for attr in attributes:
                setattr(node, attr, -1)
        else:
            children = node.children
            # Sort children by number of leaves
            children.sort(key=lambda n: n.leaves)
            minor_child, major_child = children[0], children[-1]

            # Process minor child's grandchildren
            if not minor_child.is_leaf():
                gc = minor_child.children
                gc.sort(key=lambda n: n.leaves)
                node.dist_to_minor_minor_grandchild = gc[0].dist_to_root - node.dist_to_root
                node.dist_to_minor_major_grandchild = gc[-1].dist_to_root - node.dist_to_root
                node.minor_minor_leaves = gc[0].leaves
                node.minor_major_leaves = gc[-1].leaves
            else:
                node.dist_to_minor_minor_grandchild = node.dist_to_minor_major_grandchild = -1
                node.minor_minor_leaves = node.minor_major_leaves = -1

            # Process major child's grandchildren
            if not major_child.is_leaf():
                gc = major_child.children
                gc.sort(key=lambda n: n.leaves)
                node.dist_to_major_minor_grandchild = gc[0].dist_to_root - node.dist_to_root
                node.dist_to_major_major_grandchild = gc[-1].dist_to_root - node.dist_to_root
                node.major_minor_leaves = gc[0].leaves
                node.major_major_leaves = gc[-1].leaves
            else:
                node.dist_to_major_minor_grandchild = node.dist_to_major_major_grandchild = -1
                node.major_minor_leaves = node.major_major_leaves = -1

def name_tree_nodes(tree):
    """
    Assigns unique integer names to all nodes in the tree.
    """
    for idx, node in enumerate(tree.traverse("levelorder")):
        node.name = str(idx)

def rescale_tree(tree, target_avg_length):
    """
    Rescales all branch lengths in the tree so that the average branch length equals target_avg_length.
    """
    branch_lengths = [node.dist for node in tree.traverse()]
    avg_length = np.mean(branch_lengths)
    rescale_factor = avg_length / target_avg_length

    for node in tree.traverse():
        node.dist /= rescale_factor

    return rescale_factor

def convert_zero_length_nodes_to_polytomies(tree):
    """
    Converts zero-length internal nodes into polytomies.
    """
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root() and node.dist == 0:
            parent = node.up
            parent.remove_child(node)
            for child in node.children:
                parent.add_child(child)

def process_tree(tree_str):
    """
    Processes a single tree string:
    - Parses the tree.
    - Cleans and annotates the tree.
    - Computes features for embedding.
    - Returns the embedding DataFrame and rescale factor.
    """
    # Parse the tree
    tree = Tree(tree_str, format=1)

    # Remove any root edge if present
    if len(tree.children) == 1:
        tree = tree.children[0]
        tree.up = None

    # Initialize 'visited' attribute
    for node in tree.traverse():
        node.visited = 0

    # Convert zero-length internal nodes to polytomies
    convert_zero_length_nodes_to_polytomies(tree)

    # Assign 't_s' attributes to tips
    assign_t_s_attributes(tree)

    # Add 't_s' attributes to internal nodes
    add_t_s_to_nodes(tree)

    # Name all nodes uniquely
    name_tree_nodes(tree)

    # Rescale branch lengths
    rescale_factor = rescale_tree(tree, TARGET_AVG_BL)

    # Add number of leaves to each node
    add_number_of_leaves(tree)

    # Add distances to root and present
    add_dist_to_root(tree)
    add_dist_to_present(tree)

    # Add distances to ancestors and children
    add_distances_to_ancestors(tree)
    add_distances_to_children(tree)
    add_distances_to_grandchildren(tree)

    # Build the embedding
    tree_embedding = []
    for node in tree.traverse("levelorder"):
        embedding = [
            node.dist_to_present,
            node.dist_to_root,
            node.dist_to_anc,
            node.leaves,
            node.t_s,
            node.dist_to_grand_anc,
            node.dist_to_minor_child,
            node.dist_to_major_child,
            node.minor_leaves,
            node.major_leaves,
            node.dist_to_great_grand_anc,
            node.dist_to_minor_minor_grandchild,
            node.dist_to_minor_major_grandchild,
            node.dist_to_major_minor_grandchild,
            node.dist_to_major_major_grandchild,
            node.minor_minor_leaves,
            node.minor_major_leaves,
            node.major_minor_leaves,
            node.major_major_leaves
        ]
        tree_embedding.append(embedding)

    # Create DataFrame and pad/truncate to fixed size
    df = pd.DataFrame(tree_embedding)
    df = df.reindex(range(999), fill_value=0)  # Ensure 999 rows
    df = df.reindex(range(1000), fill_value=rescale_factor)  # Add rescale factor as the last row

    return df

def main():
    parser = argparse.ArgumentParser(description='Encodes trees into a fixed-size representation with tip states.')
    parser.add_argument('-t', '--tree', type=str, required=True, help='Name of the file with Newick trees')
    parser.add_argument('-o', '--output', type=str, required=True, help='Name of the output encoded file')
    args = parser.parse_args()

    tree_file = args.tree
    output_file = args.output

    # Read trees from file
    with open(tree_file, 'r') as f:
        tree_data = f.read().replace('\n', '')
    tree_strings = [s + ';' for s in tree_data.strip().split(';') if s]

    start_time = time.time()

    # Process each tree and write embeddings to the output file
    for idx, tree_str in enumerate(tree_strings):
        try:
            df = process_tree(tree_str)
            # Write to file (append after the first tree)
            if idx == 0:
                df.to_csv(output_file, sep='\t', index=True, index_label='Index')
            else:
                df.to_csv(output_file, sep='\t', index=True, header=False, mode='a')
        except Exception as e:
            print(f"Error processing tree at index {idx}: {e}", file=sys.stderr)
            continue

        if idx % 100 == 0 and idx > 0:
            elapsed_time = time.time() - start_time
            print(f'Processed {idx} trees in {elapsed_time:.2f} seconds')

if __name__ == "__main__":
    main()
