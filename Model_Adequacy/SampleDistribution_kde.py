import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# Load the CSV file
file_path = 'PhyloCNN_1000_Posterior.csv' 
df = pd.read_csv(file_path)

# Initialize an empty dictionary to store 10,000 random draws for each column
random_samples = {}

# For each column in the DataFrame
for column in df.columns:
    # Convert column data to numpy array
    data = df[column].values
    
    # Fit a kernel density estimate (KDE) to the data
    kde = gaussian_kde(data, bw_method='scott')  # 'scott' is the default bandwidth method
    
    # Generate 10,000 points from the fitted KDE
    samples = kde.resample(10000).flatten()  # Flatten the result to a 1D array
    
    # Store the samples in the dictionary
    random_samples[column] = samples

# Convert the dictionary into a DataFrame to view or analyze the samples
sample_df = pd.DataFrame(random_samples)

# Change the order of columns
reordered_sample_df = sample_df[['R_nought', 'x_transmission', 'fraction_1', 'infectious_period_rescaled']]  # Replace with actual column names if they differ

# Add a new column 'tree_size' with all values set to 200
reordered_sample_df['tree_size'] = 200

# Add a new column 'sampling_proba' with values sampled from a uniform distribution between 0.2 and 0.3
reordered_sample_df['sampling_proba'] = np.random.uniform(0.2, 0.3, size=reordered_sample_df.shape[0])

# Save the modified DataFrame to a gzipped CSV file
reordered_sample_df.to_csv('BDSS/Priors.csv.gz', index=False)

# Print the first few rows to verify
print(reordered_sample_df.head())

# If you want, you can save these samples to a new CSV file
reordered_sample_df.to_csv('BDSS/Priors.csv.gz', index=False)