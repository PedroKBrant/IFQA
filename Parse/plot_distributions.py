import pandas as pd
import matplotlib.pyplot as plt

# Define the file paths for each CSV file
file_paths = ['results/WIDER.csv', 
              'results/WIDER_CF.csv', 
              'results/WIDER_DP2.csv', 
              'results/WIDER_DP2_CF.csv', 
              'results/WIDER_CF.csv',
              'results/WIDER_DP2_CF.csv']#'results/WIDER_CF_DP2.csv'

title = ['A) WIDER Original',
         'B) WIDER & CodeFormer',
         'C) WIDER & DeepPrivacy2',
         'D) WIDER & DeepPrivacy2 & CodeFormer',
         'E) WIDER & CodeFormer',
         'F) WIDER & CodeFormer & DeepPrivacy2',
         ]

# Create a subplot with two rows and three columns
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Iterate over the file paths and plot the distributions
for i, file_path in enumerate(file_paths):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract scores
    scores = df['Score']
    
    # Plot distribution
    axs[i].hist(scores, bins=10, edgecolor='black')
    axs[i].set_xlabel('Score')
    axs[i].set_ylabel('Frequency')
    axs[i].set_title(title[i])

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

