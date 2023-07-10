import matplotlib.pyplot as plt

mean_values = [0.0963, 0.4193, 0.0966, 0.4379]
std_values = [0.1571, 0.1593, 0.1593, 0.1571]
labels = ["WIDER", "WIDER_CF", "WIDER_DP2", "WIDER_DP2_CF"]

# Plotting the mean values
x = range(len(mean_values))
plt.bar(x, mean_values, yerr=std_values, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(x, labels)  # Set x-axis tick locations and labels
plt.ylabel('Value')
plt.title('Mean and Standard Deviation')

plt.show()
