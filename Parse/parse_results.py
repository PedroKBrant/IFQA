import csv
import statistics

# Read the CSV file and extract the scores
scores = []
with open('./results/result.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        score = float(row['Score'])
        scores.append(score)

# Sort the scores in ascending order
sorted_scores = sorted(scores)

# Determine the indices for each quarter
total_scores = len(sorted_scores)
quarter = total_scores // 4
first_quarter_scores = sorted_scores[:quarter]
second_quarter_scores = sorted_scores[quarter:2*quarter]
third_quarter_scores = sorted_scores[2*quarter:3*quarter]
fourth_quarter_scores = sorted_scores[3*quarter:]

# Calculate the mean and standard deviation for each quarter
first_quarter_mean = statistics.mean(first_quarter_scores)
second_quarter_mean = statistics.mean(second_quarter_scores)
third_quarter_mean = statistics.mean(third_quarter_scores)
fourth_quarter_mean = statistics.mean(fourth_quarter_scores)

first_quarter_std_dev = statistics.stdev(first_quarter_scores)
second_quarter_std_dev = statistics.stdev(second_quarter_scores)
third_quarter_std_dev = statistics.stdev(third_quarter_scores)
fourth_quarter_std_dev = statistics.stdev(fourth_quarter_scores)

# Print the statistics for each quarter
print("Statistics")
print("First Quarter")
print(f"Mean: {first_quarter_mean}")
print(f"Standard Deviation: {first_quarter_std_dev}")
print()
print("Second Quarter")
print(f"Mean: {second_quarter_mean}")
print(f"Standard Deviation: {second_quarter_std_dev}")
print()
print("Third Quarter")
print(f"Mean: {third_quarter_mean}")
print(f"Standard Deviation: {third_quarter_std_dev}")
print()
print("Fourth Quarter")
print(f"Mean: {fourth_quarter_mean}")
print(f"Standard Deviation: {fourth_quarter_std_dev}")
