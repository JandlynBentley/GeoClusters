"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

# This program is used in supplement to the clustering algorithms by pre-processing a specific data set
# that had already been stripped of the leading comment block and converted from a .tab file to a .csv file.
# For any other data set, adjust this code as needed.
"""

import csv

# Step 1) Copy data from .csv file into an array to work with
blank_counter = 0
data = []
with open('McDonough-etal_2019_comments_removed.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        data_row = []
        for col in row:
            data_row.append(col)  # add items to data_row
        data.append(data_row)  # add row to data

# Step 2) Collect indices of all rows with missing data (the following columns are of personal interest)
# pH - column 10
# depth sample collected - column 15
# precipitation mean wet quarter - 31
# SO4 - column 44
missing = []
index = 0
for row in data:
    if row[10] == "" or row[15] == "" or row[31] == "" or row[44] == "":
        # if pH is missing a value, add its index to the list for removal
        missing.append(index)
    index += 1
# Those rows will be removed to conform to the requirements of the K-Means algorithm

# Step 3) Loop through data array and transfer only the non-offending rows to a new array
index = 0
new_data = []

for row in data:
    # If this row's index is among the ones specified with missing data, skip it
    if index not in missing:
        new_data.append(row)
    index += 1

# Step 4) Write the smaller data set to a new .csv file which can be used for the algorithm testing
with open('McDonough-etal_2019_test.csv', mode='w') as data_file:
    file_writer = csv.writer(data_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in new_data:
        file_writer.writerow(row)

