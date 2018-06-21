import pandas as pd
import numpy as np
import csv

filename = "./sentiment labelled sentences/amazon_cells_labelled.txt"
with open(filename) as infile, open('./dataset.csv','w') as outfile:
    for line in infile:
        outfile.write(str(line.rsplit(None,1)))

# Puts reviews into a data frame
df = pd.read_csv('./dataset.csv') # read the file into a pandas data frame
print(df.head())       # print the first few rows of the data frame
