'''
we have provided a small set of data samples (cnt-bundles) to demonstrate the pipeline of
processing raw data into graph dataset in dgl format
'''
import pandas as pd

from graph import rawToDglGraph

if __name__=="__main__":

    # Modify the following variable to the location of a text file containing the data files to be used during testing.
    cntArrsFile = r'path\to\datafile\names\to\be\used\for\training'

    with open(cntArrsFile, 'r') as file:
        fileNames = [line.split()[0].strip() for line in file]
    # global feature data + labels. Modify 'usecols' to change which columns are used as features.
    data = pd.read_excel(r'path\to\the\file\CNT_bundle_global_features.xlsx', index_col=0, usecols=[0, 1, 2, 3, 4, 5, 8, 10, 14, 15])
    filenames = [file.split(".")[0] for file in fileNames]
    data = data.loc[filenames].to_numpy()
    
    globalFeatures = data[:, :7]

    labels = data[:, 7:]

    # Number of Tubes as extra node features in atomic level graph [G_s]
    extraNodeFeatures = globalFeatures[:, 5:6]

    print('Beginning processing raw data...')

    processor = rawToDglGraph()

    processor.getGraphDataset(
        dirPath=r'path\to\SI_dataset\CNT_bundle_structures',
        fileNames=fileNames,
        savePath=r'path\to\save\results',
        normalization=True,
        globalFeatures=globalFeatures,
        labels=labels,
        extraNodeFeatures=extraNodeFeatures
    )

    print('Raw data have successfully transformed into DGL graph format!')
