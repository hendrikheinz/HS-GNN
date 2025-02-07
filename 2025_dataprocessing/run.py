'''
we have provided a small set of data samples (cnt-bundles) to demonstrate the pipeline of
processing raw data into graph dataset in dgl format
'''
import pandas as pd

from graph import rawToDglGraph

if __name__=="__main__":

    cntArrsFile = 'data/toydata/fileNames'

    with open(cntArrsFile, 'r') as file:
        fileNames = [line.strip() for line in file]
        
    # global feature data + labels
    data = pd.read_csv('data/toydata/data.csv', index_col=0, usecols=[1, 2, 3, 4, 5, 6, 7, 9, 10, 14, 15])
    data = data.loc[fileNames].to_numpy()

    globalFeatures = data[:, :8]

    labels = data[:, 8:]

    # tube diameter as extra node features in atomic level graph [G_s]
    extraNodeFeatures = globalFeatures[:, 5:6]

    print('Beginning processing raw data...')

    processor = rawToDglGraph()

    processor.getGraphDataset(
        dirPath='data/toydata/raw/',
        fileNames=fileNames,
        savePath='data/toydata/',
        normalization=True,
        globalFeatures=globalFeatures,
        labels=labels,
        extraNodeFeatures=extraNodeFeatures
    )

    print('Raw data have successfully transformed into DGL graph format!')
