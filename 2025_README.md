Readme

This folder contains the code and a small test run for the paper "Prediction of Carbon Nanostructure Mechanical Properties and Role of Defects Using Machine Learning".

The implementation is run under Python 3.9 and requires the following packages: 

- dgl == 2.2.1+cu118
- numpy == 1.26.4
- pytorch == 2.1.0+cu118
- networkx == 3.3
- gudhi == 3.10.1
- scipy == 1.13.1

We have published complete benchmark datasets, CNT-bundle and CNT junctions, in: Winetrout, Jordan; Li, Zilu; Zhao, Qi; Gaber, Landon; Unnikrishnan, Vinu U.; Varshney, Vikas; et al. (2024). Dataset of Carbon Nanostructures for "Prediction of Carbon Nanostructure Mechanical Properties and the Role of Defects Using Machine Learning". figshare. Dataset. https://doi.org/10.6084/m9.figshare.27634290.v2. A small "toy" model subset is given in the folder "2025_HS_GNN_toydata".

To train HS-GNN under the benchmark dataset, please use the following script:
```shell
python train.py
```

Please be aware that script arguments are allowed, for example: 
```shell
python train.py --epoch 800 --lr 0.001 --bs 128
```
defining some hyper-params including number of total training epochs, learning rate and training mini batch size.
The default set of hyper-params in *arguments.py* is selected as the best during cross-validation.


In addition, we have provided a pipeline to process the raw data into DGl graph format, following the heterogeneous graph representation and hierarchical graph coarsening as described in the paper.
The pipeline illustrates the process of a toy dataset, which is actually a small subset from CNT-bundle dataset. 
To start the process, run the following script in terminal:
```shell
python dataprocessing/run.py
```
You can also bring your own data using HS-GNN following the provided pipeline.

In addition to HS-GNN, we also share the code for XGBoost.
