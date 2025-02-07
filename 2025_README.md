Hierarchical Spatial Graph Neural Network (HS-GNN) for property prediction of carbon nanostructures. 
The HS-GNN was jointly developed by Yusu Wang, University of California, San Diego, and Hendrik Heinz, CU Boulder. Contributors include Zilu Li and Qi Zhao, UCSD, and Jordan Winetrout, CU Boulder.


The HS-GNN is designed to predict mechanical properties of defective and pristine carbon nanostructures from 3D atomic model structures on the scale of 1 to 50 nm, such as carbon nanotube bundles, carbon-fiber cross-sections, and CNT junctions. It may be extended for other sizes, chemistries, and properties. 

The associated databases of over 2000 model structures with up to 100,000 atoms and associated stress-strain curves are available from Figshare (Winetrout, Jordan; Li, Zilu; Zhao, Qi; Gaber, Landon; Unnikrishnan, Vinu U.; Varshney, Vikas; et al. (2024). Dataset of Carbon Nanostructures for "Prediction of Carbon Nanostructure Mechanical Properties and the Role of Defects Using Machine Learning". figshare. Dataset. https://doi.org/10.6084/m9.figshare.27634290.v2) or the authors. The files would be too large to include in GitHub. Prediction times for modulus and tensile strength, after building persistent summaries and training, are under 1 second for 100+ all-atomic structures.  

Details are described in a publication under review, "Prediction of Carbon Nanostructure Mechanical Properties and Role of Defects Using Machine Learning". A preprint is available here: https://arxiv.org/abs/2110.00517.

Contact information:
Yusu Wang: yusuwang@ucsd.edu
Hendrik Heinz: hendrik.heinz@colorado.edu

************

We recommend using the folders and files starting with "2025" for the most streamlined user experience. The folders and files contain the code for HS-GNN, XGBoost and a small test run related to the paper "Prediction of Carbon Nanostructure Mechanical Properties and Role of Defects Using Machine Learning".   

The 2025 implementation is run under Python 3.9 and requires the following packages: 

- dgl == 2.2.1+cu118
- numpy == 1.26.4
- pytorch == 2.1.0+cu118
- networkx == 3.3
- gudhi == 3.10.1
- scipy == 1.13.1


A small "toy" model subset is given in the folder "2025_HS_GNN_toydata". The complete benchmark datasets, CNT-bundle and CNT junctions, are in: Winetrout, Jordan; Li, Zilu; Zhao, Qi; Gaber, Landon; Unnikrishnan, Vinu U.; Varshney, Vikas; et al. (2024). Dataset of Carbon Nanostructures for "Prediction of Carbon Nanostructure Mechanical Properties and the Role of Defects Using Machine Learning". figshare. Dataset. https://doi.org/10.6084/m9.figshare.27634290.v2. 

For the older, non-2025 folders and files, we used the requirements:
- python >= 3.8  
- pytorch == 1.12.1+cu113
- dgl == 0.9.0+cu113
- torch-geometric == 2.1.0
- networkx == 3.3 or lower

To train HS-GNN under the benchmark dataset using the older (non-2025) code, please use the following script:
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

In addition to HS-GNN, we also share the code for XGBoost and other baselines (non-2025 folders and files).
