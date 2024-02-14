Hierarchical Spatial Graph Neural Network (HS-GNN) for property prediction of carbon nanostructures. 
The HS-GNN was jointly developed by Yusu Wang, University of California, San Diego, and Hendrik Heinz, CU Boulder. Contributors include Zilu Li and Qi Zhao, UCSD, and Jordan Winetrout, CU Boulder.


The HS-GNN is designed to predict mechanical properties of defective and pristine carbon nanostructures, such as carbon nanotube bundles, carbon-fiber cross-sections, and CNT junctions on the scale of 1 to 50 nm from 3D atomic structures. It may be extended for other sizes, chemistries, and properties. 

The associated databases of over 2000 model structures are available from the authors (too large to include in GitHub). Prediction times for modulus and tensile strength, after building persistent summaries and training, are under 1 second for over 100 all-atomic structures.  

Details are described in a publication under review, "Prediction of Carbon Nanostructure Mechanical Properties and Role of Defects Using Machine Learning". A preprint is available here: https://arxiv.org/abs/2110.00517.

Contact information:
Yusu Wang: yusuwang@ucsd.edu
Hendrik Heinz: hendrik.heinz@colorado.edu

----
#### Requirements
- python >= 3.8  
- pytorch == 1.12.1+cu113
- dgl == 0.9.0+cu113
- torch-geometric == 2.1.0
- networkx == ? (seems there is a version conflict with scipy sparse)
