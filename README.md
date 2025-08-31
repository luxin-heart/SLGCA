# SLGCA: Spatial Cross-Level Graph Contrastive Autoencoder for multi-slice spatial domain Identification and microenvironment exploration
<img width="869" height="799" alt="image" src="https://github.com/user-attachments/assets/efb19bb0-4a15-418e-af21-95ad0891c7d9" />

# Overview
As shown in Figure 1, the SLGCA framework takes the preprocessed gene expression matrix and spatial adjacency matrix as input. A contrastive view is generated through data augmentation of the original input, and both views are fed into a shared GCN encoder to obtain latent embeddings. SLGCA adopts a dual-channel cross-level contrastive learning strategy: local-level contrast captures relationships between nodes and their neighborhoods, while global-level contrast ensures structural consistency across the entire dataset. Following encoding, a symmetric decoder reconstructs the gene expression matrix, and an inner product decoder reconstructs the adjacency matrix. The model is trained using a combination of reconstruction and contrastive loss, ensuring effective integration of spatial and gene expression features. After training, the reconstructed gene expression matrix is reduced via PCA, and spatial domains are identified through clustering methods such as Mclust or Leiden.

# Requirements
You'll need to install the following packages in order to run the codes.
- python==3.10
- torch==2.2.2
- torch-cluster==1.6.3
- torch-geometric==2.6.1
- torch-scatter==2.1.2
- torch-sparse==0.6.18 
- torch-spline-conv==1.2.2 
- cudnn=6.0
- scanpy==1.11.1
- numpy==1.24.4
- anndata==0.11.4
- rpy2==3.5.17
- pandas==2.2.3
- scipy==1.15.2
- scikit-learn==1.5.2
- tqdm==4.64.0
- matplotlib==3.4.2
- R==4.2.0

# Tutorial
For a step-by-step tutorial, see the tutorial folder.
In addition, we have saved the clustering results of SLGCA on the DLPFC dataset into the adata.h5ad file of the dataset. You can directly reproduce and obtain our results using the sample code provided.

# Download data
The data we used for training can be downloaded from https://zenodo.org/records/15860656. We also provide datasets processed by SLGCA, which can be downloaded from Dataset folder (in h5ad format, where obs['SLGCA'] is the clustering result, obsm['emb_pca'] is the low-dimensional feature
