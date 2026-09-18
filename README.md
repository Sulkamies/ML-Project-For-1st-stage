This repository is a basis for the data managing architecture used in our project. 

main.py has a small demo of how the data is processed, including a reverse PCA transformation. 

The data preprocessing happens in data_split.py, where data is imported and split into train and test sets, 
after which they are transformed by calling the pca_transform -function, found in pca.py.
