from data_split import dataImport, trainTestSplit
from pca import inverse_pca
from data_visualisation import retainer

# Import raw (but normalized) data
X_raw, y = dataImport()

# Performs split into train and test sets and computes the compressed representation in the k-dimensional principal subspace for both
X_train_compressed, X_test_compressed, y_train, y_test, pca = trainTestSplit(X_raw, y)

# Transform the compressed data back to the 784-dimensional feature space
X_inverse = inverse_pca(X_train_compressed, pca) 

# Visualize a given datapoint that has been transformed to the principal subspace and then transformed back to feature subspace; visualization for information lost in the tansformation
retainer(X_inverse[0, :], y_train[0])
retainer(X_inverse[5, :], y_train[5])
retainer(X_inverse[6, :], y_train[6])


# For testing

#print(X_train_compressed.shape)
#print(X_test_compressed.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(X_inverse.shape)

