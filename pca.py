from sklearn.decomposition import PCA

def pca_transformation(X_train_raw, X_test_raw, n) :
    pca = PCA(n_components = n)  
     
    X_train_compressed = pca.fit_transform(X_train_raw) # computes the covariance matrix for the training data and projects the training data to the k-dimensional principal subspace
    X_test_compressed = pca.transform(X_test_raw) # projects the test data set to the same k-dimensional principal subspace

    return X_train_compressed, X_test_compressed, pca

# returns the 
def inverse_pca(X_compressed, pca) : return pca.inverse_transform(X_compressed)