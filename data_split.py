from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pca import pca_transformation

k = 200 # number of principal components chosen

def dataImport() :
    # Fetch the dataset 
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

    # Extract features and labels
    X_raw = fashion_mnist.data       # 2D numpy array of the shape (70000, 784) 
    y = fashion_mnist.target     # 1D numpy array of the shape (70000,)

    # Normalize pixel brightness values
    X_raw_normalized = 1 / 255 * X_raw 

    return X_raw_normalized, y

def trainTestSplit(X_raw, y) : 
    # Perform split into training set (60000 data points) and test set (10000 data points)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, 
        y, 
        test_size=10000, 
        random_state=42, # ensures we get the same random split if we run it again
        stratify=y       # ensures equal amounts of each clothing type in both sets
    )

    # Compute the projections of the data to the k-dimensional principal subspace (determined by the training data)
    X_train_compressed, X_test_compressed, pca = pca_transformation(X_train_raw, X_test_raw, k)

    return X_train_compressed, X_test_compressed, y_train, y_test, pca


