from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA,FactorAnalysis, FastICA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn import cluster, metrics, mixture
import pandas as pd

# Function to load Fashion-MNIST dataset
def load_data():
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return x_train_all, y_train_all, x_test, y_test

# Function to split data into train, validation, and test sets
def split_data(x_train_all, y_train_all):
    # Split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1, random_state=42)
    return x_train, x_val, y_train, y_val

def preprocess_data(x_train, x_test, x_val):
  # Reshaping the dataset
  x_train = x_train.reshape(54000, 784)
  x_val = x_val.reshape(6000, 784)
  x_test = x_test.reshape(10000, 784)

  # Change integers to 32-bit floating point numbers
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_val = x_val.astype('float32')


  # Initialize the MinMaxScaler and fit on the training data
  scaler = MinMaxScaler()
  x_train = scaler.fit_transform(x_train)

  # Use the same scaler to transform the test and validation data
  x_test = scaler.transform(x_test)
  x_val = scaler.transform(x_val)

  return x_train,x_test,x_val


def print_dimensions(x_train, x_val, x_test):
    # Print dimensions of the datasets
    print("Διαστάσεις του training set:", x_train.shape)
    print("Διαστάσεις του validation set:", x_val.shape)
    print("Διαστάσεις του test set:", x_test.shape)

def apply_pca(x_train,x_test):

  start_time = time.time()  # Record the start time

  # Fit the PCA on the training set and set the model to retain 85% of the variance
  pca = PCA(n_components=.85)
  pca.fit(x_train)

  #Apply the mapping to the training & test set of data
  #train_img = pca.transform(x_train)
  test_img = pca.transform(x_test)

  end_time = time.time()  # Record the end time
  training_time = end_time - start_time

  print(f'Total number of components used after PCA : {pca.n_components_}')
  print("PCA time:",training_time)

  return pca, test_img, training_time

def apply_fa(x_train,x_test):
      #Factor Analysis (FA):

      # Initialize the FA model with the desired number of components
      n_components = 11

      start_time = time.time()  # Record the start time
      fa = FactorAnalysis(n_components=n_components, random_state=0)

      fa.fit(x_train)

      # Fit and transform your data to the reduced-dimensional space
      x_train_fa = fa.fit_transform(x_train)

      #Apply the mapping to test set
      test_img = fa.transform(x_test)

      end_time = time.time()  # Record the end time
      training_time = end_time - start_time

      return fa, test_img, training_time

def apply_FastICA(x_train, x_test):
      # Initialize the ICA model with the desired number of components
      n_components = 5  # Choose the number of components

      start_time = time.time()  # Record the start time
      ica = FastICA(n_components=n_components, random_state=0)

      # Fit the ICA model to the training data
      ica.fit(x_train)

      # Transform the test data using the learned components
      test_img = ica.transform(x_test)

      end_time = time.time()  # Record the end time
      training_time = end_time - start_time

      return ica, test_img, training_time

def train_SAE(x_train, x_test, x_val):
      # Define the dimensions
      original_dim = x_train.shape[1]
      encoding_dims = [128, 64, 128]  # adjust the number of neurons in each layer
      dropout_rate = 0.0  # Adjust the dropout rate as needed

      # Build the stacked autoencoder model with dropout layers
      input_layer = keras.layers.Input(shape=(original_dim,))
      encoded = keras.layers.Dense(encoding_dims[0], activation='relu')(input_layer)
      encoded = keras.layers.Dropout(dropout_rate)(encoded)

      # Build the encoder part of the stacked autoencoder
      for dim in encoding_dims[1:]:
          encoded = keras.layers.Dense(dim, activation='relu')(encoded)
          encoded = keras.layers.Dropout(dropout_rate)(encoded)

      # Build the decoder part of the stacked autoencoder
      decoded = keras.layers.Dense(encoding_dims[-2], activation='relu')(encoded)
      decoded = keras.layers.Dropout(dropout_rate)(decoded)

      for dim in reversed(encoding_dims[:-1]):
          decoded = keras.layers.Dense(dim, activation='relu')(decoded)
          decoded = keras.layers.Dropout(dropout_rate)(decoded)

      # Output layer
      output_layer = keras.layers.Dense(original_dim, activation='sigmoid')(decoded)

      # Build the full autoencoder model
      SAE = keras.models.Model(input_layer, output_layer)

      # Compile the model
      SAE.compile(optimizer='adam', loss='mean_squared_error')

      #plot the architecture
      SAE.summary()

      callback = keras.callbacks.EarlyStopping(monitor='loss', patience=45)

      start_time = time.time()  # Record the start time
      # Train the autoencoder
      history_SAE = SAE.fit(x_train, x_train,shuffle=True,epochs = 10, batch_size = 125,validation_data=(x_val, x_val),callbacks=[callback])

      end_time = time.time()  # Record the end time
      training_time = end_time - start_time

      # Use the encoder part for dimensionality reduction

      # Define the encoder using the input and the output of the encoder layers
      encoder = keras.models.Model(input_layer, encoded)

      # Apply the mapping to the test set of data
      test_img = encoder.predict(x_test)

      print(f"Original Dimension: {original_dim}")
      print(f"Encoded Dimension: {encoded.shape.as_list()[-1]}")


      return SAE, history_SAE, test_img, training_time

def train_CSAE(x_train_reshaped, x_test, x_val):

  # Convolutional Stacked Autoencoder (CSAE)
  input_layer = keras.layers.Input(shape=(28, 28, 1))
  conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
  maxpool1 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv1)
  conv2 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(maxpool1)
  maxpool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

  # Encoder part
  flatten = keras.layers.Flatten()(maxpool2)
  encoded = keras.layers.Dense(64, activation='relu')(flatten)

  # Decoder part
  decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)
  reshaped = keras.layers.Reshape((28, 28, 1))(decoded)

  # Build the CSAE model
  CSAE = keras.models.Model(input_layer, reshaped)
  CSAE.compile(optimizer='adam', loss='mean_squared_error')

  # Model summary
  CSAE.summary()

  # Early stopping callback
  callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  start_time = time.time()  # Record the start time
  # Train the CSAE
  history_CSAE = CSAE.fit(x_train_reshaped, x_train_reshaped,epochs=6,batch_size=125,shuffle=True,validation_data=(x_val, x_val),callbacks=[callback])
  end_time = time.time()  # Record the end time
  training_time = end_time - start_time

  # Use the encoder part for dimensionality reduction

  # Define the encoder using the input and the output of the encoder layers
  encoder = keras.models.Model(input_layer, encoded)

  # Apply the mapping to the training & test set of data
  test_img = encoder.predict(x_test)

  # Print original and encoded dimensions
  print(f"Original Dimension: {x_train_reshaped.shape[1:]}")
  print(f"Encoded Dimension: {encoded.shape.as_list()[1:]}")


  return CSAE, history_CSAE,test_img, training_time

def plot_original_vs_reconstructed(original_images, reconstructed_images):
	# Reshape the images to their original dimensions
	original_images = original_images.reshape(-1, 28, 28)
	reconstructed_images = reconstructed_images.reshape(-1, 28, 28)

	# Create a 2x10 grid for plotting
	fig, axes = plt.subplots(2, 10, figsize=(20, 4))
	for i in range(10):
		axes[0, i].imshow(original_images[i], cmap='gray')
		axes[0, i].set_title('Original')
		axes[0, i].axis('off')

		axes[1, i].imshow(reconstructed_images[i], cmap='gray')
		axes[1, i].set_title('Reconstructed')
		axes[1, i].axis('off')

	plt.tight_layout()
	plt.show()

def select_random_indices(y_test):
  unique_classes = np.unique(y_test)

  # Initialize an array to store selected indices
  selected_indices = []

  # Randomly select one index per class
  for class_label in unique_classes:
    class_indices = np.where(y_test == class_label)[0]
    random_index = np.random.choice(class_indices, 1, replace=False)
    selected_indices.append(random_index)

  # Convert the list of indices to a NumPy array
  selected_indices = np.concatenate(selected_indices)

  return selected_indices

#define a performance evaluation function
def performance_score(input_values, cluster_indexes, true_labels):
    try:
        silh_score = metrics.silhouette_score(input_values, cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        #print( ' ... -1: incorrect, 0: overlapping, +1: highly dense clusts.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score = metrics.calinski_harabasz_score(input_values, cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        #print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = metrics.davies_bouldin_score(input_values, cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
        #print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index score.')
        db_score = -999

    try:
        ari_score = metrics.adjusted_rand_score(true_labels, cluster_indexes)
        print(' .. Adjusted Rand Index score is {:.2f}'.format(ari_score))
        #print(' ... Perfect labeling is 1.0.')
    except:
        print(' .. Warning: could not calculate Adjusted Rand Index score.')
        ari_score = -999
        #print('... -0.5 >= Similarity <=  1.0, 0.0: random labelings, 1.0: perfect match.')

    return silh_score, ch_score, db_score, ari_score

# Function to append results to a DataFrame
def append_to_dataframe(results_df, technique_name, clustering_algorithm, training_time, execution_time, num_clusters,calinski_harabasz, davies_bouldin, silhouette, ari):
    results_df = results_df.append({
        'Dimensionality Reduction Technique': technique_name,
        'Clustering Algorithm': clustering_algorithm,
        'Training Time (s)': training_time,
        'Execution Time (s)': execution_time,
        'Number of Clusters': num_clusters,
        'Calinski–Harabasz Index': calinski_harabasz,
        'Davies–Bouldin Index': davies_bouldin,
        'Silhouette Score': silhouette,
        'Adjusted Rand Index': ari
    }, ignore_index=True)
    return results_df

def display_random_images(x_data, y_data, predicted_labels, num_classes=4, num_images_per_class=4):
    # Select 4 sample images from 4 classes
    selected_ind = np.array([select_random_indices(y_data)[:num_images_per_class] for _ in range(num_classes)]).T.flatten()

    fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(12, 8))

    j = 0
    for idx in selected_ind:
      original_image = x_data[idx]

      axes[y_data[idx], j].imshow(original_image.reshape(28, 28), cmap='gray')
      axes[y_data[idx], j].set_title(f'Cluster {predicted_labels[idx]}')
      axes[y_data[idx], j].axis('off')
      j = (j + 1) % num_images_per_class

    plt.show()

#Load data
x_train_all, y_train_all, x_test, y_test = load_data()
x_train, x_val, y_train, y_val = split_data(x_train_all, y_train_all)
print_dimensions(x_train, x_val, x_test)


x_train, x_test, x_val = preprocess_data(x_train, x_test, x_val)

print("\nAfter reshaping and scaling data...")
print_dimensions(x_train, x_val, x_test)

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Dimensionality Reduction Technique', 'Clustering Algorithm',
                                    'Training Time (s)', 'Execution Time (s)',
                                    'Number of Clusters', 'Calinski–Harabasz Index',
                                    'Davies–Bouldin Index', 'Silhouette Score', 'Adjusted Rand Index'])

# Select 10 random images (1 from each class)
selected_indices = select_random_indices(y_test)
original_images = x_test[selected_indices]

# List of dimensionality reduction techniques
techniques = ['PCA', 'Factor Analysis', 'FastICA', 'Stacked Autoencoder','CSAE', 'Raw']

for technique in techniques:
    print(f"\nApplying {technique}...")

    if technique == 'PCA':

      pca, test_img, training_time = apply_pca(x_train, x_test)

      # Plot explained variance
      plt.plot(range(pca.n_components_),pca.explained_variance_ratio_, label='component-wise')
      plt.plot(range(pca.n_components_),np.cumsum(pca.explained_variance_ratio_), label='Cumulative')
      plt.xlabel("Component ID")
      plt.ylabel("explained variance")
      plt.legend(loc='best')
      plt.title("Explained Variance")
      plt.show()

      print("Cumulative Explained Variance for PCA: ")
      print(pca.explained_variance_ratio_.cumsum())

      # Plot original vs reconstructed images
      reconstructed_images = pca.inverse_transform(test_img[selected_indices])

      plot_original_vs_reconstructed(original_images, reconstructed_images)
      eps = 2.9
      minpts = 86

    elif technique == 'Factor Analysis':


      fa, test_img, training_time = apply_fa(x_train,x_test)

      # Compute cumulative explained variance
      comulative_noise_variance = np.cumsum(fa.noise_variance_) / np.sum(fa.noise_variance_)

      explained_variance_by_component = np.var(fa.components_, axis=1)
      cumulative_explained_variance = np.cumsum(explained_variance_by_component) / np.sum(explained_variance_by_component)


      plt.plot(range(11), cumulative_explained_variance)
      plt.xlabel("Component ID")
      plt.ylabel("Explained Variance (Cumulative)")
      plt.title("Explained Variance for FA")
      plt.show()

      plt.plot(range(784), comulative_noise_variance)
      plt.xlabel("Component ID")
      plt.ylabel("Noise Variance (Cumulative)")
      plt.title("Comulative Noise Variance for FA")
      plt.show()

      reconstructed_images = test_img[selected_indices].dot(fa.components_) + fa.mean_

      plot_original_vs_reconstructed(original_images, reconstructed_images)

      eps = 0.8
      minpts = 22
    elif technique == 'FastICA':


      ica, test_img, training_time = apply_FastICA(x_train, x_test)

      # Get the kurtosis of each component
      kurtosis_values = np.abs(ica.components_).mean(axis=1)

      # Compute cumulative kurtosis
      cumulative_kurtosis = np.cumsum(kurtosis_values) / np.sum(kurtosis_values)

      # Plot cumulative kurtosis
      plt.plot(range(5), cumulative_kurtosis)
      plt.xlabel("Component ID")
      plt.ylabel("Cumulative Kurtosis")
      plt.title("Cumulative Kurtosis for ICA Components")
      plt.show()

      reconstructed_images = test_img[selected_indices].dot(ica.mixing_.T) + ica.mean_
      plot_original_vs_reconstructed(original_images, reconstructed_images)

      eps = 0.000976
      minpts = 10

    elif technique == 'Stacked Autoencoder':


      SAE, history_SAE, test_img, training_time = train_SAE(x_train, x_test, x_val)

      #plot the performance during training, for train and val sets
      #plot plt.figure(figsize=(14,6))
      plt.plot(history_SAE.history[list(history_SAE.history.keys())[0]])
      plt.plot(history_SAE.history[list(history_SAE.history.keys())[1]])
      plt.title('loss error')
      plt.ylabel(list(history_SAE.history.keys())[0])
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='best')
      plt.show()


      reconstructed_images = SAE.predict(original_images)
      plot_original_vs_reconstructed(original_images,reconstructed_images)
      eps = 2.46
      minpts = 5
    elif technique == 'CSAE':
      # Reshape the data for convolutional autoencoder
      x_train_reshaped = x_train.reshape((len(x_train), 28, 28, 1))
      x_test_reshaped = x_test.reshape((len(x_test), 28, 28, 1))
      x_val_reshaped = x_val.reshape((len(x_val), 28, 28, 1))


      CSAE, history_CSAE, test_img, training_time = train_CSAE(x_train_reshaped,x_test_reshaped, x_val_reshaped)

      # Plot training history
      plt.plot(history_CSAE.history['loss'])
      plt.plot(history_CSAE.history['val_loss'])
      plt.title('CSAE Training Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Validation'], loc='upper right')
      plt.show()


      # Apply the mapping to the test set of data
      reconstructed_images = CSAE.predict(x_test_reshaped)

      # Reshape the reconstructed images back to the original shape
      reconstructed_images = reconstructed_images.reshape((len(x_test), 784))

      # Plot original vs. reconstructed images
      reconstructed_images = reconstructed_images[selected_indices]
      plot_original_vs_reconstructed(original_images, reconstructed_images)

      eps = 4.76875
      minpts = 5
    elif technique == 'Raw':
      test_img = x_test.copy()
      training_time = 0
      eps = 4.2
      minpts = 5


    # Perform clustering and evaluate performance for each technique

    #1. Minibatch kmeans
    start_time = time.time()

    mbkm = cluster.MiniBatchKMeans(n_clusters = 10)
    mbkm.fit(test_img)
    clusterLabels = mbkm.labels_
    execution_time = time.time() - start_time

    n_clusters_ = max(clusterLabels)+1

    print("MiniBatchKMeans:")
    print('we have:', str(n_clusters_), 'clusters')
    silh_score, ch_score, db_score, ari_score =  performance_score(test_img, clusterLabels, y_test)
    results_df = append_to_dataframe(results_df, technique, 'MinibatchKmeans', training_time, execution_time,n_clusters_, ch_score,db_score, silh_score, ari_score)

    #display_random_images(x_test.copy(), y_test, clusterLabels)

    #2. DBSCAN
    start_time = time.time()

    dbc = cluster.DBSCAN(eps= eps, min_samples = minpts)
    dbc.fit(test_img)
    clusterLabels = dbc.labels_
    execution_time = time.time() - start_time

    n_clusters_ = len(set(clusterLabels)) - (1 if -1 in clusterLabels else 0)

    print("DBSCAN:")
    print('we have:', str(n_clusters_), 'clusters')
    silh_score, ch_score, db_score, ari_score = performance_score(test_img, clusterLabels, y_test)

    results_df = append_to_dataframe(results_df, technique, 'DBSCAN', training_time, execution_time,n_clusters_, ch_score,db_score, silh_score, ari_score)

    #display_random_images(x_test.copy(), y_test, clusterLabels)

    #3. Agglomerative Clustering
    start_time = time.time()

    agg_clustering = cluster.AgglomerativeClustering(n_clusters = 10)

    # Fit the model to your data
    agg_clustering.fit(test_img)

    # Obtain cluster labels
    clusterLabels = agg_clustering.labels_

    execution_time = time.time() - start_time

    n_clusters_ = max(clusterLabels)+1

    print("AGG:")
    print('we have:', str(n_clusters_), 'clusters')
    silh_score, ch_score, db_score, ari_score = performance_score(test_img, clusterLabels, y_test)

    results_df = append_to_dataframe(results_df, technique, 'AGG', training_time, execution_time, n_clusters_, ch_score,db_score, silh_score, ari_score)

    #display_random_images(x_test.copy(), y_test, clusterLabels)

    #4. Gaussian Mixture Models
    start_time = time.time()

    gm = mixture.GaussianMixture(n_components=10, random_state=0).fit(test_img)
    clusterLabels = gm.predict(test_img)

    execution_time = time.time() - start_time

    n_clusters_ = max(clusterLabels)+1

    print("GMM:")
    print('we have:', str(n_clusters_), 'clusters')
    silh_score, ch_score, db_score, ari_score = performance_score(test_img, clusterLabels, y_test)

    results_df = append_to_dataframe(results_df, technique, 'GMM', training_time, execution_time, n_clusters_, ch_score,db_score, silh_score, ari_score)

    display_random_images(x_test.copy(), y_test, clusterLabels)

    #5. Bisecting KMeans
    start_time = time.time()

    bisect_means = cluster.BisectingKMeans(n_clusters=10)
    bisect_means.fit(test_img)
    clusterLabels = bisect_means.labels_

    execution_time = time.time() - start_time

    n_clusters_ = max(clusterLabels)+1

    print("BisectingKMeans:")
    print('we have:', str(n_clusters_), 'clusters')
    silh_score, ch_score, db_score, ari_score = performance_score(test_img, clusterLabels, y_test)

    results_df = append_to_dataframe(results_df, technique, 'BisectingKMeans', training_time, execution_time, n_clusters_, ch_score,db_score, silh_score, ari_score)

    #display_random_images(x_test.copy(), y_test, clusterLabels)


results_df.to_excel('output.xlsx', index=False)

