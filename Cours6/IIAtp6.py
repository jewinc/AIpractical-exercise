import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#! DO NOT MODIFY THE IMPORTS

STUDENT_LAST_NAME = "CHENG"  #Fill in your last name
STUDENT_FIRST_NAME = "Jewin"  #Fill in your first name


# Data preparation function
def prepare_sentiment_data(max_features=2000, max_documents=2000):
    """
    Prepare the 20 newsgroups dataset for sentiment analysis

    Parameters:
    -----------
    max_features : int
        Maximum number of words to include in vocabulary
    max_documents : int
        Maximum number of documents to use

    Returns:
    --------
    X : array-like of shape (n_samples, max_features)
        The document-term matrix
    labels : array-like of shape (n_samples,)
        The target labels
    vectorizer : CountVectorizer
        The fitted vectorizer for transforming new texts
    """
    #! DO NOT MODIFY THIS FUNCTION
    # Select specific categories (topics) from the dataset
    categories = ["rec.sport.baseball", "sci.space"]

    # Fetch the dataset while removing metadata (headers, footers, and quotes) for cleaner text
    newsgroups = fetch_20newsgroups(
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    # Limit the dataset to the specified number of documents
    texts = newsgroups.data[:max_documents]
    labels = newsgroups.target[:max_documents]

    # Use CountVectorizer to convert text data into a document-term matrix
    vectorizer = CountVectorizer(
        max_features=max_features,  # Limit vocabulary size to the most frequent words
        stop_words="english",  # Remove common stop words
        min_df=2,  # Include words that appear in at least 2 documents
        max_df=0.95,  # Exclude words that appear in more than 95% of documents
    )

    # Fit the vectorizer to the text and transform it into a sparse matrix
    X = vectorizer.fit_transform(texts).toarray()

    # Normalize the matrix so that each row (document) sums to 1.
    # This preprocessing step that makes the feature values comparable across documents, regardless of their lengths.
    # Summing to 1 is equivalent to representing each row as a probability distribution over the vocabulary.
    X = X / np.maximum(np.sum(X, axis=1, keepdims=True), 1e-12)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Number of documents: {X.shape[0]}")
    print(f"Features per document: {X.shape[1]}")

    return X, labels, vectorizer


class Activations:
    @staticmethod
    def relu(z):
        return max(0, z)

    @staticmethod
    def softmax(z):
        # TODO 
        tmp = z.copy()
        for i in range(len(tmp)):
            denom = np.sum(tmp[i])
            for j in range(len(tmp[i])):
                tmp[i][j] = np.exp(tmp[i][j])/denom
        return tmp


class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation = getattr(Activations, activation)
        self.activation_derivative = getattr(
            Activations, f"{activation}_derivative", None
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        # TODO: YOUR CODE HERE - Initialize weights and biases
        # Biases should be initialized to zeros
        # Weights should be initialized using He initialization for ReLU and Xavier initialization for softmax
        self.bias = [0] * self.output_size
        if self.activation_name =='relu':
            #HE init
            self.weights = np.random.normal(0, np.sqrt(2/self.input_size), (self.input_size, self.output_size))
        elif self.activation_name == "softmax":
            #Xavier init
            self.weights = np.random.normal(0, np.sqrt(1/(self.input_size + self.output_size)), (self.input_size, self.output_size))
        else:
            raise ValueError("activation function not recognized")


class BatchNormLayer:
    """
    Implements batch normalization for a single layer in a neural network.

    Batch normalization normalizes the output of a layer to have a mean of 0
    and a variance of 1 during training, and scales and shifts this normalized
    output using learnable parameters (gamma and beta). During inference, it
    uses a running mean and variance computed during training.

    Remember that batch normalization speeds up training, mitigates exploding and vanishing gradients, and improves generalization.

    Attributes:
    -----------
    size : int
        The number of neurons (features) in the layer.
    eps : float
        A small constant added to the variance to avoid division by zero.
    gamma : ndarray of shape (size,)
        The scaling parameter, learned during training.
    beta : ndarray of shape (size,)
        The shifting parameter, learned during training.
    running_mean : ndarray of shape (size,)
        The running mean of the layer's activations, used during inference.
    running_var : ndarray of shape (size,)
        The running variance of the layer's activations, used during inference.
    """

    def __init__(self, size, eps=1e-8):
        self.size = size
        self.eps = eps
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)
        self.running_mean = np.zeros(size)
        self.running_var = np.ones(size)

    def forward(self, x, training=True):
        # TODO: YOUR CODE HERE - Implement batch normalization forward pass
        # 1. In training mode, calculate the mean and variance along the correct axis.
        # 1.1 Update the running mean and variance using the calculated mean and variance: 0.9 x running_mean + 0.1 x mean (same with variance).
        #       (0.9 and 0.1 are chosen empirically because it works well in most scenarios)
        # 2. In inference mode, use the running mean and variance to normalize the input.
        # 3. Normalize the input using the mean and variance (x_norm = ???).
        #       x_norm = (x - mean) / sqrt(var + eps)
        # 4. Scale and shift the normalized input using gamma and beta.
        #       return self.gamma * x_norm + self.beta
        if training:
            mean = np.mean(x, axis=1)
            var = np.var(x, axis=1)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var

            x_norm = (x - self.running_mean)/np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
        else:
            x_norm = (x - self.running_mean)/np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta


class DeepNLP:
    def __init__(self, layer_sizes, activation="relu", use_batch_norm=True):
        """
        Initialize the DeepNLP model with specified layer sizes and optional batch normalization.

        Parameters:
        -----------
        layer_sizes : list of int
            A list where each entry specifies the number of neurons in each layer.
            - For example, [input_size, hidden1, hidden2, ..., output_size].
        activation : str, optional
            The activation function to use for hidden layers (default is 'relu').
            - The last layer always uses 'softmax' as the activation function for classification tasks.
        use_batch_norm : bool, optional
            Whether to include batch normalization layers (default is True).

        Attributes:
        -----------
        layers : list of Layer
            Stores the neural network's fully connected layers.
        batch_norm_layers : list of BatchNormLayer or None
            Stores batch normalization layers corresponding to each layer (None if not used).
        """
        # Initialize lists to store layers and batch normalization layers.
        self.layers = []
        self.batch_norm_layers = []

        # Iterate through the specified layer sizes to construct the network.
        for i in range(len(layer_sizes) - 1):
            # Create a fully connected (dense) layer.
            layer = Layer(
                layer_sizes[i],  # Number of inputs to the layer.
                layer_sizes[i + 1],  # Number of outputs (neurons) in the layer.
                (
                    "softmax" if i == len(layer_sizes) - 2 else activation
                ),  # Use 'softmax' for the last layer, otherwise the specified activation.
            )
            # Add the layer to the list of layers.
            self.layers.append(layer)

            # If batch normalization is enabled and this is not the last layer, add a BatchNormLayer.
            if use_batch_norm and i < len(layer_sizes) - 2:
                bn_layer = BatchNormLayer(
                    layer_sizes[i + 1]
                )  # Batch normalization is applied to the outputs of this layer.
                self.batch_norm_layers.append(bn_layer)
            else:
                # Append None if batch normalization is not used for this layer.
                self.batch_norm_layers.append(None)

    def forward(self, X, training=True):
        """Forward pass through the network
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data
        training : bool
            Whether to use training mode for batch normalization
        Returns:
        --------
        activations : list of arrays
                The output of each layer in the network
        layer_inputs : list of arrays
                The input to each layer in the network
        """
        layer_inputs = [X]
        activations = []

        n = len(self.layers)
        for i in range(n):
                layer: Layer = self.batch_norm_layers[i] if training else self.layers[i] 
                x2 = np.matmul(layer_inputs[i], layer.weights)
                x3 = np.add(x2, layer.bias)
                activations.append(x3)
                if i<n:
                    layer_inputs.append(x3)
        return (activations, layer_inputs)


class ModelTrainer:
    def __init__(
        self, model, learning_rate=0.01, batch_size=32, lr_decay=0.9, patience=5
    ):
        """
        Initialize the ModelTrainer with the given parameters.

        Parameters:
        -----------
        model : DeepNLP
            The neural network model to be trained.
        learning_rate : float, optional
            Initial learning rate for weight updates (default is 0.01).
        batch_size : int, optional
            Number of samples per mini-batch during training (default is 32).
        lr_decay : float, optional
            Factor by which the learning rate is multiplied when validation accuracy
            does not improve for `patience` epochs (default is 0.9).
        patience : int, optional
            Number of epochs to wait without validation improvement before decaying
            the learning rate (default is 5).

        Attributes:
        -----------
        model : DeepNLP
            The neural network model to be trained.
        learning_rate : float
            Current learning rate for training.
        batch_size : int
            Number of training samples in each batch.
        lr_decay : float
            Factor by which the learning rate is reduced after `patience` epochs.
        patience : int
            Epoch patience before decaying the learning rate.
        history : dict
            Tracks training and validation accuracy, and learning rate history for
            monitoring and visualization.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.patience = patience
        self.history = {"train_acc": [], "valid_acc": [], "learning_rate": []}

    def train_batch(self, X_batch, y_batch):
        """Train on a single batch"""
        # TODO: YOUR CODE HERE - Implement backpropagation and parameter updates
        pass

    def evaluate(self, X, y):
        """Compute accuracy for the given data"""
        # Forward pass through the model (training=False ensures no parameter updates).
        activations, _ = self.model.forward(X, training=False)

        # Use the last layer's activations to make predictions (highest probability class).
        # `np.argmax` extracts the predicted class for each sample.
        predictions = np.argmax(activations[-1], axis=1)

        # Compute the mean of correct predictions (accuracy).
        return np.mean(predictions == y)

    def train_epoch(self, X, y):
        # one epoch = one complete pass through the entire training dataset

        # Shuffle the indices for the current epoch
        # to break correlations in the data (if data is ordered in some way)
        # and to ensure more effective mini-batch training.
        indices = np.random.permutation(len(X))
        total_loss = 0

        # Iterate through the dataset in mini-batches
        # (faster convergence, better generalization)
        for start_idx in range(0, len(X), self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            self.train_batch(X[batch_idx], y[batch_idx])

        # Return the average loss for the epoch
        return total_loss / (start_idx // self.batch_size + 1)

    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        # Initialize the best validation accuracy and patience counter.
        best_val_acc = 0
        patience_counter = 0

        # Loop through the specified number of epochs.
        for epoch in range(epochs):
            # Train for one epoch.
            train_loss = self.train_epoch(X_train, y_train)
            # the loss is not used in this implementation because we are using accuracy as the metric

            # Compute training and validation accuracy.
            train_acc = self.evaluate(X_train, y_train)
            valid_acc = self.evaluate(
                X_valid, y_valid
            )  # used to prevent overfitting and to monitor generalization

            # Check if validation accuracy improves.
            if valid_acc > best_val_acc:
                # Update best validation accuracy and reset patience counter.
                best_val_acc = valid_acc
                patience_counter = 0
            else:
                # Increment patience counter and decay learning rate if patience is exceeded.
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.learning_rate *= self.lr_decay
                    patience_counter = 0

            # Record metrics and learning rate in the training history.
            self.history["train_acc"].append(train_acc)
            self.history["valid_acc"].append(valid_acc)
            self.history["learning_rate"].append(self.learning_rate)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Accuracy: {train_acc:.4f} | "
                    f"Valid Accuracy: {valid_acc:.4f} | "
                    f"Learning Rate: {self.learning_rate:.6f}"
                )

        return self.history


class Visualizer:
    #! DO NOT MODIFY THIS CLASS
    @staticmethod
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history["train_acc"], label="Train")
        ax1.plot(history["valid_acc"], label="Validation")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        ax2.plot(history["learning_rate"])
        ax2.set_title("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test your implementation
    X, y, vectorizer = prepare_sentiment_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2
    )

    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))

    layer_sizes = [input_size, 512, 256, 128, num_classes]
    
    model = DeepNLP(layer_sizes, activation="relu", use_batch_norm=True)
    
    trainer = ModelTrainer(
        model, learning_rate=0.01, batch_size=32, lr_decay=0.9, patience=5
    )
    '''
    history = trainer.train(X_train, y_train, X_valid, y_valid, epochs=50)
    
    visualizer = Visualizer()
    visualizer.plot_training_history(history)
    '''
