# model.py
from sklearn.neural_network import MLPClassifier


class MLPModel:
    def __init__(
        self,
        hidden_layer_sizes=(128,),
        max_iter=200,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.001,
        activation="relu",
        batch_size="auto",
        momentum=0.9,
        learning_rate="constant",
        tol=1e-4,
        random_state=1,
        verbose=False,
    ):
        """
        Initialize the MLP model with full hyperparameter customization.

        :param hidden_layer_sizes: Tuple, number of neurons in hidden layers.
        :param max_iter: Maximum number of training iterations.
        :param alpha: L2 regularization parameter.
        :param solver: Optimization algorithm ('sgd', 'adam', 'lbfgs').
        :param learning_rate_init: Initial learning rate.
        :param activation: Activation function ('relu', 'tanh', 'logistic').
        :param batch_size: Batch size for training.
        :param momentum: Momentum for 'sgd' solver.
        :param learning_rate: Learning rate schedule ('constant', 'adaptive', 'invscaling').
        :param tol: Tolerance for stopping criterion.
        :param random_state: Random seed.
        :param verbose: Whether to print training progress.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            solver=solver,
            learning_rate_init=learning_rate_init,
            activation=activation,
            batch_size=batch_size,
            momentum=momentum,
            learning_rate=learning_rate,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def train(self, X_train, Y_train):
        """Train the MLP model."""
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """Predict output for given input data."""
        return self.model.predict(X)

    def evaluate(self, X_test, Y_test):
        """Evaluate the model using test data."""
        y_pred = self.predict(X_test)
        return y_pred
