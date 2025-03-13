import numpy as np

class SkipGramModel:
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-0.01, 0.01, (embedding_dim, vocab_size))
        self.lr = learning_rate
        # embedding_dim = N
        # vocab_size = M
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def forward(self, one_hot_vector):
        hidden_layer = np.dot(one_hot_vector, self.W1)  # Shape: (N,)
        output_layer = np.dot(hidden_layer, self.W2)    # Shape: (M,)
        output_layer = self._softmax(output_layer)      # Shape: (M,)
        if np.isnan(output_layer).any() or np.isinf(output_layer).any():
            raise ValueError("Softmax output contains NaN or Inf.")
        return hidden_layer, output_layer
    
    def backward(self, one_hot_vector, target_vector, learning_rate=0.1, clip_value=1.0):
        hidden_layer, output_layer = self.forward(one_hot_vector)
        error = target_vector - output_layer  # Shape: (M,)
        output_layer_gradient = np.outer(hidden_layer, error)  # Shape: (N,M) # perkalian utk optimasi kkomputer 
        hidden_layer_gradient = np.dot(self.W2, error)  # Shape: (N,)
        input_index = np.argmax(one_hot_vector)
        # Clip gradients
        hidden_layer_gradient = np.clip(hidden_layer_gradient, -clip_value, clip_value)
        output_layer_gradient = np.clip(output_layer_gradient, -clip_value, clip_value)
        # Update weights
        self.W1[input_index] += learning_rate * hidden_layer_gradient
        self.W2 += learning_rate * output_layer_gradient

    def save(self, file_path):
        np.savez(file_path, W1=self.W1, W2=self.W2)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        data = np.load(file_path)
        self.W1 = data['W1']
        self.W2 = data['W2']
        print(f"Model loaded from {file_path}")


