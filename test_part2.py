# test.py

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from google.colab import drive
import pickle

# Set seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Paths and constants
MODEL_SAVE_PATH = "/content/drive/MyDrive/best_model.keras"
TOKENIZER_SAVE_PATH = "/content/drive/MyDrive/tokenizer.pkl"
LABEL_ENCODER_SAVE_PATH = "/content/drive/MyDrive/label_encoder.pkl"
EMBEDDING_MATRIX_SAVE_PATH = "/content/drive/MyDrive/embedding_matrix.npy"
TEST_DATA_SAVE_PATH = "/content/drive/MyDrive/test_data.npz"
PREDICTIONS_SAVE_PATH = "/content/drive/MyDrive/predictions.npz"

EMOTIONS = ['ang', 'hap', 'sad', 'neu']

# Monte Carlo Dropout
@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super(MCDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)

# Monte Carlo Inference
def monte_carlo_inference(model, input_data, batch_size=None, n_samples=10):
    if batch_size is None:
        predictions = [model(input_data, training=True) for _ in range(n_samples)]
        predictions = tf.stack(predictions)
        mean_prediction = tf.reduce_mean(predictions, axis=0)
        variance = tf.reduce_mean(tf.math.reduce_variance(predictions, axis=0), axis=0)
        return mean_prediction, variance
    else:
        all_mean_predictions = []
        all_variances = []
        total_samples = len(input_data[0])
        for batch_start in range(0, total_samples, batch_size):
            batch_end = batch_start + batch_size
            audio_batch = input_data[0][batch_start:batch_end]
            text_batch = input_data[1][batch_start:batch_end]
            batch_predictions = [model([audio_batch, text_batch], training=True) for _ in range(n_samples)]
            batch_predictions = tf.stack(batch_predictions)
            mean_batch_prediction = tf.reduce_mean(batch_predictions, axis=0)
            variance_batch = tf.reduce_mean(tf.math.reduce_variance(batch_predictions, axis=0), axis=0)
            all_mean_predictions.append(mean_batch_prediction)
            all_variances.append(variance_batch)
        mean_prediction = tf.concat(all_mean_predictions, axis=0)
        variance = tf.concat(all_variances, axis=0)
        return mean_prediction, variance

def main(): 
    
    # Load tokenizer and label encoder
    with open(TOKENIZER_SAVE_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_SAVE_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    # Load embedding matrix
    embedding_matrix = np.load(EMBEDDING_MATRIX_SAVE_PATH)

    # Load test data
    test_data = np.load(TEST_DATA_SAVE_PATH)
    X_audio_test = test_data['X_audio_test']
    X_text_test = test_data['X_text_test']
    y_test = test_data['y_test']
    
    # Load the trained model 
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'MCDropout': MCDropout}, safe_mode=False)

    # Monte Carlo Inference
    mean_prediction, variance = monte_carlo_inference(model, [X_audio_test, X_text_test], batch_size=32, n_samples=10)
    y_pred_classes = np.argmax(mean_prediction, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Save predictions and true labels
    np.savez_compressed(PREDICTIONS_SAVE_PATH, y_pred_classes=y_pred_classes, y_true_classes=y_true_classes)

if __name__ == "__main__":
    main()
