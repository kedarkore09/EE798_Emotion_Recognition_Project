# evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# Load paths
PREDICTIONS_SAVE_PATH = "/content/drive/MyDrive/predictions.npz"
LABEL_ENCODER_SAVE_PATH = "/content/drive/MyDrive/label_encoder.pkl"

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load predictions and true labels
predictions_data = np.load(PREDICTIONS_SAVE_PATH)
y_pred_classes = predictions_data['y_pred_classes']
y_true_classes = predictions_data['y_true_classes']

# Load label encoder to get class names
with open(LABEL_ENCODER_SAVE_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Calculate metrics
wa = accuracy_score(y_true_classes, y_pred_classes)
ua = recall_score(y_true_classes, y_pred_classes, average='macro')
print(f'Weighted Accuracy (WA): {wa}')
print(f'Unweighted Accuracy (UA): {ua}')
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
