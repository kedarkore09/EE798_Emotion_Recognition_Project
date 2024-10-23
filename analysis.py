import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from helpers import prepare_data

# Constants
BASE_PATH = "/content/drive/MyDrive/IEMOCAP/IEMOCAP_full_release"

def plot_confusion_matrix():
    # Load test data
    X_audio, X_text, y, tokenizer, label_encoder = prepare_data(BASE_PATH)
    X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

    # Load trained model
    model = load_model('/content/drive/MyDrive/emotion_model.h5')

    # Predict and generate confusion matrix
    y_pred = model.predict([X_audio_test, X_text_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()
