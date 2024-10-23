import numpy as np
from sklearn.metrics import accuracy_score, recall_score, classification_report
from tensorflow.keras.models import load_model
from helpers import prepare_data

# Constants
BASE_PATH = "/content/drive/MyDrive/IEMOCAP/IEMOCAP_full_release"
BATCH_SIZE = 64

def test_model():
    # Load test data
    X_audio, X_text, y, tokenizer, label_encoder = prepare_data(BASE_PATH)
    X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

    # Load trained model
    model = load_model('/content/drive/MyDrive/emotion_model.h5')

    # Evaluate model
    y_pred = model.predict([X_audio_test, X_text_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy and report
    wa = accuracy_score(y_true_classes, y_pred_classes)
    ua = recall_score(y_true_classes, y_pred_classes, average='macro')

    print(f'Weighted Accuracy (WA): {wa}')
    print(f'Unweighted Accuracy (UA): {ua}')
    print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

if __name__ == "__main__":
    test_model()
