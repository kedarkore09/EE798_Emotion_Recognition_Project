import os
import numpy as np
import random
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer, pad_sequences
from tensorflow.keras.utils import to_categorical
from google.colab import drive
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Constants
BASE_PATH = "/content/drive/MyDrive/IEMOCAP/IEMOCAP_full_release"
GLOVE_PATH = "/content/drive/MyDrive/glove.6B.300d.txt"
SAMPLE_RATE = 16000
N_MFCC = 32
MAX_AUDIO_DURATION = 3.0  
MAX_AUDIO_LENGTH = int(MAX_AUDIO_DURATION * (SAMPLE_RATE))  
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 30
EMBEDDING_DIM = 300
EMOTIONS = ['ang', 'hap', 'sad', 'neu']

# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Load emotion labels
def load_emotion_labels(session, audio_files):
    labels = []
    eval_path = os.path.join(BASE_PATH, session, 'dialog', 'EmoEvaluation')
    for audio_file in audio_files:
        eval_file = audio_file.replace('.wav', '.txt')
        eval_file_path = os.path.join(eval_path, eval_file)
        if os.path.exists(eval_file_path):
            with open(eval_file_path, 'r') as file:
                for line in file:
                    if line.startswith('['):
                        parts = line.strip().split('\t')
                        start_end_time = parts[0].strip('[]').split(' - ')
                        emotion = parts[2]
                        if emotion in EMOTIONS:
                            labels.append({
                                'start_time': float(start_end_time[0]),
                                'end_time': float(start_end_time[1]),
                                'emotion': emotion,
                                'file': audio_file,
                                'session': session
                            })
    return pd.DataFrame(labels)

# Audio data augmentation
def augment_audio(y, sr):
    y_stretched = librosa.effects.time_stretch(y, rate=0.9)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    noise_factor = 0.005
    noise = np.random.randn(len(y))
    y_noisy = y + noise_factor * noise
    return [y_stretched, y_shifted, y_noisy]

# Synonym replacement for text augmentation
def synonym_replacement(text):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    if random_word_list:
        word_to_replace = random.choice(random_word_list)
        synonyms = wordnet.synsets(word_to_replace)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == word_to_replace else word for word in words]
    return ' '.join(new_words)

# Text data augmentation
def augment_text(text):
    augmented_texts = [text, synonym_replacement(text)]
    return augmented_texts

# Extract audio features using MFCC
def extract_features(audio_file, start_time, duration=MAX_AUDIO_DURATION, sr=SAMPLE_RATE):
    y, sr = librosa.load(audio_file, sr=sr, offset=start_time, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
    features = features[:, :MAX_AUDIO_LENGTH] if features.shape[1] > MAX_AUDIO_LENGTH else np.pad(features, ((0, 0), (0, MAX_AUDIO_LENGTH - features.shape[1])), 'constant')
    return features.T

# Retrieve all audio files for a session
def get_audio_files(base_path):
    audio_files = []
    sessions = [f'Session{i}' for i in range(1, 6)]
    for session in sessions:
        session_path = os.path.join(base_path, session, 'dialog', 'wav')
        files = [file for file in os.listdir(session_path) if file.endswith('.wav')]
        audio_files.extend((session, file) for file in files)
    return audio_files

# Prepare the dataset
def prepare_data(base_path):
    audio_files = get_audio_files(base_path)
    label_encoder = LabelEncoder()  
    label_encoder.fit(EMOTIONS)

    tokenizer = Tokenizer()
    X_audio, X_text, y = [], [], []

    for session, audio_file in audio_files:
        try:
            batch_labels = load_emotion_labels(session, [audio_file])
            if batch_labels.empty:
                continue

            audio_path = os.path.join(base_path, session, 'dialog', 'wav', audio_file)
            y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, offset=batch_labels['start_time'].iloc[0], duration=MAX_AUDIO_DURATION)
            augmented_audios = augment_audio(y_audio, sr)

            for augmented_audio in [y_audio] + augmented_audios:
                features = extract_features(audio_path, batch_labels['start_time'].iloc[0])
                X_audio.append(features.T)
                X_text.append(batch_labels['emotion'].iloc[0])  
                y.append(batch_labels['emotion'].iloc[0])

            text_file = audio_file.replace('.wav', '.txt')
            text_path = os.path.join(base_path, session, 'dialog', 'transcriptions', text_file)
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    text = f.read().strip()
                    augmented_texts = augment_text(text)
                    for augmented_text in augmented_texts:
                        X_text.append(augmented_text)
                        X_audio.append(features.T)  
                        y.append(batch_labels['emotion'].iloc[0])

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    tokenizer.fit_on_texts(X_text)
    X_text = tokenizer.texts_to_sequences(X_text)
    X_text = pad_sequences(X_text, maxlen=MAX_TEXT_LENGTH)

    y_encoded = label_encoder.transform(y)
    y = to_categorical(y_encoded, num_classes=len(EMOTIONS))

    return np.array(X_audio, dtype=np.float32), np.array(X_text), y, tokenizer, label_encoder  

def tdnn_layer(inputs, num_units, context_size, dilation=1):
    return layers.Conv1D(num_units, kernel_size=context_size, dilation_rate=dilation, activation='relu')(inputs)

def extract_x_vectors(inputs):
    x = tdnn_layer(inputs, 128, context_size=5)
    x = tdnn_layer(x, 128, context_size=3)
    x = layers.GlobalAveragePooling1D()(x)
    return x

def swem_pooling(embeddings):
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(embeddings)
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(embeddings)
    concat = layers.concatenate([max_pool, avg_pool], axis=-1)
    return concat

def create_mscnn(inputs, filters, kernel_sizes):
    convs = []
    for k in kernel_sizes:
        conv = layers.Conv1D(filters=filters, kernel_size=k, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
        convs.append(conv)
    concatenated = layers.concatenate(convs, axis=-1)
    return concatenated

def statistical_pooling(features):
    max_pool = layers.GlobalMaxPooling1D()(features)
    avg_pool = layers.GlobalAveragePooling1D()(features)
    std_pool = layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1))(features)
    output = layers.concatenate([max_pool, avg_pool, std_pool], axis=-1)
    return output

def attention_layer(context_vector, features):
    context_vector = layers.Dense(features.shape[-1])(context_vector) 
    context_vector = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(context_vector)
    attention_scores = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([features, context_vector])
    attention_weights = layers.Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_scores)
    weighted_sum = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([features, attention_weights])
    
    return weighted_sum

def build_model(audio_input_shape, text_input_shape, embedding_matrix, num_classes):
    audio_input = layers.Input(shape=audio_input_shape)
    audio_cnn_1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(audio_input)
    audio_cnn_1 = layers.BatchNormalization()(audio_cnn_1)
    audio_cnn_1 = layers.Dropout(0.5)(audio_cnn_1)
    audio_mscnn = create_mscnn(audio_cnn_1, filters=128, kernel_sizes=[5,7,9,11])
    audio_spu = statistical_pooling(audio_mscnn)

    xvector = extract_x_vectors(audio_input)
    xvector_dense = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(xvector)
    xvector_dense = layers.BatchNormalization()(xvector_dense)
    xvector_dense = layers.Dropout(0.5)(xvector_dense)

    audio_features = layers.concatenate([audio_spu, xvector_dense])

    text_input = layers.Input(shape=(text_input_shape,))
    embedding_layer = layers.Embedding(input_dim=embedding_matrix.shape[0],
                                        output_dim=EMBEDDING_DIM,
                                        weights=[embedding_matrix],
                                        trainable=False)(text_input)

    swem_features = swem_pooling(embedding_layer)

    text_cnn_1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(embedding_layer)
    text_cnn_1 = layers.BatchNormalization()(text_cnn_1)
    text_cnn_1 = layers.Dropout(0.5)(text_cnn_1)
    text_mscnn = create_mscnn(text_cnn_1, filters=128, kernel_sizes=[2,3,4,5])
    text_spu = statistical_pooling(text_mscnn)

    attended_text = attention_layer(audio_features, text_mscnn)

    combined = layers.concatenate([audio_features, text_spu, attended_text, swem_features], axis=-1)

    fc = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(0.5)(fc)

    fc2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(fc)
    fc2 = layers.BatchNormalization()(fc2)
    fc2 = layers.Dropout(0.5)(fc2)

    output = layers.Dense(num_classes, activation='softmax')(fc2)

    model = models.Model(inputs=[audio_input, text_input], outputs=output)
    return model

def train_model():
    X_audio, X_text, y, tokenizer, label_encoder = prepare_data(BASE_PATH)
    num_classes = y.shape[1]

    embeddings_index = load_glove_embeddings(GLOVE_PATH)
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

    audio_input_shape = X_audio.shape[1:]
    text_input_shape = X_text.shape[1]
    model = build_model(audio_input_shape, text_input_shape, embedding_matrix, num_classes)

    optimizer = optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)),
                                                        y=np.argmax(y_train, axis=1))
    class_weights_dict = dict(enumerate(class_weights))

    model.fit([X_audio_train, X_text_train], y_train,
                validation_data=([X_audio_test, X_text_test], y_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight=class_weights_dict,
                callbacks=[reduce_lr, early_stopping])

    model.save('/content/drive/MyDrive/emotion_model.h5')

if __name__ == "__main__":
    train_model()
