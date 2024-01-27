import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import logging
from tqdm import tqdm

# Установка уровня логирования
logging.basicConfig(level=logging.INFO)


# Функция для загрузки данных из папки с подпапками
def load_data_from_folder(folder_path):
    file_paths = []
    labels = []

    for label, subfolder in enumerate(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if filename.endswith(".wav"):
                    file_paths.append(file_path)
                    labels.append(label)

    return file_paths, labels


# Функция для предобработки данных
def preprocess_data(file_paths, labels, max_length=431):
    spectrograms = []
    for file_path, label in tqdm(zip(file_paths, labels), desc='Preprocessing'):
        # Загрузка аудиофайла и преобразование в спектрограмму
        y, sr = librosa.load(file_path, duration=3)  # Ограничим длительность для упрощения
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Выравнивание спектрограммы до максимальной длины
        pad_width = max_length - spectrogram.shape[1]
        if pad_width > 0:
            spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Преобразование в dB шкалу
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        spectrograms.append(spectrogram)

    # Преобразование в numpy массивы
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    return spectrograms, labels


# Загрузка данных
folder_path = "/content/drive/MyDrive/dataset"  # Замените на реальный путь к вашей папке dataset
file_paths, labels = load_data_from_folder(folder_path)

# Предобработка данных
spectrograms, labels = preprocess_data(file_paths, labels)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Преобразование меток в one-hot кодировку
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

y_train_int = np.argmax(y_train_one_hot, axis=1)
y_test_int = np.argmax(y_test_one_hot, axis=1)

X_train_expanded = np.expand_dims(X_train, axis=-1)
X_test_expanded = np.expand_dims(X_test, axis=-1)

# Создание модели CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 431, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(7, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_int, epochs=10, validation_data=(X_test, y_test_int))

test_loss, test_accuracy = model.evaluate(X_test, y_test_int)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
