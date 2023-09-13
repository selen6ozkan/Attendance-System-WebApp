import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Verileri yükle ve kontrol et
data = pd.read_csv(os.path.join("C:\\Users\\SELEN\\Desktop\\flaskweb\\flask_api\\dataset.csv"))

# Veri setini düzenleme ve resimleri yükleme
data["label_id"] = pd.Categorical(data["label"]).codes

# Resimleri yükle ve boyutlandır
data["image"] = data["image_path"].map(lambda x: Image.open(x).resize((224, 224)))

print(data.head())
print(data["image"].map(lambda x: np.array(x).shape).value_counts())

# Train-test bölmeleri
X = np.array([np.array(img) for img in data["image"]])  # PIL Image nesnelerini numpy dizilerine dönüştür
X = preprocess_input(X.astype('float32'))  # Verileri önceden işle (RGB renk düzeni ve VGG16 için)

y = data["label_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# VGG16 ile özellik çıkarma
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Özelliklerin çıkarılması
X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

# Özellikleri düzleştir
X_train_flatten = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_flatten = X_test_features.reshape(X_test_features.shape[0], -1)

# CNN (Convolutional Neural Network) modeli
input_shape = (7, 7, 512)
num_classes = len(np.unique(y))

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

datagen.fit(X_train_flatten.reshape(X_train_flatten.shape[0], 7, 7, 512))

epochs = 200
batch_size = 32

earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        verbose=1,
                                        patience=20)

history = model.fit_generator(datagen.flow(X_train_flatten.reshape(X_train_flatten.shape[0], 7, 7, 512), y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_test_flatten.reshape(X_test_flatten.shape[0], 7, 7, 512), y_test),
                              steps_per_epoch=X_train_flatten.shape[0] // batch_size,
                              callbacks=[earlystopping])

# Modelin eğitimi sırasında kaydedilen accuracy ve loss değerlerinin görselleştirilmesi
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Modelin test edilmesi
y_pred = model.predict(X_test_flatten.reshape(X_test_flatten.shape[0], 7, 7, 512))
y_pred_classes = np.argmax(y_pred, axis=1)

# Doğruluk skoru
print("Test Score: {}".format(accuracy_score(y_test, y_pred_classes)))

# Karmaşıklık matrisi
conf_mat = confusion_matrix(y_test, y_pred_classes)
# Karmaşıklık matrisinin görselleştirilmesi
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Sınıflandırma raporu
print(classification_report(y_test, y_pred_classes))

# save model
model.save("cnn_model.h5")
