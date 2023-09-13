import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

pd.set_option('display.max_columns', None)

# Verileri yükle ve kontrol et
data = pd.read_csv(os.path.join("dataset.csv"))

# Veri setini düzenleme ve resimleri yükleme
data["label_id"] = pd.Categorical(data["label"]).codes

# Resimleri yükle ve boyutlandır
data["image"] = data["image_path"].map(lambda x: Image.open(x).resize((224, 224)))

print(data.head())
print(data["image"].map(lambda x: np.array(x).shape).value_counts())

# train test bölmeleri
X = np.array([np.array(img) for img in data["image"]])  # PIL Image nesnelerini numpy dizilerine dönüştür
X = preprocess_input(X.astype('float32'))  # Verileri önceden işle (RGB renk düzeni ve VGG16 için)

y = data["label_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# VGG16 ile özellik çıkarma
vgg16_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

X_train_features = vgg16_model.predict(X_train)
X_test_features = vgg16_model.predict(X_test)

# Özellikleri düzleştir
X_train_flatten = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_flatten = X_test_features.reshape(X_test_features.shape[0], -1)

# SVM modeli
svm_model = SVC(kernel='linear', C=1.0, random_state=0)
svm_model.fit(X_train_flatten, y_train)

# Modeli test et
y_pred = svm_model.predict(X_test_flatten)

# Doğruluk skoru
print("Test Score: {}".format(accuracy_score(y_test, y_pred)))

# Karmaşıklık matrisi
conf_mat = confusion_matrix(y_test, y_pred)
# Karmaşıklık matrisini görselleştir
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.title("Karmaşıklık Matrisi")
plt.show()

# Sınıflandırma raporu
print(classification_report(y_test, y_pred))

pickle.dump(svm_model, open("svm_model.pkl", "wb"))