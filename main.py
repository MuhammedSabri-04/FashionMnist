# ============================================================
# FASHION-MNIST - BASİT CNN
# PNG görüntülerinden CNN ile sınıflandırma
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ============================================================
# 1. SINIFLAR
# ============================================================

CLASS_NAMES = [
    "Tişört",
    "Pantolon",
    "Kazak",
    "Elbise",
    "Ceket",
    "Sandalet",
    "Gömlek",
    "Spor Ayakkabı",
    "Çanta",
    "Bot"
]


# ============================================================
# 2. VERİ KLASÖRLERİ
# ============================================================

DATA_DIR = "data/fashion_mnist"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


# ============================================================
# 3. PNG GÖRÜNTÜLERİNİ OKUMA
# ============================================================

def load_images_from_folder(folder):

    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_folder = os.path.join(folder, class_name)

        files = sorted([
            file
            for file in os.listdir(class_folder)
            if file.lower().endswith(".png")
        ])

        print(f"{class_name}: {len(files)} görüntü")

        for file in files:

            file_path = os.path.join(class_folder, file)

            image = Image.open(file_path).convert("L")

            image = np.array(image, dtype=np.uint8)

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# ============================================================
# 4. EĞİTİM VERİLERİNİ OKU
# ============================================================

print("\nEğitim görüntüleri okunuyor...")

train_images, train_labels = load_images_from_folder(TRAIN_DIR)


# ============================================================
# 5. TEST VERİLERİNİ OKU
# ============================================================

print("\nTest görüntüleri okunuyor...")

test_images, test_labels = load_images_from_folder(TEST_DIR)


# ============================================================
# 6. VERİ BOYUTLARI
# ============================================================

print("\nVeri boyutları:")

print("Train images:", train_images.shape)
print("Train labels:", train_labels.shape)

print("Test images :", test_images.shape)
print("Test labels :", test_labels.shape)


# ============================================================
# 7. SINIF DAĞILIMLARINI KONTROL ET
# ============================================================

print("\nEğitim sınıf dağılımı:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(train_labels == i)

    print(f"{i}: {class_name:18s} -> {count}")


print("\nTest sınıf dağılımı:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(test_labels == i)

    print(f"{i}: {class_name:18s} -> {count}")


# ============================================================
# 8. NORMALİZASYON
# ============================================================

# Piksel değerleri:
# 0 - 255
#
# 255'e bölerek:
# 0 - 1
#
# aralığına getiriyoruz.

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


# ============================================================
# 9. VERİYİ KARIŞTIR
# ============================================================
#
# ÖNEMLİ:
#
# Görüntüler klasörlerden sınıf sınıf okunduğu için
# veri başlangıçta şu şekilde sıralıdır:
#
# Tişört -> Pantolon -> Kazak -> ...
#
# validation_split kullanıldığında verinin son %10'u
# validation olarak ayrılır.
#
# Bu nedenle eğitim verisini önce karıştırıyoruz.
#

np.random.seed(42)

indices = np.random.permutation(len(train_images))

train_images = train_images[indices]
train_labels = train_labels[indices]


# ============================================================
# 10. KANAL BOYUTU EKLE
# ============================================================

# CNN giriş formatı:
#
# yükseklik × genişlik × kanal
#
# 28 × 28 × 1

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)


print("\nCNN giriş boyutu:")

print("Train:", train_images.shape)
print("Test :", test_images.shape)


# ============================================================
# 11. CNN MODELİ
# ============================================================

model = Sequential([

    # --------------------------------------------------------
    # CONVOLUTION
    # --------------------------------------------------------

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # --------------------------------------------------------
    # MAX POOLING
    # --------------------------------------------------------

    MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    Flatten(),

    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    Dense(
        64,
        activation="relu"
    ),

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 12. MODELİ GÖSTER
# ============================================================

print("\n==============================")
print("MODEL")
print("==============================")

model.summary()


# ============================================================
# 13. MODELİ DERLE
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 14. MODELİ EĞİT
# ============================================================

print("\n==============================")
print("MODEL EĞİTİLİYOR")
print("==============================\n")


history = model.fit(

    train_images,

    train_labels,

    epochs=5,

    batch_size=64,

    validation_split=0.1,

    shuffle=True
)


# ============================================================
# 15. EĞİTİM GEÇMİŞİ
# ============================================================

print("\n==============================")
print("EĞİTİM GEÇMİŞİ")
print("==============================")

for epoch in range(len(history.history["loss"])):

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={history.history['loss'][epoch]:.4f}, "
        f"Accuracy={history.history['accuracy'][epoch]:.4f}, "
        f"Val Loss={history.history['val_loss'][epoch]:.4f}, "
        f"Val Accuracy={history.history['val_accuracy'][epoch]:.4f}"
    )


# ============================================================
# 16. LOSS GRAFİĞİ
# ============================================================

epochs = range(
    1,
    len(history.history["loss"]) + 1
)


plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    history.history["loss"],
    marker="o",
    label="Eğitim Loss"
)


plt.plot(
    epochs,
    history.history["val_loss"],
    marker="o",
    label="Doğrulama Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Eğitim ve Doğrulama Loss"
)

plt.xticks(epochs)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. ACCURACY GRAFİĞİ
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    history.history["accuracy"],
    marker="o",
    label="Eğitim Accuracy"
)


plt.plot(
    epochs,
    history.history["val_accuracy"],
    marker="o",
    label="Doğrulama Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Eğitim ve Doğrulama Accuracy"
)

plt.xticks(epochs)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 18. TEST VERİSİ ÜZERİNDE BAŞARI
# ============================================================

test_loss, test_accuracy = model.evaluate(

    test_images,

    test_labels,

    verbose=0
)


print("\n==============================")
print("TEST SONUÇLARI")
print("==============================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : %{test_accuracy * 100:.2f}"
)


# ============================================================
# 19. TAHMİNLER
# ============================================================

predictions = model.predict(

    test_images,

    verbose=0
)


predicted_labels = np.argmax(

    predictions,

    axis=1
)


# ============================================================
# 20. TEST SONUÇLARINI KONTROL ET
# ============================================================

correct = np.sum(
    predicted_labels == test_labels
)

incorrect = np.sum(
    predicted_labels != test_labels
)


print("\n==============================")
print("TAHMİN SONUÇLARI")
print("==============================")

print("Doğru tahmin :", correct)

print("Yanlış tahmin:", incorrect)

print(
    "Toplam       :", len(test_labels)
)


# ============================================================
# 21. RASTGELE 10 TEST GÖRÜNTÜSÜNÜ GÖSTER
# ============================================================

np.random.seed(42)

random_indices = np.random.choice(
    len(test_images),
    size=10,
    replace=False
)


plt.figure(figsize=(15, 6))


for plot_index, i in enumerate(random_indices):

    plt.subplot(2, 5, plot_index + 1)

    plt.imshow(
        test_images[i].squeeze(),
        cmap="gray"
    )

    real_class = CLASS_NAMES[test_labels[i]]

    predicted_class = CLASS_NAMES[predicted_labels[i]]

    confidence = predictions[i][predicted_labels[i]] * 100

    if test_labels[i] == predicted_labels[i]:
        result = "DOĞRU"
    else:
        result = "YANLIŞ"

    title = (
        f"Gerçek: {real_class}\n"
        f"Tahmin: {predicted_class}\n"
        f"Güven: %{confidence:.1f}\n"
        f"{result}"
    )

    plt.title(title, fontsize=9)

    plt.axis("off")


plt.tight_layout()

plt.show()

# ============================================================
# 22. TAHMİN SINIF DAĞILIMI
# ============================================================

print("\n==============================")
print("TAHMİN SINIF DAĞILIMI")
print("==============================")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(predicted_labels == i)

    percentage = count / len(predicted_labels) * 100

    print(
        f"{class_name:18s}: "
        f"{count:5d} (%{percentage:.2f})"
    )