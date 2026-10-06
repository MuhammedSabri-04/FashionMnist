# ============================================================
# TEK FOTOĞRAF TAHMİNİ (DEMO)
# Eğitilmiş modele, eğitimde hiç görmediği tek bir PNG verip
# hangi sınıfa ait olduğunu adım adım tahmin ettirir.
#
# Kullanım:
#   python tek_foto_tahmin.py
#   python tek_foto_tahmin.py "data/fashion_mnist/test/Gömlek/image_00005.png"
# ============================================================

import os
import sys

# TensorFlow'un bilgi mesajlarının çoğunu gizle (ekran temiz kalsın)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from PIL import Image

tf.get_logger().setLevel("ERROR")


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

MODEL_PATH = "model.keras"

DEFAULT_IMAGE = os.path.join("data", "fashion_mnist", "test", "Çanta", "image_00001.png")

image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE


def print_header(text):
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def wait_for_enter():
    input("\n   [Devam etmek için Enter'a bas]")


# ============================================================
# ADIM 1: EĞİTİLMİŞ MODELİ YÜKLE
# ============================================================

print_header("ADIM 1: Eğitilmiş model yükleniyor")

# compile=False: sadece tahmin yapacağız, optimizer'a gerek yok
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model yüklendi         :", MODEL_PATH)
print("Beklediği giriş boyutu :", model.input_shape)

wait_for_enter()


# ============================================================
# ADIM 2: FOTOĞRAFI GRİ SEVİYEDE OKU (main.py ile aynı)
# ============================================================

print_header("ADIM 2: Fotoğraf okunuyor")

image = Image.open(image_path).convert("L")

print("Dosya :", image_path)
print("Boyut :", image.size, "piksel")
print("Mod   :", image.mode, "(L = gri seviye)")

wait_for_enter()


# ============================================================
# ADIM 3: NUMPY DİZİSİNE ÇEVİR VE TERMİNALDE GÖSTER
# ============================================================

print_header("ADIM 3: Fotoğraf sayılara (NumPy dizisi) çevriliyor")

pixels = np.array(image, dtype=np.uint8)

print("Şekil                   :", pixels.shape)
print("En küçük / büyük piksel :", pixels.min(), "/", pixels.max())

# Her pikseli parlaklığına göre bir karakterle çiz (koyu -> boşluk, açık -> @)
chars = " .:-=+*#%@"

print("\nFotoğrafın terminaldeki hâli:\n")

for row in pixels:
    print("   " + "".join(chars[int(p) * len(chars) // 256] * 2 for p in row))

print("\nOrtadan 10 piksel, pixels[14, 9:19]:", pixels[14, 9:19])

wait_for_enter()


# ============================================================
# ADIM 4: NORMALİZASYON (0-255 -> 0-1)
# ============================================================

print_header("ADIM 4: Normalizasyon (0-255 -> 0-1)")

x = pixels.astype("float32") / 255.0

print("Aynı 10 piksel, x[14, 9:19]:", np.round(x[14, 9:19], 2))

wait_for_enter()


# ============================================================
# ADIM 5: MODELİN BEKLEDİĞİ BİÇİME GETİR
# ============================================================

print_header("ADIM 5: Modelin beklediği biçime getiriliyor")

print("Şu an         :", x.shape)

# Kanal boyutu: 28 x 28 -> 28 x 28 x 1 (main.py'deki gibi)
x = np.expand_dims(x, axis=-1)

print("Kanal eklendi :", x.shape)

# Batch boyutu: model fotoğrafları hep grup hâlinde alır, burada grup 1 fotoğraf
x = np.expand_dims(x, axis=0)

print("Batch eklendi :", x.shape)

wait_for_enter()


# ============================================================
# ADIM 6: TAHMİN (10 SINIF OLASILIĞI)
# ============================================================

print_header("ADIM 6: Model tahmin ediyor (10 olasılık)")

# [0]: gruptaki ilk (ve tek) fotoğrafın 10 olasılığı
probabilities = model.predict(x, verbose=0)[0]

for i, class_name in enumerate(CLASS_NAMES):

    bar = "█" * int(round(probabilities[i] * 40))

    print(f"{i}: {class_name:14s} %{probabilities[i] * 100:6.2f}  {bar}")

print(f"\nToplam: %{probabilities.sum() * 100:.2f}")

wait_for_enter()


# ============================================================
# ADIM 7: SONUÇ
# ============================================================

print_header("ADIM 7: Sonuç")

predicted_label = np.argmax(probabilities)

confidence = probabilities[predicted_label] * 100

# Gerçek sınıf klasör adından gelir; model bu bilgiyi hiç görmedi
real_class = os.path.basename(os.path.dirname(image_path))

print("Tahmin :", CLASS_NAMES[predicted_label])
print(f"Güven  : %{confidence:.2f}")

if real_class in CLASS_NAMES:

    print("Gerçek :", real_class, "(klasör adından)")

    if CLASS_NAMES[predicted_label] == real_class:
        print("Sonuç  : DOĞRU")
    else:
        print("Sonuç  : YANLIŞ")

else:
    print("Gerçek : bilinmiyor (dosya bir sınıf klasöründe değil)")
