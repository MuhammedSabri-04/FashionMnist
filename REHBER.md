# Fashion-MNIST CNN Projesi: Öğrenme Rehberi

Bu rehber, projedeki `main.py` dosyasını Python'a ve bilgisayarlı görüye yeni başlayan biri için baştan sona açıklıyor. Her bölümde kodun **ne yaptığını**, **neden yaptığını** ve **arkasındaki kavramı** bulacaksın. Sonda, eğitilmiş modelle tek bir fotoğrafı adım adım tahmin eden `tek_foto_tahmin.py` demosu da anlatılıyor.

`main.py` 22 numaralı bölüme ayrılmış (örneğin `# 8. NORMALİZASYON`). Rehberdeki "Bölüm" başlıkları aynı numaraları kullanıyor; parantez içinde de o bölümün ödev yönergesindeki hangi maddeyi karşıladığı yazıyor (örneğin "Yönerge 7"). Kodu ve rehberi yan yana açıp birlikte okuyabilirsin.

> **Örnek çıktılar hakkında:** Rehberdeki ekran çıktıları bu bilgisayarda `main.py` çalıştırılarak alındı. Modelin başlangıç ağırlıkları her çalıştırmada rastgele seçildiği için senin loss ve accuracy değerlerin birkaç binde bir farklı çıkabilir (sebebi Bölüm 9'da).

**Önerilen okuma sırası:**

1. **Büyük resim** ve **Temel kavramlar**: Ne yaptığımızı ve bir görüntünün bilgisayarda nasıl tutulduğunu anla.
2. **Kodda geçen Python yapıları**: Kodda karşılaşacağın Python yazımlarına göz at.
3. **Kodun bölüm bölüm açıklaması**: `main.py`'yi baştan sona takip et.
4. **Tek fotoğraf demosu**: Modelin tek bir görüntü için nasıl karar verdiğini adım adım izle.
5. **Kendini test et**: Sözlü sınava hazırlık soruları. Yönerge, kodun temel bölümlerini sözlü olarak açıklayabilmeni istiyor.

---

## Büyük resim: Ne yapıyoruz?

Elimizde 70.000 küçük kıyafet fotoğrafı var. Her biri 10 kategoriden birine ait: tişört, pantolon, kazak, elbise, ceket, sandalet, gömlek, spor ayakkabı, çanta ve bot. Amacımız, **daha önce hiç görmediği** bir fotoğrafa bakıp hangi kategoriden olduğunu söyleyebilen bir program yazmak. Bu işe **görüntü sınıflandırma** denir.

Bunu klasik programlamayla, yani kuralları tek tek yazarak yapmak neredeyse imkânsız. "Pantolonun iki bacağı vardır" diye bir kural yazsan bile her pantolonun kalıbı, deseni ve parlaklığı farklı. Makine öğrenmesinde kuralları biz yazmayız: Modele binlerce **etiketli örnek** (fotoğraf + doğru cevap) gösteririz, kuralları model kendisi çıkarır.

Kullandığımız model bir **CNN** (Convolutional Neural Network, Türkçesiyle evrişimli sinir ağı). CNN, görüntülerdeki kenar, köşe ve doku gibi desenleri yakalamak için tasarlanmış bir yapay sinir ağı türüdür.

`main.py`'nin akışı:

```
PNG görüntüleri (klasörlerde)
   ↓  Bölüm 1-5    Sınıfları tanımla, görüntüleri oku, klasör adına göre etiketle
   ↓  Bölüm 6-7    Veri boyutlarını ve sınıf dağılımını kontrol et
   ↓  Bölüm 8      Normalize et (0-255 → 0-1)
   ↓  Bölüm 9      Eğitim verisini karıştır
   ↓  Bölüm 10     Kanal boyutu ekle (28×28 → 28×28×1)
   ↓  Bölüm 11-12  CNN modelini kur ve özetini yazdır
   ↓  Bölüm 13     Modeli derle (nasıl öğreneceğini belirle)
   ↓  Bölüm 14-15  Modeli eğit, her epoch'un sonuçlarını yazdır
   ↓  Bölüm 16-17  Loss ve accuracy grafiklerini çiz
   ↓  Bölüm 18     Test verisiyle değerlendir
   ↓  Bölüm 19-20  Tahmin yap, doğru ve yanlış tahminleri say
   ↓  Bölüm 21     Rastgele 10 test görüntüsünü tahminleriyle göster
      Bölüm 22     Tahminlerin sınıflara dağılımını say
```

Bu akış, yönergenin sonundaki "Beklenen Proje Akışı" şemasıyla aynı sırayı izliyor.

---

## Projeyi çalıştırma

### Klasör yapısı

```
FashionMnist/
├── main.py               ← projenin ana kodu: okuma, eğitim, test, grafikler
├── tek_foto_tahmin.py    ← kayıtlı modelle tek bir fotoğrafı adım adım tahmin eden demo
├── model.keras           ← eğitilmiş model dosyası (demo bunu kullanır)
├── REHBER.md             ← bu rehber
├── fashionmnsit ödev yönergesi.docx
├── screenshots/          ← teslim edilecek ekran görüntüleri buraya kaydedilecek
└── data/
    └── fashion_mnist/
        ├── train/        ← 60.000 eğitim görüntüsü, her sınıfta 6.000
        │   ├── Tişört/       image_00001.png ... image_06000.png
        │   ├── Pantolon/
        │   └── ...           (toplam 10 sınıf klasörü)
        └── test/         ← 10.000 test görüntüsü, her sınıfta 1.000
            └── ...           (aynı 10 sınıf klasörü; image_00001.png ... image_01000.png)
```

Yönergenin kuralına uygun olarak `main.py` veri setini internetten indirmez, görüntüleri yalnızca bu klasörlerden okur. Etiketler de bir CSV ya da Excel dosyasından değil, klasör adlarından gelir (Bölüm 3).

### Gerekli kütüphaneler

Projede ayrı bir sanal ortam (`.venv`) yok; kütüphaneler bilgisayardaki Python 3.12'ye kurulu. Bu bilgisayardaki sürümler:

| Kütüphane | Sürüm | Ne için kullanılıyor? |
|---|---|---|
| NumPy | 2.2.6 | Sayı dizileri (matrisler) |
| Matplotlib | 3.11.2 | Grafik çizme, görüntü gösterme |
| Pillow (PIL) | 12.3.0 | PNG dosyalarını açıp okuma |
| TensorFlow | 2.21.0 | Sinir ağının hesaplamaları |
| Keras | 3.15.1 | Modeli kurma ve eğitme arayüzü (TensorFlow ile birlikte kurulur) |

Başka bir bilgisayara kurman gerekirse:

```powershell
python -m pip install numpy matplotlib pillow tensorflow
```

> `pip install ...` yerine `python -m pip install ...` yazmak, paketlerin tam olarak `python` komutunun çalıştırdığı Python'a kurulmasını garanti eder. Bilgisayarda birden fazla Python kuruluysa bu fark önemlidir.

### Programı çalıştırma

**VS Code ile:** `main.py` açıkken sağ üstteki ▶ (Run Python File) butonuna bas. Sağ alttaki durum çubuğunda Python 3.12'nin seçili olduğundan emin ol; değilse `Ctrl+Shift+P` → **Python: Select Interpreter**.

**Terminal ile** (VS Code'da üst menüden Terminal → New Terminal):

```powershell
python main.py
```

> **Önemli:** Programı **proje klasöründeyken** çalıştır. Koddaki `DATA_DIR = "data/fashion_mnist"` göreli (relative) bir yoldur: Python bu yolu `main.py`'nin bulunduğu yere göre değil, **terminalin o an bulunduğu klasöre** göre arar (Bölüm 2). Başka bir klasörden çalıştırırsan `FileNotFoundError` alırsın. VS Code'da proje klasörünü açtıysan ▶ butonu bunu zaten doğru yapar.

### Çalışınca ne olur?

Bu bilgisayarda toplam süre 2 dakika civarında:

1. 70.000 PNG okunur ve her sınıftaki görüntü sayısı yazılır (yaklaşık 40 saniye).
2. Veri boyutları, sınıf dağılımları, CNN giriş boyutu ve model özeti yazılır.
3. Model 5 epoch eğitilir. Her epoch için bir ilerleme çubuğu görürsün (epoch başına 10-13 saniye).
4. Eğitim geçmişi (her epoch'un dört değeri) yazılır.
5. **Loss grafiği** penceresi açılır. **Program, sen pencereyi kapatana kadar bekler.**
6. Pencereyi kapatınca **accuracy grafiği** açılır; onu da kapat.
7. Test sonuçları ve doğru/yanlış tahmin sayıları yazılır.
8. **10 örnek tahmin** penceresi açılır; kapat.
9. Tahmin sınıf dağılımı yazılır ve program biter.

Programın grafiklerde durmasının sebebi her grafikten sonra çağrılan `plt.show()`: Bu komut, pencere kapanana kadar programı bekletir.

### Teslim için ekran görüntüleri

Yönergenin 19. maddesi `screenshots/` klasöründe beş görüntü istiyor. `main.py` grafikleri dosyaya kaydetmiyor, yalnızca pencerede gösteriyor; bu yüzden onları sen kaydetmelisin:

| Dosya | Nasıl alınır? |
|---|---|
| `loss.png` | Loss grafiği penceresinin alt çubuğundaki disket simgesine (Save the figure) tıkla, `screenshots` klasörüne `loss.png` adıyla kaydet |
| `accuracy.png` | Accuracy grafiği penceresinde aynı şekilde |
| `predictions.png` | 10 örnek tahmin penceresinde aynı şekilde |
| `model_summary.png` | Terminaldeki model özeti tablosunun (Bölüm 12) ekran görüntüsünü `Win+Shift+S` ile al |
| `test_results.png` | Terminaldeki `TEST SONUÇLARI` bölümünün (Bölüm 18) ekran görüntüsünü `Win+Shift+S` ile al |

Pencereyi kapatınca program bir sonraki adıma geçtiği için her grafiği **kapatmadan önce** kaydet.

### Başta görebileceğin mesajlar

Program çalışırken TensorFlow ve Keras bazı mesajlar yazar. Hiçbiri hata değildir:

| Mesaj | Anlamı |
|---|---|
| `oneDNN custom operations are on...` | TensorFlow, işlemcide bir hızlandırma kütüphanesi kullanıyor. Sonuçların son basamaklarında çok küçük yuvarlama farkları olabilir. |
| `UserWarning: Do not pass an input_shape/input_dim argument to a layer...` | Keras 3, giriş boyutunun `Conv2D(..., input_shape=...)` içinde değil, modelin başında ayrı bir `Input(shape=...)` ile verilmesini öneriyor. Yalnızca bir uyarı; model aynı şekilde kurulur ve çalışır (Bölüm 11). |
| `This TensorFlow binary is optimized to use available CPU instructions...` | İşlemcinin hızlı komutlarından yararlanıldığını söylüyor. |
| `TensorFlow GPU support is not available on native Windows...` | TensorFlow Windows'ta ekran kartını (GPU) kullanamaz; eğitim işlemcide (CPU) yapılır. Bu küçük model için CPU yeterli. |

---

## Temel kavramlar

### Bilgisayar bir görüntüyü nasıl "görür"?

Bilgisayar için bir görüntü, sayılardan oluşan bir tablodur (**matris**). Tablonun her hücresi bir **piksel**dir. Gri seviyeli bir görüntüde her piksel 0 ile 255 arasında tek bir sayıdır: 0 siyah, 255 beyaz, aradaki değerler grinin tonlarıdır.

Fashion-MNIST görüntüleri 28 satır × 28 sütundur, yani her görüntüde 28 × 28 = 784 piksel vardır. Aşağıda projedeki `train/Pantolon/image_00001.png` dosyasının piksellerini değerlerine göre farklı karakterlerle çizdim: `██` çok parlak (170'ten büyük), `▓▓` orta (101-170), `░░` soluk (31-100), boşluk siyaha yakın (0-30). Soldaki sayılar satır numarası:

```
 0 │                  ░░▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓▓██
 1 │                  ▓▓████████████████████
 2 │                  ▓▓▓▓██████████████▓▓██░░
 3 │                  ▓▓████████████████████░░
 4 │                  ██████████████████████▓▓
 5 │                  ██████████████████████▓▓
 6 │                  ██▓▓▓▓██████████████████
 7 │                  ██▓▓▓▓████████▓▓▓▓▓▓████
 8 │                ░░██▓▓▓▓████  ████▓▓▓▓████
 9 │                ░░██▓▓▓▓████  ████████████
10 │                ░░██▓▓▓▓▓▓██  ████▓▓██████
11 │                ░░██▓▓▓▓████  ▓▓██▓▓██████
12 │                ░░▓▓▓▓▓▓████  ▓▓██▓▓▓▓████
13 │                ░░▓▓▓▓▓▓████  ░░██▓▓▓▓████
14 │                ░░██▓▓▓▓████    ██▓▓▓▓████
15 │                ░░██▓▓▓▓██▓▓    ██▓▓▓▓████
16 │                ░░██▓▓▓▓██▓▓    ██▓▓▓▓██▓▓
17 │                ░░██▓▓████░░    ██▓▓▓▓██▓▓
18 │                ░░██▓▓████░░    ████▓▓██▓▓
19 │                ░░██▓▓████      ▓▓██████▓▓
20 │                ░░██▓▓████      ▓▓██████▓▓
21 │                ░░██▓▓████      ▓▓██████▓▓
22 │                ░░██▓▓████      ░░██████▓▓
23 │                ░░██▓▓██▓▓      ░░██████▓▓
24 │                ░░██▓▓██▓▓      ░░██████▓▓
25 │                ░░██▓▓██▓▓      ░░██████▓▓
26 │                ░░██████▓▓      ░░██████▓▓
27 │                ░░▓▓▓▓▓▓░░        ▓▓████▓▓
```

16-20. satırların 7-21. sütunlarındaki gerçek sayılar aşağıda (bu satırlarda diğer sütunların hepsi 0):

```
sütun →      7    8    9   10   11   12   13   14   15   16   17   18   19   20   21
satır 16     0   58  184  122  166  198  115    0    0  217  158  160  180  167    0
satır 17     0   58  204  111  172  203   79    0    0  204  164  155  188  169    0
satır 18     0   55  214  138  177  209   41    0    0  186  180  152  187  168    0
satır 19     0   55  214  149  177  210   23    0    0  158  199  173  192  163    0
satır 20     0   55  216  159  187  204    6    0    0  136  198  189  208  165    0
```

Sağ ve sol uçtaki 0'lar siyah arka plan, ortadaki iki sütunluk 0'lar pantolonun iki bacağı arasındaki boşluk, büyük sayılar ise kumaş. Model "pantolon" diye bir şey bilmez, yalnızca bu sayıları görür. Eğitim boyunca öğrendiği şey, "bu tür sayı desenleri genellikle pantolondur" bağlantısıdır.

> Fashion-MNIST'te arka plan siyah (0), kıyafet açık renklidir. Bu yüzden programın gösterdiği görüntülerde kıyafetler siyah zemin üzerinde beyaz görünür.

**Gri seviye ve renkli görüntü:** Renkli bir görüntüde her piksel 3 sayıdan oluşur: kırmızı (R), yeşil (G) ve mavi (B) miktarı. Bu sayıların her birine **kanal** denir. Gri seviyeli görüntünün tek kanalı vardır: parlaklık. Bizim görüntülerimiz tek kanallıdır.

### NumPy dizisi ve `shape`

**NumPy**, Python'da büyük sayı tablolarıyla hızlı çalışmayı sağlayan kütüphanedir. Bir NumPy dizisinin (array) `shape` özelliği, her boyutta kaç eleman olduğunu söyler:

| `shape` | Anlamı |
|---|---|
| `(10,)` | 10 sayılık tek bir sıra. Sondaki virgül dizinin tek boyutlu olduğunu gösterir. |
| `(28, 28)` | Tek bir görüntü: 28 satır, 28 sütun |
| `(60000, 28, 28)` | 60.000 tane 28×28 görüntü |
| `(60000, 28, 28, 1)` | 60.000 görüntü, her biri 28×28 ve 1 kanallı |
| `(10000, 10)` | 10.000 görüntünün her biri için 10 sınıf olasılığı (Bölüm 19) |

`(60000, 28, 28)` bir kitap gibi düşünülebilir: 60.000 sayfa, her sayfada 28×28'lik bir sayı tablosu.

### Train, validation, test: Neden üç ayrı veri?

Sınava hazırlanan bir öğrenci düşün:

- **Train (eğitim) verisi** ders kitabındaki çözümlü sorulardır. Öğrenci bunlarla çalışır.
- **Validation (doğrulama) verisi** deneme sınavıdır. Öğrenci bu sorularla çalışmaz, yalnızca "ne durumdayım?" diye kendini ölçer.
- **Test verisi** final sınavıdır. En sonda bir kez yapılır.

Model sınav sorularını önceden görürse cevapları ezberler ve aldığı not gerçek bilgisini yansıtmaz. Bu yüzden değerlendirme her zaman modelin **eğitimde görmediği** verilerle yapılır.

Bu projedeki dağılım:

```
train klasörü (60.000) ─┬─ 54.000 → eğitim: model bunlarla öğrenir
                        └─  6.000 → doğrulama: her epoch sonunda kontrol (validation_split=0.1)
test klasörü  (10.000) ─── test: eğitim tamamen bitince bir kez ölçülür
```

### Bir model nasıl öğrenir?

Model aslında çok sayıda ayarlanabilir sayıdan oluşur. Bunlara **ağırlık** (weight) ya da **parametre** denir; bizim modelimizde 347.146 tane var. Eğitim şöyle işler:

1. Başta ağırlıklar rastgeledir. Model tahmin yapar ve büyük ihtimalle yanılır.
2. **Loss** fonksiyonu tahminlerin ne kadar yanlış olduğunu tek bir sayıyla ölçer.
3. **Optimizer**, loss'u biraz azaltacak yönde her ağırlığı küçük bir miktar değiştirir.
4. Bu döngü binlerce kez tekrarlanır. Loss azaldıkça tahminler iyileşir.

Bunu sisli bir dağda en alçak noktayı aramaya benzetebilirsin. Etrafı göremezsin ama ayağının altındaki eğimi hissedersin; her seferinde en dik iniş yönüne küçük bir adım atarsın. Loss dağın yüksekliği, ağırlıklar bulunduğun konum, optimizer da adımları atan kişidir. Bu yöntemin adı **gradient descent** (gradyan inişi).

---

## Kodda geçen Python yapıları

Kodu okurken karşılaşacağın Python yazımlarının kısa açıklamaları. Örneklerin çoğu doğrudan `main.py`'den.

**`import`: kütüphane yükleme**

```python
import numpy as np                                # numpy'yi "np" kısa adıyla kullan
from PIL import Image                             # PIL kütüphanesinden yalnızca Image'ı al
from tensorflow.keras.layers import Conv2D, Dense # bir modülden birden fazla şey al
```

**Değişken, liste ve indeks**

```python
CLASS_NAMES = ["Tişört", "Pantolon", "Kazak"]  # liste: sıralı elemanlar
CLASS_NAMES[0]      # "Tişört" → sayma 0'dan başlar!
CLASS_NAMES[-1]     # "Kazak"  → eksi indeks sondan sayar
len(CLASS_NAMES)    # 3        → eleman sayısı
```

Kodda bazı değişkenler büyük harfle yazılmış (`CLASS_NAMES`, `DATA_DIR`, `TRAIN_DIR`). Bu Python'da bir gelenektir, zorunluluk değil: Program boyunca değişmeyecek **sabit ayarlar** büyük harfle yazılır.

**`for` döngüsü, `range` ve `enumerate`**

```python
for class_name in ["Tişört", "Pantolon"]:
    print(class_name)          # önce "Tişört", sonra "Pantolon" yazar

for i, class_name in enumerate(["Tişört", "Pantolon"]):
    print(i, class_name)       # önce "0 Tişört", sonra "1 Pantolon" yazar

list(range(5))                 # [0, 1, 2, 3, 4]
list(range(1, 6))              # [1, 2, 3, 4, 5] → bitiş değeri (6) dahil değil
```

`enumerate` her elemanı sıra numarasıyla birlikte verir. `main.py`'de bu sıra numarası sınıfın **etiketi** olarak kullanılıyor (Bölüm 3).

**Fonksiyon: `def` ve `return`**

```python
def kare_al(x):        # x: fonksiyona verilen girdi (parametre)
    return x * x       # return: fonksiyonun geri verdiği sonuç

kare_al(5)             # 25
```

Bir fonksiyon birden fazla değer döndürebilir: `return a, b` yazılır, çağıran taraf da `x, y = fonksiyon()` diye iki değişkene birden alır. `load_images_from_folder` böyle çalışıyor (Bölüm 3-5).

Python'da girintiler (satır başındaki boşluklar) önemlidir: Bir fonksiyonun, `for` döngüsünün ya da `if` bloğunun içindeki satırlar içeriden başlar. Girinti bitince blok da biter.

**List comprehension: kısa yoldan liste oluşturma**

`main.py`'deki şu yazım:

```python
files = sorted([
    file
    for file in os.listdir(class_folder)
    if file.lower().endswith(".png")
])
```

aşağıdaki uzun yazımla aynı işi yapar:

```python
files = []
for file in os.listdir(class_folder):    # klasördeki her dosya adı için
    if file.lower().endswith(".png"):    # adı .png ile bitiyorsa
        files.append(file)               # listeye ekle
files = sorted(files)                    # alfabetik sıraya koy
```

Köşeli parantezin içi şöyle okunur: "Klasördeki her `file` için, eğer `.png` ile bitiyorsa, `file`'ı listeye al."

**Metin (string) metotları**

```python
"IMAGE_00001.PNG".lower()               # 'image_00001.png' → hepsini küçük harfe çevir
"image_00001.png".endswith(".png")      # True → ".png" ile mi bitiyor?
```

**`if` / `else` ve karşılaştırma**

```python
if test_labels[i] == predicted_labels[i]:   # == : "eşit mi?" sorusu (True ya da False)
    result = "DOĞRU"
else:
    result = "YANLIŞ"
```

Tek `=` bir değişkene değer **atar**, çift `==` iki değeri **karşılaştırır**. `!=` "eşit değil mi?" demektir.

**f-string: değişkenleri metnin içine yerleştirme**

```python
class_name, count, oran = "Çanta", 6000, 0.90871
f"{class_name}: {count} görüntü"   # 'Çanta: 6000 görüntü'
f"{oran:.4f}"                      # '0.9087'  → virgülden sonra 4 basamak
f"%{oran * 100:.2f}"               # '%90.87'  → yüzdeye çevir, 2 basamak
f"{class_name:18s}|"               # 'Çanta             |' → 18 karakterlik alana sola yasla
f"{count:5d}"                      # ' 6000'   → 5 karakterlik alana sağa yasla (tam sayı)
```

Baştaki `%` işareti f-string için özel bir şey değildir, düz metindir; ekranda `%90.87` görünsün diye yazılmış. Hizalama (`18s`, `5d`) ekrandaki tabloların düzgün sütunlar hâlinde görünmesini sağlar.

**Uzun satırları bölme**

`main.py`'de birçok komut parantez içinde alt alta yazılmış:

```python
history = model.fit(

    train_images,

    train_labels,
    ...
)
```

Python, açık bir parantez kapanana kadar satır sonlarını ve boş satırları önemsemez; yukarıdaki yazım `model.fit(train_images, train_labels, ...)` ile aynıdır. Benzer şekilde parantez içinde alt alta yazılmış metinler otomatik olarak birleşir:

```python
title = (
    "Gerçek: Çanta\n"
    "Tahmin: Çanta"
)                     # 'Gerçek: Çanta\nTahmin: Çanta'   (\n: alt satıra geç)
```

**Sözlük (dictionary)**

```python
# history.history şuna benzer bir sözlüktür:
# {"loss": [0.4446, 0.3012, ...], "accuracy": [0.8454, 0.8927, ...],
#  "val_loss": [...], "val_accuracy": [...]}
history.history["loss"]       # "loss" anahtarıyla listeye ulaşılır
history.history["loss"][0]    # 0.4446 → ilk epoch'un loss değeri
```

**NumPy'ye özgü iki yazım**

```python
etiketler = np.array([0, 2, 2, 1, 2])
etiketler == 2             # [False, True, True, False, True] → her eleman ayrı ayrı karşılaştırılır
np.sum(etiketler == 2)     # 3 → True'lar 1, False'lar 0 sayılır; yani "kaç tane 2 var?"
```

```python
harfler = np.array(["A", "B", "C", "D"])
harfler[[2, 0, 3, 1]]      # ['C', 'A', 'D', 'B'] → bir indeks listesiyle elemanları yeniden dizme
```

İlki Bölüm 7, 20 ve 22'de saymak için, ikincisi Bölüm 9'da karıştırmak için kullanılıyor.

---

## Kodun bölüm bölüm açıklaması

### Başlangıç: kütüphaneler

```python
import os                          # Dosya ve klasör yollarıyla çalışma
import numpy as np                 # Sayı dizileri (matrisler) ile hızlı hesaplama
import matplotlib.pyplot as plt    # Grafik çizme ve görüntü gösterme
import tensorflow as tf            # Sinir ağı hesaplamalarını yapan motor

from PIL import Image              # PNG dosyalarını açıp okuma (Pillow kütüphanesi)
from tensorflow.keras.models import Sequential                            # Katmanları sırayla dizen model türü
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Modelin katmanları
```

`os` Python'la birlikte gelir; diğerleri sonradan kurulan kütüphanelerdir. Yönergede istenen kütüphanelerin hepsi burada: NumPy, Matplotlib, Pillow ve TensorFlow/Keras.

`tf` kısaltması `main.py`'de doğrudan kullanılmıyor; model ve katmanlar `tensorflow.keras` üzerinden içe aktarılıyor. O satırın durması zarar vermez. (`tek_foto_tahmin.py` ise modeli yüklerken `tf`'yi kullanıyor.)

> **Keras ile TensorFlow'un ilişkisi:** TensorFlow, sinir ağlarının matematiksel hesaplamalarını yapan motordur. Keras bu motoru kolayca kullanmamızı sağlayan arayüzdür. "Şu katmanları üst üste koy, şöyle eğit" demek için Keras'ı kullanırız, hesaplamaları arka planda TensorFlow yapar. `tensorflow.keras` yazınca TensorFlow'un içindeki Keras'a ulaşırız. Programın başında TensorFlow mesajlarını görmenin sebebi de bu.

---

### Bölüm 1: Sınıflar (Yönerge 5-A)

```python
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
```

Bu liste iki iş görür:

1. **Klasörleri bulmak:** Program `train/Tişört`, `train/Pantolon`, ... klasörlerinin adlarını bu listeden alır. Bu yüzden listedeki isimler klasör adlarıyla, Türkçe karakterler ve boşluklar dahil, **harfi harfine** aynı olmalı (`Tişört`, `Spor Ayakkabı`).
2. **Sayı ↔ isim çevirisi:** Model isimlerle değil sayılarla çalışır. Listedeki sıra numarası (indeks) sınıfın **etiketidir**:

| Etiket | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sınıf | Tişört | Pantolon | Kazak | Elbise | Ceket | Sandalet | Gömlek | Spor Ayakkabı | Çanta | Bot |

Çanta'nın etiketi 8'dir, çünkü 0'dan sayınca listede 8. sıradadır. Model "8" tahmin ettiğinde `CLASS_NAMES[8]` ile bunu "Çanta"ya çeviririz. Listedeki sıra, yönergedeki etiket tablosuyla birebir aynı.

---

### Bölüm 2: Veri klasörleri (Yönerge 3)

```python
DATA_DIR = "data/fashion_mnist"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
```

- **`os.path.join(...)`**: Yol parçalarını işletim sistemine uygun ayraçla birleştirir. Windows'ta sonuç `data/fashion_mnist\train` olur. Windows hem `/` hem `\` ayracını kabul ettiği için bu karışık görünen yol sorunsuz çalışır; Linux ve macOS'ta da `join` doğru ayracı (`/`) kullanır.
- **Göreli yol:** `"data/fashion_mnist"` bir sürücü harfiyle (`C:\...`) başlamadığı için **göreli** bir yoldur. Python onu, programın çalıştırıldığı andaki **çalışma klasörüne** (terminalin bulunduğu klasöre) ekleyerek arar:

```
Terminal C:\Projects\BilgisayarGörmesi\FashionMnist içindeyse:
  → C:\Projects\BilgisayarGörmesi\FashionMnist\data\fashion_mnist\train   ✓ bulunur

Terminal C:\Projects içindeyse:
  → C:\Projects\data\fashion_mnist\train                                  ✗ FileNotFoundError
```

Bu yüzden programı proje klasöründen çalıştırmak gerekiyor.

> **İpucu:** Yolu `main.py`'nin kendi konumundan hesaplamak (`os.path.dirname(os.path.abspath(__file__))` ile) programı her klasörden çalışır hâle getirir. Bu projede gerekli değil, ama "göreli yol neden sorun çıkarabilir?" diye sorulursa cevabı bu.

---

### Bölüm 3: PNG görüntülerini okuma (Yönerge 5-B)

Yönerge bu adım için bir fonksiyon istiyor: klasörü açacak, PNG dosyalarını bulacak, görüntüleri gri seviyede okuyacak, NumPy dizisine dönüştürecek, etiketleri oluşturacak ve görüntülerle etiketleri ayrı diziler olarak döndürecek.

```python
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
```

**Fonksiyonun mantığı, düz Türkçeyle:**

```
her sınıf için (etiket=0 "Tişört", etiket=1 "Pantolon", ...):
    klasör yolunu oluştur:  .../train/Tişört
    klasördeki .png dosyalarının adlarını bul ve sırala
    "Tişört: 6000 görüntü" yaz
    her PNG dosyası için:
        aç → gri seviyeye çevir → sayı matrisine dönüştür → images listesine ekle
        o klasörün etiketini labels listesine ekle
iki listeyi NumPy dizisine çevirip döndür
```

**Satır satır:**

- **`images = []`, `labels = []`**: İki boş liste. Her görüntü okunduğunda görüntü birinci listeye, etiketi ikinci listeye eklenir. İki liste **aynı sırayla** dolduğu için `images[i]` görüntüsünün etiketi her zaman `labels[i]`'dir. Bu eşleşme bütün projenin temelidir.
- **`for label, class_name in enumerate(CLASS_NAMES):`** Sınıfları sırayla gezer. İlk turda `label=0, class_name="Tişört"`, ikinci turda `label=1, class_name="Pantolon"` olur ve böyle devam eder.
- **`os.path.join(folder, class_name)`**: Sınıf klasörünün yolunu oluşturur, örneğin `data/fashion_mnist\train\Tişört`.
- **`os.listdir(class_folder)`**: Klasördeki bütün dosyaların **adlarını** liste olarak verir (tam yolunu değil). Klasör yoksa program burada `FileNotFoundError` ile durur.
- **`if file.lower().endswith(".png")`**: Yalnızca PNG dosyalarını alır; klasörde başka bir dosya (örneğin Windows'un oluşturduğu `Thumbs.db`) varsa atlanır. `.lower()` sayesinde `.PNG` gibi büyük harfli uzantılar da yakalanır.
- **`sorted(...)`**: `os.listdir` dosyaları belirli bir sırada vermeyi garanti etmez. `sorted` listeyi alfabetik sıraya koyar, böylece dosyalar her çalıştırmada aynı sırayla okunur. Dosya adlarındaki sayılar 5 haneye sıfırla tamamlandığı için (`image_00002`, `image_00010`) alfabetik sıra sayısal sırayla aynıdır. Sıfırlar olmasaydı `image_10.png` alfabetik olarak `image_2.png`'den önce gelirdi.
- **`print(f"{class_name}: {len(files)} görüntü")`**: Yönergenin istediği "her sınıfta kaç görüntü var" çıktısı. `len(...)` listenin eleman sayısını verir.
- **`Image.open(file_path).convert("L")`**: Pillow ile PNG dosyasını açar ve görüntüyü **gri seviyeye** çevirir. "L" (luminance, yani parlaklık) modunda her piksel 0-255 arasında tek bir sayıdır. Fashion-MNIST görüntüleri zaten gri olduğu için değerler değişmez. Görüntü renkli olsaydı her pikselin R, G, B değerleri tek bir parlaklık değerine  dönüştürülürdü (yaklaşık %30 kırmızı + %59 yeşil + %11 mavi). Bu satır, yönergedeki "görüntüleri gri seviyede okumalı" maddesini garanti eder.
- **`np.array(image, dtype=np.uint8)`**: Pillow görüntüsünü 28×28'lik bir NumPy sayı matrisine çevirir (Temel kavramlar bölümündeki tablo). `uint8`, 0-255 arası tam sayıları tutan veri tipidir: 8 bitlik işaretsiz tam sayı, yani her piksel 1 byte.
- **`labels.append(label)`**: **Etiketin klasör adından belirlendiği satır budur.** `label`, o anda okunan klasörün `CLASS_NAMES` listesindeki sıra numarasıdır. Örneğin `train/Çanta/` içindeki her dosya 8 etiketini alır. Yönergenin önemli kuralı (etiket CSV veya Excel'den değil, klasör adından gelmeli) böylece sağlanır.
- **`return np.array(images), np.array(labels)`**: Listeleri NumPy dizilerine çevirip döndürür. 60.000 tane 28×28 matristen oluşan liste, `(60000, 28, 28)` boyutunda tek bir üç boyutlu diziye dönüşür.

**Yönergedeki maddelerin koddaki karşılıkları:**

| Yönerge maddesi | Koddaki karşılığı |
|---|---|
| İlgili klasörü açmalı | `os.path.join(folder, class_name)` ve `os.listdir(class_folder)` |
| PNG dosyalarını bulmalı | `if file.lower().endswith(".png")` |
| Görüntüleri gri seviyede okumalı | `Image.open(file_path).convert("L")` |
| Görüntüleri NumPy dizisine dönüştürmeli | `np.array(image, dtype=np.uint8)` |
| Görüntünün sınıf etiketini oluşturmalı | `labels.append(label)` |
| Görüntüleri ve etiketleri ayrı diziler hâlinde döndürmeli | `return np.array(images), np.array(labels)` |

---

### Bölüm 4-5: Eğitim ve test verilerini okuma (Yönerge 5-B)

```python
train_images, train_labels = load_images_from_folder(TRAIN_DIR)
test_images, test_labels = load_images_from_folder(TEST_DIR)
```

Fonksiyon iki değer döndürdüğü için sonuç iki değişkene birden atanır. Aynı fonksiyon önce `train`, sonra `test` klasörü için çağrılıyor. Fonksiyon yazmanın faydası burada görülüyor: Aynı kodu iki kez yazmak yerine aynı fonksiyonu iki kez çağırıyoruz.

**Ekranda göreceğin:**

```
Eğitim görüntüleri okunuyor...
Tişört: 6000 görüntü
Pantolon: 6000 görüntü
...
Bot: 6000 görüntü

Test görüntüleri okunuyor...
Tişört: 1000 görüntü
...
Bot: 1000 görüntü
```

---

### Bölüm 6: Veri boyutları (Yönerge 6)

```python
print("Train images:", train_images.shape)
print("Train labels:", train_labels.shape)
print("Test images :", test_images.shape)
print("Test labels :", test_labels.shape)
```

**Ekranda göreceğin:**

```
Veri boyutları:
Train images: (60000, 28, 28)
Train labels: (60000,)
Test images : (10000, 28, 28)
Test labels : (10000,)
```

| Çıktı | Okunuşu |
|---|---|
| `Train images: (60000, 28, 28)` | 60.000 eğitim görüntüsü, her biri 28 satır × 28 sütun piksel |
| `Train labels: (60000,)` | 60.000 eğitim etiketi (her görüntüye bir tane); tek boyutlu bir sayı dizisi |
| `Test images : (10000, 28, 28)` | 10.000 test görüntüsü, her biri 28×28 piksel |
| `Test labels : (10000,)` | 10.000 test etiketi |

Kontrol etmen gereken şey: Görüntü sayısıyla etiket sayısı **eşit olmalı** (60.000 = 60.000 ve 10.000 = 10.000). Eşit olmasaydı bazı görüntülerin etiketi yok demekti.

> **Rapor için:** Yönerge bu boyutların ne anlama geldiğini raporda açıklamanı istiyor. Yukarıdaki tablo bunun için.

---

### Bölüm 7: Sınıf dağılımlarını kontrol etme

Bu bölüm yönergede ayrı bir madde değil; okunan verinin doğru olduğundan emin olmak için eklenmiş bir kontrol.

```python
for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(train_labels == i)

    print(f"{i}: {class_name:18s} -> {count}")
```

- **`train_labels == i`**: 60.000 etiketin her birini `i` ile karşılaştırır ve 60.000 tane `True`/`False` değerinden oluşan bir dizi verir.
- **`np.sum(...)`**: `True`'ları 1, `False`'ları 0 sayarak toplar; sonuç "etiketi `i` olan kaç görüntü var?" sorusunun cevabıdır.
- **`{class_name:18s}`**: Sınıf adını 18 karakterlik bir alana yazar, böylece oklar alt alta hizalanır.

**Ekranda göreceğin:**

```
Eğitim sınıf dağılımı:
0: Tişört             -> 6000
1: Pantolon           -> 6000
...
9: Bot                -> 6000
```

Test için de aynı tablo (her sınıfta 1000) yazılır. Her sınıfta eşit sayıda görüntü var; buna **dengeli veri seti** denir. Dengeli olması, modelin bir sınıfı diğerlerinden çok daha sık görüp tahminlerini o sınıfa kaydırmasını önler. Bir klasör boş kalsaydı ya da dosyalar yanlış klasöre kopyalanmış olsaydı, sorun bu tabloda hemen görünürdü.

---

### Bölüm 8: Normalizasyon (Yönerge 7)

Ön işleme, veriyi modelin rahat çalışabileceği biçime getirmektir. Projede iki ön işleme adımı var: normalizasyon (bu bölüm) ve kanal boyutu ekleme (Bölüm 10).

```python
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
```

- **`astype("float32")`**: Pikseller şu an 0-255 arasında tam sayı (`uint8`). Bölmenin sonucu ondalıklı olacağı için diziyi ondalıklı sayı tipine çeviriyoruz. `float32` seçilmesinin sebebi, TensorFlow'un varsayılan olarak bu tipi kullanması ve `float64`'ün yarısı kadar bellek tutması. 60.000 eğitim görüntüsü `uint8` olarak 47 MB, `float32` olarak yaklaşık 188 MB, `float64` olarak 376 MB yer kaplar.
- **`/ 255.0`**: Her pikseli 255'e böler. NumPy bu işlemi dizideki 47 milyon pikselin hepsine tek satırda uygular.

| Piksel (0-255) | 255'e bölününce (0-1) |
|---|---|
| 0 (siyah) | 0.0 |
| 51 | 0.2 |
| 128 | 0.502 |
| 255 (beyaz) | 1.0 |

**Neden normalize ediyoruz?** Modelin ağırlıkları başlangıçta küçük sayılardır. Girdiler 0-255 gibi büyük olursa ara hesaplamalar da çok büyür; ağırlık güncellemeleri sert ve dengesiz olur, eğitim yavaşlar ya da hiç ilerlemeyebilir. Girdiler 0-1 aralığındayken hesaplamalar dengeli kalır ve optimizer daha rahat çalışır.

Test verisine de **aynı işlem** uygulanıyor. Model eğitimde 0-1 aralığında sayılar gördüyse, test ederken ve yeni bir fotoğrafı tahmin ederken de (bkz. `tek_foto_tahmin.py`) 0-1 aralığında sayılar görmelidir.

---

### Bölüm 9: Veriyi karıştırma (Yönerge 8)

#### Sorun

Görüntüleri klasör klasör okuduğumuz için eğitim verisi sınıflara göre **sıralı**:

```
indeks:  0 ... 5999 | 6000 ... 11999 | ... | 48000 ... 53999 | 54000 ... 59999
sınıf :    Tişört   |    Pantolon    | ... |      Çanta      |       Bot
                                                              └── son %10 ───┘
                                                            validation_split=0.1
                                                                burayı ayırır
```

Bölüm 14'teki `validation_split=0.1`, doğrulama verisini **verinin sonundan** ayırır. Keras'ın kendi açıklaması: *"The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling."* (Doğrulama verisi, verilen `x` ve `y` verisinin son örneklerinden, karıştırmadan önce seçilir.)

60.000'in %10'u 6.000'dir ve her sınıfta tam 6.000 eğitim görüntüsü var. Yani veriyi karıştırmasaydık doğrulama verisi **Bot klasörünün tamamı** olurdu. Bunun sonuçları:

1. Model eğitimde **hiç** bot görmezdi; bot diye bir sınıfı tanımayı öğrenemezdi.
2. Doğrulama verisi yalnızca modelin hiç görmediği bu sınıftan oluşurdu. Validation accuracy sıfıra yakın çıkar ve modelin gerçek başarısı hakkında hiçbir şey söylemezdi.
3. Model 10 değil 9 sınıflı bir problem öğrenmiş olurdu; testteki 1.000 botun hepsini yanlış bilirdi.

`model.fit` içindeki `shuffle=True` bu sorunu **çözmez**. O ayar, ayırma işleminden **sonra** kalan eğitim kısmını her epoch başında kendi içinde karıştırır; doğrulama kısmı ise karıştırmadan önce, verinin sonundan ayrılmış olur.

#### Çözüm

```python
np.random.seed(42)

indices = np.random.permutation(len(train_images))

train_images = train_images[indices]
train_labels = train_labels[indices]
```

- **`np.random.permutation(60000)`**: 0'dan 59999'a kadar sayıları rastgele bir sırayla içeren bir dizi üretir, örneğin `[12628, 37730, 39991, 8525, ...]`.
- **`dizi[indices]`**: NumPy'da bir diziye köşeli parantez içinde bir indeks listesi verirsen, elemanları o sırayla dizilmiş yeni bir dizi alırsın.

Küçük bir örnek:

```
images  = [A, B, C, D]      (A'nın etiketi 0, B'nin 1, C'nin 2, D'nin 3)
labels  = [0, 1, 2, 3]
indices = [2, 0, 3, 1]

images[indices]  →  [C, A, D, B]
labels[indices]  →  [2, 0, 3, 1]    ✓ C hâlâ 2 ile, A hâlâ 0 ile eşleşiyor
```

**Neden ikisi de aynı `indices` ile?** Görüntüleri ve etiketleri ayrı ayrı rastgele karıştırsaydık eşleşme bozulurdu: C görüntüsünün yanına 0 etiketi gelebilirdi. Model "bu pantolon bir tişörttür" gibi yanlış bilgilerle eğitilir ve hiçbir şey öğrenemezdi. Yönergedeki "aynı indeksler kullanılarak" ifadesi bunu anlatıyor.

#### Bizim verimizde ne değişti?

`main.py` bunu ekrana yazmıyor; aşağıdaki değerler aynı tohumla (42) ve aynı okuma sırasıyla ayrıca hesaplandı:

```
Karıştırmadan ÖNCE
  İlk 10 etiket                        : [0 0 0 0 0 0 0 0 0 0]   (hepsi Tişört)
  Son 6.000'deki (validation) sınıflar : yalnızca 9 (Bot)
Karıştırdıktan SONRA
  İlk 10 etiket                        : [2 6 6 1 1 8 2 2 1 5]
  Son 6.000'deki (validation) sınıflar : 10 sınıfın hepsi, her sınıftan 563-643 görüntü
```

Kendin görmek istersen Bölüm 9'daki karıştırma satırlarının hemen öncesine ve hemen sonrasına şu iki satırı ekleyip programı çalıştırabilirsin:

```python
print("İlk 10 etiket:", train_labels[:10])
print("Validation'a düşecek sınıflar:", np.unique(train_labels[-6000:]))
```

`train_labels[:10]` ilk 10 elemanı, `train_labels[-6000:]` son 6.000 elemanı verir (dilimleme); `np.unique` içindeki **farklı** değerleri listeler.

Test verisini karıştırmıyoruz. Test verisi bölünmüyor ve eğitimde kullanılmıyor, bu yüzden sırası sonucu etkilemez.

#### Rastgelelik tohumu (seed): `np.random.seed(42)`

Bilgisayardaki "rastgele" sayılar aslında bir başlangıç değerinden (**tohum**) hesaplanır; tohum aynı olursa aynı sayı dizisi üretilir. `np.random.seed(42)` sayesinde karıştırma her çalıştırmada **aynı** sırayla yapılır (ilk 10 etiket hep `[2 6 6 1 1 8 2 2 1 5]` olur). 42'nin özel bir anlamı yok, herhangi bir sayı olabilirdi.

Ama dikkat: Bu satır yalnızca **NumPy'nin** rastgeleliğini sabitler. Modelin başlangıç ağırlıklarını ve `fit`'in her epoch'taki karıştırmasını Keras ve TensorFlow kendi rastgele sayı üreteçleriyle yapar; `np.random.seed` bunları etkilemez. Bu yüzden `main.py`'yi her çalıştırdığında loss ve accuracy değerleri biraz farklı çıkar. Her şeyi sabitlemek istenseydi `tf.keras.utils.set_random_seed(42)` kullanılabilirdi; bu tek satır Python'un, NumPy'nin ve TensorFlow'un tohumlarını birlikte ayarlar.

> **Rapor için:** Yönerge bu adımın neden gerekli olduğunu, özellikle `validation_split` kullanırken veri sırasının sonuçları nasıl etkileyebileceğini açıklamanı istiyor. "Karıştırmasaydık doğrulama verisinin tamamı Bot olurdu" örneği, açıklamanı destekleyen somut bir kanıt.

---

### Bölüm 10: Kanal boyutu ekleme (Yönerge 7)

```python
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
```

Conv2D katmanı her görüntüyü (yükseklik, genişlik, kanal) biçiminde bekler. Bizim görüntülerimiz `(28, 28)` biçiminde; kanal bilgisi eksik. `np.expand_dims(..., axis=-1)` dizinin **sonuna** (`axis=-1` "son konum" demektir) boyutu 1 olan yeni bir boyut ekler:

```
(60000, 28, 28)  →  (60000, 28, 28, 1)
```

Değerler değişmez, yalnızca "paketlenme" biçimi değişir. Küçük bir örnek:

```
Önce: shape (2, 2)       Sonra: shape (2, 2, 1)
[[0.1, 0.5],             [[[0.1], [0.5]],
 [0.9, 0.0]]              [[0.9], [0.0]]]
```

Her piksel artık "tek kanallı bir değer listesi" olarak tutuluyor. Görüntü renkli olsaydı her pikselde 3 değer olurdu ve şekil `(28, 28, 3)` olurdu. Yönergedeki "beklenen görüntü biçimi 28 × 28 × 1" maddesi bu.

**Ekranda göreceğin:**

```
CNN giriş boyutu:
Train: (60000, 28, 28, 1)
Test : (10000, 28, 28, 1)
```

> **Sıra hakkında:** Kodda normalizasyon (Bölüm 8), karıştırma (Bölüm 9) ve kanal ekleme (Bölüm 10) bu sırayla yapılıyor. Bu üç işlem birbirini etkilemediği için sıraları değişse de sonuç aynı olurdu. Önemli olan, görüntülerle etiketlerin aynı indekslerle karıştırılması ve test verisine de eğitim verisiyle aynı ön işlemenin uygulanması.

> **Rapor için:** Yönerge ön işleme bölümünde gri seviye dönüşümünü (Bölüm 3), normalizasyonu (Bölüm 8), kanal boyutunu (Bölüm 10) ve veri karıştırmayı (Bölüm 9) açıklamanı istiyor.

---

### Bölüm 11: CNN modeli (Yönerge 9)

#### Neden CNN?

Yalnızca Dense katmanlarından oluşan basit bir sinir ağı, görüntüyü 784 sayılık düz bir liste olarak görür; hangi pikselin hangisinin yanında olduğunu bilmez. CNN ise küçük bir pencereyle (filtre) görüntünün üzerinde gezinir ve **yan yana duran piksellerin oluşturduğu desenleri** (kenar, köşe, doku) yakalar. Aynı filtre görüntünün her yerinde kullanıldığı için bir desen görüntünün neresinde olursa olsun tanınabilir.

#### Model kodu

Kodda her katmanın üstünde başlık yorumları var. Burada yapı net görünsün diye yorumları ve boş satırları çıkardım:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
```

`Sequential([...])` katmanları listedeki sırayla arka arkaya bağlar: Bir katmanın çıktısı bir sonrakinin girdisi olur. Yapı yönergedeki şemayla aynı: Input → Conv2D → MaxPooling2D → Flatten → Dense → Output.

#### Verinin şekli katmanlardan geçerken nasıl değişiyor?

```
Girdi             (28, 28, 1)   tek kanallı 28×28 görüntü
  ↓ Conv2D        (26, 26, 32)  32 filtre → 32 özellik haritası
  ↓ MaxPooling2D  (13, 13, 32)  her harita yarı boyuta iner
  ↓ Flatten       (5408,)       13 × 13 × 32 = 5408 sayılık düz vektör
  ↓ Dense         (64,)         64 nöron
  ↓ Dense (çıkış) (10,)         10 sınıfın olasılıkları
```

#### Input (giriş)

Yönergedeki şemanın ilk kutusu olan **Input**, bizim kodda ayrı bir satır değil; ilk katmana verilen `input_shape=(28, 28, 1)` ile tanımlanıyor. Bu, modele "sana 28×28×1 boyutunda görüntüler gelecek" bilgisini verir. Keras bu bilgiyle arka planda bir giriş katmanı oluşturur (kayıtlı `model.keras` dosyasının içinde de `InputLayer` olarak görünüyor). Giriş katmanı hesaplama yapmadığı için model özetinde ayrı bir satır olarak görünmez.

Program çalışırken gördüğün `Do not pass an input_shape...` uyarısı buradan geliyor. Keras 3 aynı şeyin şöyle yazılmasını öneriyor:

```python
from tensorflow.keras import Input

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    ...
])
```

İki yazım aynı modeli kurar; mevcut kod da doğru çalışır.

#### Conv2D (evrişim katmanı)

`Conv2D(32, (3, 3), activation="relu", ...)`: 32 tane 3×3'lük filtre; aktivasyon fonksiyonu ReLU.

**Filtre nedir?** 3×3'lük küçük bir sayı tablosudur. Görüntünün sol üst köşesine konur, altındaki 9 pikselle karşılıklı çarpılır, sonuçlar toplanır ve toplama bir **bias** (sabit sayı) eklenir. Çıkan sayı, çıktı haritasının bir hücresi olur. Sonra filtre bir piksel sağa kayar ve işlem tekrarlanır; satır bitince bir alt satıra geçilir. Böylece görüntünün tamamı taranır.

Elle hazırlanmış bir filtreyle, Temel kavramlar bölümündeki pantolonun gerçek pikselleri üzerinde bir örnek. Aşağıdaki filtre, solu koyu sağı açık bölgelerde, yani dikey kenarlarda büyük bir sayı üretir. Görüntü parçası 16-18. satırlar ile 14-16. sütunların 255'e bölünmüş değerleri; bacakların arasındaki boşluktan sağ bacağa geçilen kenar:

```
 Görüntü parçası        Filtre            Karşılıklı çarpım
 0.00  0.00  0.85       -1   0   1        0.00  0.00  0.85
 0.00  0.00  0.80   ×   -1   0   1   →    0.00  0.00  0.80
 0.00  0.00  0.73       -1   0   1        0.00  0.00  0.73
```

Toplam = 0.85 + 0.80 + 0.73 = **2.38**, yani güçlü bir sinyal: "burada dikey bir kenar var". Aynı filtre siyah arka planda (dokuz değerin hepsi 0) gezerse sonuç **0** olur: "burada kenar yok". Sol bacaktan boşluğa geçilen kenarda (12-14. sütunlar, soldan sağa açıktan koyuya) ise sonuç yaklaşık **−2.39** çıkar. Yani bu filtre yalnızca bir yöndeki kenarları yakalar; ters yöndeki kenarları başka bir filtre yakalar.

Bizim modelimizde filtreleri biz yazmıyoruz. Filtrelerin içindeki sayılar da birer ağırlıktır: Başta rastgeledir ve eğitim sırasında **öğrenilir**. Eğitim sonunda her filtre işe yarayan bir deseni (yatay kenar, dikey kenar, köşe, doku...) yakalayacak hâle gelir. 32 filtre, 32 farklı desen demektir.

**Çıktı neden 26×26?** 3×3'lük bir filtre, görüntünün dışına taşmadan 28 piksellik bir kenar boyunca 28 − 3 + 1 = 26 farklı konuma yerleşebilir. Yatayda 26, dikeyde 26 konum olduğu için her filtre 26×26'lık bir **özellik haritası** (feature map) üretir. 32 filtreyle çıktı `(26, 26, 32)` olur.

**ReLU aktivasyonu:** Her hücrenin değeri negatifse 0 yapılır, pozitifse olduğu gibi bırakılır: `ReLU(x) = max(0, x)`. Örneğin ReLU(2.38) = 2.38, ReLU(−2.39) = 0.

Neden gerekli? Aktivasyon fonksiyonu olmasaydı her katman yalnızca çarpma ve toplama yapardı. Üst üste kaç katman koyarsan koy, sonuç tek bir büyük "çarp ve topla" işlemine eşit olurdu ve model yalnızca çok basit (doğrusal) ilişkileri öğrenebilirdi. ReLU gibi doğrusal olmayan bir fonksiyon, modelin karmaşık desenleri öğrenebilmesini sağlar.

**Parametre sayısı: 320.** Her filtrede 3 × 3 = 9 ağırlık ve 1 bias, yani 10 öğrenilebilir sayı vardır. 32 filtre × 10 = **320**. (Görüntü tek kanallı olduğu için filtreler 3×3×1'dir. Renkli görüntüde her filtre 3×3×3 = 27 ağırlık içerirdi.)

#### MaxPooling2D (havuzlama katmanı)

`MaxPooling2D((2, 2))`: Her özellik haritasını 2×2'lik bloklara böler ve her bloktan **en büyük** değeri alır:

```
     4×4 giriş               2×2 çıkış
  1   3 │ 2   0
  5   6 │ 1   2      →        6   2
 ───────┼───────              8   4
  7   8 │ 0   4
  2   1 │ 3   1
```

- **Boyut yarıya iner:** 26×26 → 13×13. Sonraki katmanların işi ve parametre sayısı azalır.
- **En güçlü sinyal korunur:** "Bu bölgede kenar var mı?" sorusunun cevabı korunur, kenarın tam olarak hangi pikselde olduğu bilgisi atılır. Bu, desen bir iki piksel kaysa bile modelin onu tanımasına yardım eder.
- **Parametresi yoktur (0):** Öğreneceği bir şey yoktur, yalnızca "en büyüğü al" kuralını uygular.

#### Flatten (düzleştirme)

`Flatten()`: `(13, 13, 32)` boyutundaki üç boyutlu çıktıyı tek bir sıraya dizer: 13 × 13 × 32 = **5408** sayı. Sonraki Dense katmanı tek boyutlu girdi beklediği için bu adım gerekli. Yalnızca şekil değiştirir, parametresi **0**'dır.

Benzetme: Her sayfasında 13×13'lük bir tablo olan 32 sayfalık bir defterdeki bütün sayıları tek bir uzun satıra yazmak gibi.

#### Dense (tam bağlantılı katman)

`Dense(64, activation="relu")`: 64 nöronlu katman. "Dense" (yoğun, tam bağlantılı) ifadesi, 64 nöronun **her birinin** 5408 girdinin **hepsine** bağlı olduğunu anlatır. Her nöron her girdiyi kendi ağırlığıyla çarpar, sonuçları toplar, bias ekler ve ReLU uygular.

Conv2D katmanı "görüntünün neresinde hangi desen var" bilgisini çıkardı. Dense katmanı bu desenleri birleştirerek daha üst düzey ilişkiler öğrenir; örneğin "iki uzun dikey parça + üstte yatay bir bant → pantolon".

**Parametre sayısı:** 5408 × 64 ağırlık + 64 bias = **346.176**. Modelin parametrelerinin %99,7'si bu katmanda.

#### Çıkış katmanı ve softmax

`Dense(10, activation="softmax")`: 10 sınıf olduğu için 10 nöron, her sınıf için bir tane. Parametre sayısı: 64 × 10 + 10 = **650**.

**Softmax**, 10 nöronun ham çıktılarını (skorları) **toplamı 1 olan olasılıklara** çevirir; büyük skor büyük olasılık alır. Her skorun e üssü alınır (e ≈ 2,718) ve bu sayılar toplamlarına bölünür. Böylece hepsi pozitif olur ve toplamları 1 eder. Üç sınıflı küçük bir örnek:

```
Ham skorlar :  Tişört 2.0    Pantolon 1.0    Kazak 0.1
Softmax     :  Tişört 0.66   Pantolon 0.24   Kazak 0.10     (toplam = 1.00)
```

Çok sınıflı sınıflandırmada çıkışta softmax kullanılır, çünkü her görüntü **tam olarak bir** sınıfa aittir ve istediğimiz şey "hangi sınıf, hangi olasılıkla?" sorusunun cevabıdır. Yönergedeki "çıkış katmanında sınıflandırma için uygun aktivasyon fonksiyonu" ifadesi bunu kastediyor.

---

### Bölüm 12: Model özeti (Yönerge 9)

```python
model.summary()
```

`model.summary()` modelin özetini yazdırır. `model_summary.png` için bu tablonun ekran görüntüsünü alacaksın:

```
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 5408)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │       346,176 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 347,146 (1.32 MB)
 Trainable params: 347,146 (1.32 MB)
 Non-trainable params: 0 (0.00 B)
```

- **Layer (type):** Katmanın adı ve türü. İsimleri Keras kendisi verir; iki Dense katmanı aynı adı alamayacağı için ikincisi `dense_1` olmuş.
- **Output Shape:** Katmanın çıktı boyutu. Baştaki `None` **batch boyutudur**: Model aynı anda kaç görüntü işleyeceğini önceden bilmez (eğitimde 64, tek fotoğraf demosunda 1), bu yüzden o boyut boş bırakılır.
- **Param #:** Katmanın öğrenilebilir sayılarının (ağırlık + bias) adedi.
- **Total params: 347,146 (1.32 MB):** Eğitim sırasında ayarlanan toplam sayı adedi ve bunların bellekte kapladığı yer (her biri 4 byte'lık bir `float32`).

**Parametre hesabı:**

| Katman | Hesap | Parametre |
|---|---|---|
| Conv2D | (3 × 3 × 1 + 1) × 32 | 320 |
| MaxPooling2D | öğrenilecek bir şey yok | 0 |
| Flatten | yalnızca şekil değiştirir | 0 |
| Dense | 5408 × 64 + 64 | 346.176 |
| Çıkış (Dense) | 64 × 10 + 10 | 650 |
| **Toplam** | | **347.146** |

Modelimiz, yönergenin rapor bölümündeki örnek mimari tablosuyla birebir aynı: Conv2D 32 filtre 3×3, MaxPooling 2×2, Flatten, Dense 64, Output 10 sınıf.

> **Rapor için:** "CNN Mimarisi" bölümündeki katman tablosunu bu özetten ve yukarıdaki hesaptan hazırlayabilirsin.

---

### Bölüm 13: Modeli derleme (Yönerge 10)

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

`compile`, modele **nasıl öğreneceğini** söyler. Üç şey belirlenir: hatanın nasıl ölçüleceği (loss), ağırlıkların nasıl güncelleneceği (optimizer) ve eğitim sırasında neyin takip edileceği (metrik).

#### Loss: `sparse_categorical_crossentropy`

**Cross-entropy (çapraz entropi)**, modelin **doğru sınıfa** verdiği olasılığa bakar:

```
loss = −ln(doğru sınıfın olasılığı)
```

| Model doğru sınıfa ne kadar olasılık verdi? | Loss |
|---|---|
| 0.99 (emin ve haklı) | 0.01 |
| 0.90 | 0.11 |
| 0.50 (kararsız) | 0.69 |
| 0.10 (10 sınıf arasında rastgele tahmin seviyesi) | 2.30 |
| 0.01 (emin ama yanılıyor) | 4.61 |

Doğru sınıfa ne kadar yüksek olasılık verilirse loss o kadar küçük olur. "Emin ama yanlış" tahminler çok ağır cezalandırılır.

**Güzel bir sağlama:** Henüz eğitilmemiş bir model 10 sınıfa yaklaşık eşit olasılık (0.1) verir, bu yüzden eğitimin en başında loss ≈ 2.30 olmalıdır. Bizim çalıştırmamızda eğitimin ilk adımındaki loss **2.29** idi.

**Neden "sparse"?** Etiketlerimiz tam sayı: Çanta = 8. Bu biçimle çalışan versiyon `sparse_categorical_crossentropy`'dir. Etiketler **one-hot** biçiminde yazılsaydı (Çanta = `[0,0,0,0,0,0,0,0,1,0]`: 10 elemanlı, yalnızca kendi sırası 1) `categorical_crossentropy` kullanılırdı. İkisi aynı hesabı yapar, yalnızca etiketin yazılış biçimi farklıdır.

**Neden bu loss fonksiyonu?** Problemimiz **çok sınıflı sınıflandırma** (10 sınıf, her görüntü tek bir sınıfa ait). Çıkış katmanı softmax ile olasılık üretiyor; cross-entropy de tam olarak olasılık tahminlerinin kalitesini ölçmek için tasarlanmıştır. Karşılaştırma için: İki sınıflı problemlerde `binary_crossentropy`, sayı tahmin eden problemlerde (ör. ev fiyatı) `mse` (ortalama kare hata) kullanılır.

#### Optimizer: Adam

Temel kavramlardaki "sisli dağdan iniş" benzetmesini hatırla: Optimizer adımları atan kişidir.

- En basit yöntem olan SGD (stochastic gradient descent), bütün ağırlıklar için aynı sabit adım büyüklüğünü kullanır.
- **Adam** (Adaptive Moment Estimation) her ağırlık için adım büyüklüğünü kendisi ayarlar: Hep aynı yöne itilen ağırlıklarda hızlanır, inişli çıkışlı (gürültülü) değişen ağırlıklarda daha temkinli adımlar atar.
- Varsayılan öğrenme hızı (learning rate) 0.001'dir ve çoğu problemde ayar gerektirmeden iyi çalışır. En yaygın tercih olmasının sebebi budur.

#### Metrik: accuracy

**Accuracy (doğruluk)** = doğru tahmin sayısı ÷ toplam tahmin sayısı. 10.000 test görüntüsünden 9.087'si doğru bilinirse accuracy = 0.9087, yani %90,87'dir.

**Loss ile accuracy'nin farkı:** Accuracy yalnızca "en yüksek olasılıklı sınıf doğru mu?" diye bakar. Loss ise modelin ne kadar **emin** olduğunu da hesaba katar. Model doğru sınıfa 0.51 de verse 0.99 da verse accuracy açısından ikisi de "doğru"dur, ama loss ikincisinde çok daha düşüktür. Model loss'u azaltarak öğrenir. Accuracy eğitimi etkilemez; sonucu insanların kolayca anlayacağı biçimde raporlamak için takip edilir.

> **Rapor için:** Yönergenin bu bölümdeki dört sorusu: Kullanılan optimizer nedir (Adam)? Kullanılan loss fonksiyonu nedir (`sparse_categorical_crossentropy`)? Neden bu loss fonksiyonu seçildi? Accuracy neyi ifade ediyor? Cevaplar yukarıda; raporda kendi cümlelerinle yaz.

---

### Bölüm 14: Modeli eğitme (Yönerge 11)

Boş satırları çıkarılmış hâli:

```python
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)
```

| Parametre | Değer | Anlamı |
|---|---|---|
| `train_images`, `train_labels` | | Girdiler (görüntüler) ve doğru cevaplar (etiketler) |
| `epochs` | 5 | Eğitim verisi modele baştan sona 5 kez gösterilir |
| `batch_size` | 64 | Her ağırlık güncellemesinde 64 görüntüye bakılır |
| `validation_split` | 0.1 | Verinin son %10'u (6.000 görüntü) eğitimde kullanılmaz, doğrulama için ayrılır |
| `shuffle` | True | Kalan 54.000 eğitim görüntüsü her epoch başında yeniden karıştırılır. Bölüm 9'daki sorunu çözmez, ama modelin her epoch'ta görüntüleri farklı bir sırayla görmesini sağlar. |

#### Epoch, batch ve step

- 54.000 eğitim görüntüsü 64'erli gruplara (**batch**) bölünür: 54.000 ÷ 64 = 843,75. Yani 843 tam grup ve 48 görüntülük son bir grup; toplam **844 grup**.
- Her grup için model 64 görüntüye tahmin yapar, ortalama loss hesaplanır ve optimizer ağırlıkları bir kez günceller. Buna bir **step** (adım) denir.
- 844 adım bitince model eğitim verisinin tamamını bir kez görmüş olur. Buna bir **epoch** denir.
- Toplamda 5 epoch × 844 adım = 4.220 ağırlık güncellemesi yapılır.

Neden bütün veriye bir kerede bakmıyoruz? 54.000 görüntü üzerinden tek seferde hesap yapmak hem çok bellek ister hem de epoch başına yalnızca 1 güncelleme demektir; öğrenme çok yavaş olur. Her görüntüden sonra güncellemek ise tek bir örneğe bakıldığı için çok gürültülüdür. Küçük gruplar (batch) bu ikisi arasında dengeli bir yoldur.

Yönerge en az 5 epoch istiyor; kodumuz tam 5 epoch eğitiyor. `epochs=5` değerini değiştirerek farklı değerleri deneyebilirsin (Bölüm 16-17'deki "Kendin dene" kutusuna bak).

#### Ekranda göreceğin satır

Her epoch için ilerleme çubuğu aynı satırda güncellenir. Epoch bitince satır şöyle görünür:

```
Epoch 1/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 13s 13ms/step - accuracy: 0.8454 - loss: 0.4446 - val_accuracy: 0.8870 - val_loss: 0.3261
```

| Parça | Anlamı |
|---|---|
| `Epoch 1/5` | 5 epoch'tan birincisi |
| `844/844` | 844 adımın 844'ü tamamlandı |
| `13s 13ms/step` | Epoch 13 saniye sürdü, adım başına yaklaşık 13 milisaniye |
| `accuracy`, `loss` | **Eğitim** verisindeki doğruluk ve loss (Training Accuracy, Training Loss) |
| `val_accuracy`, `val_loss` | **Doğrulama** verisindeki doğruluk ve loss (Validation Accuracy, Validation Loss) |

Yönergede takip edilmesi istenen dört değer bunlardır.

#### Bir adımın içinde neler oluyor?

1. **İleri yayılım (forward pass):** 64 görüntü modelden geçer, 64 tahmin çıkar.
2. **Loss hesabı:** Tahminler doğru etiketlerle karşılaştırılır.
3. **Geri yayılım (backpropagation):** Her ağırlığın loss'u ne yönde ve ne kadar etkilediği (gradyan) hesaplanır.
4. **Güncelleme:** Adam her ağırlığı loss'u azaltacak yönde biraz değiştirir.

Bunların hepsini `model.fit` bizim yerimize yapar.

---

### Bölüm 15: Eğitim geçmişi (Yönerge 11)

`fit` bir `history` nesnesi döndürür. `history.history` sözlüğünde dört metriğin epoch epoch değerleri liste olarak durur: `"loss"`, `"accuracy"`, `"val_loss"`, `"val_accuracy"`. Kod bunları, yönergenin istediği gibi, eğitim bitince ekrana yazdırır:

```python
for epoch in range(len(history.history["loss"])):

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={history.history['loss'][epoch]:.4f}, "
        f"Accuracy={history.history['accuracy'][epoch]:.4f}, "
        f"Val Loss={history.history['val_loss'][epoch]:.4f}, "
        f"Val Accuracy={history.history['val_accuracy'][epoch]:.4f}"
    )
```

- `len(history.history["loss"])` epoch sayısıdır (5); `range(5)` 0'dan 4'e kadar sayar. Ekranda epoch numarası 1'den başlasın diye `epoch + 1` yazdırılır.
- f-string'ler çift tırnakla yazıldığı için içerideki sözlük anahtarları tek tırnakla yazılmış (`['loss']`). İkisi de çift tırnak olsaydı Python metnin nerede bittiğini karıştırabilirdi.
- Beş parça metin parantez içinde alt alta yazıldığı için tek bir metin olarak birleşir (Python yapıları bölümüne bak).

**Ekranda göreceğin** (bizim çalıştırmamızdaki değerler):

```
Epoch 1: Loss=0.4446, Accuracy=0.8454, Val Loss=0.3261, Val Accuracy=0.8870
Epoch 2: Loss=0.3012, Accuracy=0.8927, Val Loss=0.2895, Val Accuracy=0.8993
Epoch 3: Loss=0.2576, Accuracy=0.9074, Val Loss=0.2751, Val Accuracy=0.9045
Epoch 4: Loss=0.2278, Accuracy=0.9177, Val Loss=0.2606, Val Accuracy=0.9090
Epoch 5: Loss=0.2023, Accuracy=0.9261, Val Loss=0.2370, Val Accuracy=0.9187
```

---

### Bölüm 16-17: Loss ve accuracy grafikleri (Yönerge 12-13)

Loss grafiğinin kodu (boş satırlar çıkarıldı):

```python
epochs = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, history.history["loss"], marker="o", label="Eğitim Loss")
plt.plot(epochs, history.history["val_loss"], marker="o", label="Doğrulama Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim ve Doğrulama Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

| Satır | Ne yapar? |
|---|---|
| `epochs = range(1, ... + 1)` | X ekseni için 1, 2, 3, 4, 5 sayıları. `+ 1` gerekiyor, çünkü `range(1, 6)` bitiş değeri olan 6'yı dahil etmez. (Buradaki `epochs` bir sayı değil, bir sayı aralığı.) |
| `plt.figure(figsize=(8, 5))` | 8×5 inç boyutunda yeni, boş bir grafik açar |
| `plt.plot(x, y, marker="o", label="...")` | X ekseninde epoch'lar, Y ekseninde loss değerleri olan bir çizgi çizer. `marker="o"` her noktaya yuvarlak işaret koyar, `label` lejantta görünecek isimdir |
| `plt.xlabel`, `plt.ylabel`, `plt.title` | Eksen isimleri (X: Epoch, Y: Loss) ve grafik başlığı |
| `plt.xticks(epochs)` | X eksenine yalnızca 1, 2, ..., 5 işaretlerini koyar (yoksa matplotlib 1.5, 2.5 gibi ara değerler gösterebilir) |
| `plt.legend()` | Hangi çizginin hangisi olduğunu gösteren kutuyu (lejant) ekler |
| `plt.grid(True)` | Okumayı kolaylaştıran ızgara çizgileri |
| `plt.tight_layout()` | Başlık ve eksen yazıları kesilmesin diye kenar boşluklarını ayarlar |
| `plt.show()` | Grafiği pencerede gösterir ve **pencere kapanana kadar programı bekletir** |

Accuracy grafiği (Bölüm 17) aynı şekilde, loss yerine `accuracy` ve `val_accuracy` değerleriyle ayrı bir pencerede çizilir. Yönergedeki şartlar (iki eğri aynı grafikte, X ekseni Epoch, Y ekseni Loss ya da Accuracy, accuracy ayrı bir grafikte) böylece sağlanıyor.

> Grafikleri `screenshots/loss.png` ve `screenshots/accuracy.png` olarak kaydetmeyi unutma ("Teslim için ekran görüntüleri" bölümüne bak).

#### Grafikler nasıl okunur?

- **Training loss azalıyor mu?** Azalıyorsa model eğitim verisinden öğreniyor demektir. Loss'un azalmasının sebebi, optimizer'ın her adımda ağırlıkları loss'u azaltacak yönde güncellemesidir.
- **Validation loss da azalıyor mu?** Azalıyorsa öğrenilen şeyler modelin görmediği verilere de genelleniyor demektir.
- **İki eğri arasındaki fark:** Ağırlıklar doğrudan eğitim verisine göre ayarlandığı için model, eğitim verisinde genellikle doğrulama verisinden biraz daha iyidir. Küçük bir fark normaldir. Fark epoch'lar ilerledikçe **büyüyorsa**, model genel kuralları öğrenmek yerine eğitim verisinin ayrıntılarını ezberlemeye başlamıştır.
- **Overfitting (aşırı öğrenme) işareti:** Training loss azalmaya devam ederken validation loss bir noktada durur ve **yükselmeye** başlar. Accuracy grafiğinde de training accuracy artmaya devam ederken validation accuracy yerinde sayar ya da düşer. Benzetme: Soruları anlamak yerine cevaplarını ezberleyen öğrenci, çalıştığı sorularda mükemmeldir ama yeni sorularda zorlanır.
- **Underfitting (yetersiz öğrenme):** İki loss da yüksek kalıyorsa model problemi yeterince öğrenememiştir. Model çok basit ya da epoch sayısı az olabilir.
- **1. epoch'taki tuhaflık:** İlk epoch'ta validation accuracy, training accuracy'den yüksek görünebilir. Sebebi şu: Training değerleri epoch **boyunca** ölçülüp ortalanır, yani epoch'un başında model henüz çok kötüyken yapılan tahminler de ortalamaya girer. Validation değeri ise epoch **sonunda**, model o epoch'taki öğrenmeyi tamamladıktan sonra ölçülür.

#### Bizim çalıştırmamızda ne görüldü?

Bölüm 15'teki değerlere göre:

- Training loss her epoch'ta azaldı: 0.4446 → 0.2023.
- Validation loss da her epoch'ta azaldı: 0.3261 → 0.2370. 5 epoch boyunca validation loss hiç yükselmediği için **belirgin bir overfitting görülmüyor**.
- 1. epoch'ta validation accuracy (0.8870) training accuracy'den (0.8454) yüksek: yukarıda anlatılan "1. epoch tuhaflığı". 2. ve 3. epoch arasında eğriler yer değiştiriyor; bundan sonra eğitim değerleri biraz daha iyi.
- Son epoch'ta training accuracy 0.9261, validation accuracy 0.9187: Aradaki fark yaklaşık 0,7 puan, yani küçük.

Senin değerlerin biraz farklı olacak; yorumu kendi grafiğine bakarak yap.

> **Kendin dene:** `epochs=5` değerini 15 yapıp programı yeniden çalıştır (denemeden sonra 5'e geri almayı unutma). Bu bilgisayarda denendiğinde:
>
> - Training loss 15 epoch boyunca düşmeye devam etti: 0.4446 → 0.0696.
> - Validation loss 8. epoch'ta en düşük değerine indi (0.2403), sonra yükselmeye başladı ve 15. epoch'ta 0.3116'ya çıktı.
> - Son epoch'ta training accuracy 0.9759, validation accuracy 0.9107: fark 0,7 puandan 6,5 puana açıldı.
> - Test accuracy %90,79 çıktı; 3 kat uzun eğitim, 5 epoch'luk sonucu (%90,87) geçemedi. Test loss ise 0.2557'den 0.3474'e yükseldi, çünkü model yanıldığı görüntülerde de aşırı emin hâle geldi.
>
> Bu, overfitting'in tipik görüntüsü: Model eğitim verisinde gittikçe iyileşirken görmediği verilerde iyileşmeyi bırakıyor.

> **Rapor için:** Yönerge bu grafikleri yorumlamanı istiyor: Loss neden azalıyor? Training ve validation loss arasındaki fark ne anlama geliyor? Overfitting görülüyor mu? Accuracy eğitim boyunca nasıl değişti?

---

### Bölüm 18: Test (Yönerge 14)

Boş satırları çıkarılmış hâli:

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : %{test_accuracy * 100:.2f}")
```

- **`evaluate`**: Modeli verilen veride çalıştırıp loss ve accuracy'yi hesaplar. **Ağırlıkları değiştirmez**, yalnızca ölçer.
- **`verbose=0`**: İlerleme çubuğu gösterilmez.
- İki değer döndürür: loss ve `compile`'da istediğimiz metrik (accuracy). `a, b = ...` yazımıyla iki değişkene birden atanır.
- Son satır accuracy'yi yüzde olarak yazar; yönerge bunu ayrıca istiyor.

**Ekranda göreceğin** (`test_results.png` için bu bölümün ekran görüntüsünü al):

```
==============================
TEST SONUÇLARI
==============================
Test Loss     : 0.2557
Test Accuracy : 0.9087
Test Accuracy : %90.87
```

**Neden ayrı bir test verisi?** Validation verisini eğitim boyunca sürekli izliyoruz ve epoch sayısı gibi kararları ona bakarak veriyoruz. Bu yüzden model dolaylı olarak validation verisine "uyum sağlamış" olabilir. Test verisi ise hiçbir karar için kullanılmadı; modelin gerçek hayattaki başarısının en dürüst tahmini budur.

**Test ve validation accuracy'yi karşılaştırırken:** Bölüm 15'teki **son epoch'un** `Val Accuracy` değerine bak, çünkü test, eğitimin bittiği andaki ağırlıklarla yapılıyor. Bizim çalıştırmamızda validation accuracy 0.9187, test accuracy 0.9087 çıktı: yaklaşık 1 puanlık fark. İki değer farklı görüntü kümeleriyle ölçüldüğü için (6.000 doğrulama, 10.000 test görüntüsü) aralarında küçük bir fark olması normaldir. 1 puan, 10.000 test görüntüsünde yaklaşık 100 görüntü demek.

> **Rapor için:** Test loss ve test accuracy değerlerini vermen, "Model test verisinde ne kadar başarılı?" ve "Test accuracy ile validation accuracy arasındaki fark nasıl açıklanabilir?" sorularını cevaplaman isteniyor.

---

### Bölüm 19: Tahminler (Yönerge 15)

```python
predictions = model.predict(test_images, verbose=0)

predicted_labels = np.argmax(predictions, axis=1)
```

**`model.predict`** her test görüntüsü için softmax çıktısını, yani 10 sınıfın olasılığını verir. Sonuç `(10000, 10)` boyutunda bir tablodur: Her satır bir görüntü, her sütun bir sınıf. Örneğin bir çanta görüntüsünün satırı şöyle görünebilir (değerler `tek_foto_tahmin.py`'nin çıktısından):

```
sınıf:    0       1       2       3       4       5       6       7       8       9
       [0.0004  0.0003  0.0002  0.0000  0.0002  0.0000  0.0003  0.0000  0.9985  0.0000]
```

**`np.argmax(predictions, axis=1)`**: Her satırda en büyük değerin **hangi sütunda** olduğunu (indeksini) verir. `axis=1` "her satırın içinde, sütunlar boyunca bak" demektir. Sonuç 10.000 elemanlı bir dizidir; her eleman o görüntü için tahmin edilen sınıf numarasıdır. Yukarıdaki satırda en büyük değer (0.9985) 8. sütunda, yani tahmin `CLASS_NAMES[8]`, "Çanta".

```python
np.argmax([0.1, 0.7, 0.2])   # 1   → en büyük değer (0.7) 1 numaralı sırada
np.max([0.1, 0.7, 0.2])      # 0.7 → en büyük değerin kendisi
```

Böylece her test görüntüsü `i` için yönergede istenen üç bilgi elde edilir:

| Bilgi | Kod |
|---|---|
| Gerçek sınıf | `CLASS_NAMES[test_labels[i]]` |
| Tahmin edilen sınıf | `CLASS_NAMES[predicted_labels[i]]` |
| Tahmin güveni | `predictions[i][predicted_labels[i]]`: satırdaki en büyük olasılık (Bölüm 21'de kullanılıyor) |

**"Güven" hakkında bir uyarı:** Güven, modelin **kendi** verdiği olasılıktır, doğru olduğunun garantisi değildir. Model bazen yüksek güvenle de yanılabilir. Ekranda `%100.0` görmen olasılığın tam 1 olduğu anlamına gelmez; 0.99996 gibi bir değer yuvarlanmıştır.

---

### Bölüm 20: Doğru ve yanlış tahminleri sayma

```python
correct = np.sum(predicted_labels == test_labels)
incorrect = np.sum(predicted_labels != test_labels)
```

`predicted_labels == test_labels` iki diziyi eleman eleman karşılaştırır: Tahmin doğruysa `True`, yanlışsa `False`. `np.sum` `True`'ları sayar (Bölüm 7'deki yöntem).

**Ekranda göreceğin:**

```
Doğru tahmin : 9087
Yanlış tahmin: 913
Toplam       : 10000
```

Bu sayılar Bölüm 18'deki accuracy ile tutarlı olmalı: 9087 ÷ 10000 = 0.9087. Yani accuracy'yi `evaluate`'e güvenmeden kendimiz de hesaplayıp sağlamasını yapmış oluyoruz.

---

### Bölüm 21: Rastgele 10 test görüntüsünü gösterme (Yönerge 16)

Boş satırları çıkarılmış hâli:

```python
np.random.seed(42)

random_indices = np.random.choice(len(test_images), size=10, replace=False)

plt.figure(figsize=(15, 6))

for plot_index, i in enumerate(random_indices):

    plt.subplot(2, 5, plot_index + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")

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
```

- **`np.random.seed(42)`**: Tohum burada yeniden ayarlanıyor, bu yüzden her çalıştırmada **aynı 10 görüntü** seçilir (aşağıdaki tabloya bak). Tahminler ve güven değerleri ise model her seferinde yeniden eğitildiği için biraz değişebilir.
- **`np.random.choice(10000, size=10, replace=False)`**: 0-9999 arasından 10 sayı seçer. `replace=False` "çekileni geri koyma" demektir, yani aynı görüntü iki kez seçilmez.
- **`for plot_index, i in enumerate(random_indices)`**: `plot_index` kutunun sırası (0-9), `i` seçilen görüntünün test verisindeki indeksi.
- **`plt.subplot(2, 5, plot_index + 1)`**: Pencereyi 2 satır × 5 sütunluk bir ızgaraya böler ve çizimin hangi kutuya yapılacağını seçer. Kutular soldan sağa, yukarıdan aşağıya 1'den başlayarak numaralanır; `plot_index` 0'dan başladığı için `+ 1` yazılır:

```
┌────┬────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │ 5  │
├────┼────┼────┼────┼────┤
│ 6  │ 7  │ 8  │ 9  │ 10 │
└────┴────┴────┴────┴────┘
```

- **`.squeeze()`**: Boyutu 1 olan boyutları atar: `(28, 28, 1)` → `(28, 28)`. Bölüm 10'da eklediğimiz kanal boyutunu geri kaldırır, çünkü `imshow` gri bir görüntüyü iki boyutlu bekler.
- **`plt.imshow(..., cmap="gray")`**: Sayı matrisini görüntü olarak çizer. `cmap="gray"` küçük değerleri siyah, büyükleri beyaz gösterir. (Yazılmasaydı matplotlib varsayılan renk paletiyle mor-sarı tonlarda çizerdi.)
- **`confidence = predictions[i][predicted_labels[i]] * 100`**: `i`. görüntünün olasılık satırından tahmin edilen sınıfın olasılığını alır ve yüzdeye çevirir (0.776 → 77.6).
- **`title = (...)`**: Dört satırlık başlık; `\n` alt satıra geçer. Yönergenin istediği Gerçek / Tahmin / Güven bilgileri ve en altta DOĞRU ya da YANLIŞ yazılır. Yönergedeki "doğru ve yanlış tahminler görsel olarak birbirinden ayırt edilebilmeli" şartını bu son satır sağlıyor.
- **`plt.axis("off")`**: Eksen çizgilerini ve sayılarını gizler. **`plt.tight_layout()`**: Kutuların başlıkları birbirine binmesin diye aralıkları ayarlar.

> **İstersen:** Doğru tahminleri yeşil, yanlışları kırmızı yazdırarak farkı daha da belirgin yapabilirsin: `plt.title(title, fontsize=9, color="green" if result == "DOĞRU" else "red")`.

**Hangi görüntüler seçiliyor?** Test görüntüleri sınıf sınıf ve dosya adı sırasıyla okunduğu için `i` indeksi doğrudan bir dosyaya karşılık gelir: Sınıf `CLASS_NAMES[i // 1000]`, dosya numarası `i % 1000 + 1` (`//` tam bölme, `%` bölümden kalan). Örneğin 6252 // 1000 = 6 (Gömlek) ve 6252 % 1000 + 1 = 253, yani `test/Gömlek/image_00253.png`. Tohum 42 ile seçilen 10 görüntü ve bizim çalıştırmamızdaki sonuçlar:

| Kutu | İndeks | Dosya | Tahmin (bizim çalıştırmamızda) | Güven | Sonuç |
|---|---|---|---|---|---|
| 1 | 6252 | `test/Gömlek/image_00253.png` | Gömlek | %48.5 | DOĞRU |
| 2 | 4684 | `test/Ceket/image_00685.png` | Ceket | %77.6 | DOĞRU |
| 3 | 1731 | `test/Pantolon/image_00732.png` | Pantolon | %100.0 | DOĞRU |
| 4 | 4742 | `test/Ceket/image_00743.png` | Ceket | %75.0 | DOĞRU |
| 5 | 4521 | `test/Ceket/image_00522.png` | Ceket | %99.7 | DOĞRU |
| 6 | 6340 | `test/Gömlek/image_00341.png` | Tişört | %58.8 | **YANLIŞ** |
| 7 | 576 | `test/Tişört/image_00577.png` | Tişört | %100.0 | DOĞRU |
| 8 | 5202 | `test/Sandalet/image_00203.png` | Sandalet | %100.0 | DOĞRU |
| 9 | 6363 | `test/Gömlek/image_00364.png` | Gömlek | %73.9 | DOĞRU |
| 10 | 439 | `test/Tişört/image_00440.png` | Tişört | %49.3 | DOĞRU |

Dikkat çeken noktalar: Pantolon ve sandalet %100 güvenle bilinirken düşük güvenli tahminlerin hepsi gömlek, ceket ve tişört gibi üst giyim sınıflarında. Tek yanlış tahmin de kolsuz bir gömleğin tişört sanılması. 28×28 piksellik siyah-beyaz görüntülerde bu kıyafetler gerçekten birbirine çok benziyor.

> **Rapor için:** 10 test görüntüsünün gerçek ve tahmin edilen sınıfları raporda gösterilmeli (`predictions.png`).

---

### Bölüm 22: Tahmin sınıf dağılımı (Yönerge 17)

```python
for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(predicted_labels == i)

    percentage = count / len(predicted_labels) * 100

    print(
        f"{class_name:18s}: "
        f"{count:5d} (%{percentage:.2f})"
    )
```

Her sınıf için "model kaç görüntüye bu sınıf dedi?" sorusunun cevabını sayar (Bölüm 7'deki yöntemin aynısı, bu sefer gerçek etiketler yerine tahminler üzerinde) ve bunun bütün tahminlere oranını yüzde olarak yazar.

**Ekranda göreceğin** (bizim çalıştırmamız):

```
Tişört            :   964 (%9.64)
Pantolon          :  1004 (%10.04)
Kazak             :   927 (%9.27)
Elbise            :  1020 (%10.20)
Ceket             :  1034 (%10.34)
Sandalet          :  1006 (%10.06)
Gömlek            :  1049 (%10.49)
Spor Ayakkabı     :  1003 (%10.03)
Çanta             :  1000 (%10.00)
Bot               :   993 (%9.93)
```

**Bu kontrol neden önemli?** Bozuk bir model (örneğin etiketler yanlış eşleşmişse ya da eğitim hiç ilerlememişse) her görüntüye aynı sınıfı söyleyebilir. Böyle bir model, 10 sınıflı dengeli bir test setinde yine de %10 doğruluk alır ve bu durum yalnızca accuracy'ye bakınca hemen fark edilmeyebilir. Dağılım tablosunda bir sınıfta 10.000, diğerlerinde 0 görmek sorunu anında ortaya çıkarır. Sağlıklı bir modelde tahminler sınıflara, gerçek dağılıma (her sınıfta 1.000) yakın biçimde yayılır. Bizim tablomuzda 10 sınıfın hepsi tahmin edilmiş ve her biri 1.000'e yakın; model bütün görüntüleri tek bir sınıfa toplamıyor.

**Tablo nasıl yorumlanır?** Gerçekte her sınıftan 1.000 görüntü var. Bir sınıf belirgin biçimde **az** tahmin ediliyorsa (Kazak: 927), o sınıfın bazı örnekleri başka sınıflarla karıştırılıyor demektir. Belirgin biçimde **fazla** tahmin ediliyorsa (Gömlek: 1.049), başka sınıfların örnekleri o sınıfa kayıyor demektir. Bu tablo karışıklığın **varlığını** gösterir ama **hangi sınıfın hangisiyle** karıştığını tek başına göstermez: Gömlek'in 1.049 kez tahmin edilmesi, bütün gömleklerin doğru bilindiği anlamına gelmez.

**Hangi sınıflar karışıyor?** Fikir vermesi için, projedeki kayıtlı `model.keras` 10.000 test görüntüsü üzerinde ayrıca denendi (`main.py` bu hesabı yapmıyor). Sınıf bazında doğru bilinme oranları:

| Sınıf | Gömlek | Kazak | Tişört | Ceket | Elbise | Spor Ayakkabı | Bot | Pantolon | Çanta | Sandalet |
|---|---|---|---|---|---|---|---|---|---|---|
| Doğru bilinme | %67,4 | %81,6 | %90,5 | %91,2 | %92,5 | %95,7 | %96,0 | %97,3 | %97,9 | %98,4 |

En sık karışan çiftler (gerçek → tahmin): Gömlek → Tişört (154 görüntü), Kazak → Ceket (92), Gömlek → Ceket (82), Tişört → Gömlek (59), Kazak → Gömlek (54), Gömlek → Kazak (52).

Yani en zor sınıf açık ara **Gömlek**; karışıklıkların çoğu birbirine benzeyen üst giyim sınıfları (gömlek, tişört, kazak, ceket) arasında. Aralarındaki farklar yaka, düğme ya da kol boyu gibi küçük ayrıntılar ve 28×28 piksellik bir görüntüde bu ayrıntıların çoğu kayboluyor. Şekli belirgin olan pantolon, çanta ve ayakkabı sınıfları ise %95'in üzerinde doğru biliniyor.

> **Rapor için:** "En çok hangi sınıfların karıştırıldığını düşünüyorsunuz?" sorusunu cevaplarken bu tabloyu, Bölüm 21'deki YANLIŞ örnekleri ve sınıfların birbirine görsel benzerliğini birlikte değerlendir.

---

### Başarıyı artırmak için neler yapılabilir?

Yönergenin değerlendirme sorularından biri bu. Bunlar ödevin zorunlu bir parçası değil; soruyu cevaplarken fikir vermesi için:

| Yöntem | Ne işe yarar? |
|---|---|
| Daha fazla epoch + validation loss takibi | Model öğrenmeye devam ediyorsa başarı artar. Validation loss yükselmeye başladığında durmak gerekir; Keras'ta bunu otomatik yapan `EarlyStopping` adında bir araç var. Tek başına epoch artırmak yetmez: Bu modelde 15 epoch, 5 epoch'tan daha iyi test sonucu vermedi (Bölüm 16-17'deki "Kendin dene" kutusu). |
| Daha derin model (ikinci bir Conv2D + MaxPooling2D bloğu, daha fazla filtre) | Kenarların birleşiminden oluşan daha karmaşık desenleri (yaka, düğme, kol) yakalayabilir. |
| `Dropout` katmanı | Eğitim sırasında nöronların bir kısmını rastgele kapatır; modelin ezberlemesini (overfitting) zorlaştırır. |
| Veri artırma (data augmentation) | Eğitim görüntülerini hafifçe kaydırarak ya da yatay çevirerek yeni örnekler üretir; model daha çeşitli örnek görür. |
| `BatchNormalization` | Katman çıktılarını normalize ederek eğitimi daha hızlı ve kararlı hâle getirir. |
| Öğrenme hızını (learning rate) ayarlamak | Adım büyüklüğü çok büyükse eğitim dalgalanır, çok küçükse yavaş ilerler. |

Modelin kapasitesini artıran yöntemler (daha derin model) underfitting'e, ezberlemeyi azaltan yöntemler (Dropout, veri artırma, EarlyStopping) overfitting'e karşı işe yarar.

---

## Tek fotoğraf demosu: `tek_foto_tahmin.py`

Bu dosya ödevin teslim listesinde yok; modelin **tek bir** fotoğraf için nasıl karar verdiğini adım adım görmek için yazılmış bir demo. `main.py` gibi sıfırdan eğitim yapmaz, daha önce eğitilip kaydedilmiş `model.keras` dosyasını yükler. Bu yüzden birkaç saniyede çalışır. Sözlü sunumda modelin nasıl çalıştığını göstermek için de kullanışlı.

### Çalıştırma

```powershell
python tek_foto_tahmin.py
python tek_foto_tahmin.py "data/fashion_mnist/test/Gömlek/image_00341.png"
```

- Dosya yolu verilmezse varsayılan görüntü `test/Çanta/image_00001.png` kullanılır.
- Program 7 adımdan oluşur ve her adımdan sonra **Enter'a basmanı** bekler; böylece her adımı rahatça okuyabilirsin.
- Yol boşluk içeriyorsa (ör. `Spor Ayakkabı`) tırnak içinde yaz.
- `main.py` gibi bu da proje klasöründen çalıştırılmalı (`model.keras` ve `data` göreli yollarla aranıyor).

### Adım adım

**Hazırlık:** Programın başındaki `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` ve `tf.get_logger().setLevel("ERROR")` satırları TensorFlow'un bilgi mesajlarının çoğunu gizler (oneDNN mesajı yine de görünebilir). `sys.argv`, komut satırında yazılan kelimelerin listesidir: `sys.argv[0]` programın adı, varsa `sys.argv[1]` verdiğin dosya yolu.

```python
image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
```

"Komut satırında bir yol verildiyse onu, verilmediyse varsayılan görüntüyü kullan" demek.

**Adım 1: Modeli yükle**

```python
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
```

`compile=False`, optimizer ve loss bilgilerinin yüklenmemesini sağlar; yalnızca tahmin yapacağımız için onlara gerek yok. `model.input_shape` modelin beklediği girişi gösterir: `(None, 28, 28, 1)`. `None` yine batch boyutudur.

**Adım 2-4: Oku, sayılara çevir, normalize et.** `main.py` ile **birebir aynı** ön işleme: `Image.open(...).convert("L")` → `np.array(..., dtype=np.uint8)` → `astype("float32") / 255.0`. Varsayılan çanta görüntüsü terminalde şöyle çizilir (boş satırlar kırpıldı); her piksel parlaklığına göre ` .:-=+*#%@` karakterlerinden biriyle gösterilir, koyu → boşluk, açık → `@`:

```
              ==
            ::==::::
            ********####******##**********####**++++--
          **##**********####################**######**
        ..==******##********##################**##****::
        ::==##******##****############**######**######::
      ::..==##**####****######****##########********%%--
      --  ++##****##****##**************########****%%--
      --  ++##**####****######**********######******%%--
      ::  ==##**##************################******%%::
        ----##**######**********####****************%%::
        ..%%##**************************************##::
          ++################**######################%%::
          ==%%++**##################################%%::
          ==####++++######################****##****##..
        @@****##%%****##########********************##..
    --==..**##########**##%%%%%%%%%%%%%%%%%%######**%%..
..==..    ::====--++**++==                          ..
```

Çizim kodundaki `chars[int(p) * len(chars) // 256]`, 0-255 arasındaki bir pikseli 0-9 arası bir sıra numarasına çevirir: 0 → ilk karakter (boşluk), 255 → son karakter (`@`). Her karakter iki kez yazılır, çünkü terminaldeki karakterler enlerinden daha uzundur; tek yazılsa görüntü yatayda basık görünürdü.

Program ardından görüntünün ortasındaki 10 pikseli normalizasyondan önce ve sonra gösterir:

```
Ortadan 10 piksel, pixels[14, 9:19]: [175 176 177 179 179 178 181 184 183 183]
Aynı 10 piksel, x[14, 9:19]: [0.69 0.69 0.69 0.7  0.7  0.7  0.71 0.72 0.72 0.72]
```

`pixels[14, 9:19]`: 14. satırın 9'dan 18'e kadarki sütunları (dilimlemede bitiş değeri olan 19 dahil değil).

**Adım 5: Modelin beklediği biçime getir**

```python
x = np.expand_dims(x, axis=-1)   # (28, 28)    → (28, 28, 1)     kanal boyutu
x = np.expand_dims(x, axis=0)    # (28, 28, 1) → (1, 28, 28, 1)  batch boyutu
```

İlki `main.py`'deki kanal boyutu (Bölüm 10). İkincisi yeni: Model görüntüleri her zaman **grup (batch) hâlinde** alır; model özetindeki `None` bu boyuttu. Tek bir fotoğraf vereceğimiz için başa boyutu 1 olan bir grup boyutu ekliyoruz: "1 fotoğraflık bir grup". `main.py`'de buna gerek yoktu, çünkü 10.000 görüntülük dizi zaten `(10000, 28, 28, 1)` biçimindeydi.

**Adım 6: Tahmin**

```python
probabilities = model.predict(x, verbose=0)[0]
```

`predict` sonucu yine grup hâlinde, `(1, 10)` biçiminde döndürür. `[0]` gruptaki ilk (ve tek) fotoğrafın 10 olasılığını alır. Ekranda her olasılık bir çubukla gösterilir (`"█" * int(round(p * 40))`; %100 olsaydı 40 karakterlik bir çubuk):

```
0: Tişört         %  0.04
1: Pantolon       %  0.03
2: Kazak          %  0.02
3: Elbise         %  0.00
4: Ceket          %  0.02
5: Sandalet       %  0.00
6: Gömlek         %  0.03
7: Spor Ayakkabı  %  0.00
8: Çanta          % 99.85  ████████████████████████████████████████
9: Bot            %  0.00

Toplam: %100.00
```

Olasılıkların toplamı %100: softmax'in özelliği (Bölüm 11).

**Adım 7: Sonuç**

```python
predicted_label = np.argmax(probabilities)
real_class = os.path.basename(os.path.dirname(image_path))
```

`argmax` en yüksek olasılığın sırasını verir (8 → Çanta). Gerçek sınıf ise dosyanın bulunduğu klasörün adından çıkarılır: `os.path.dirname(...)` dosya adını atıp klasör yolunu bırakır (`data/fashion_mnist/test/Çanta`), `os.path.basename(...)` bu yolun son parçasını alır (`Çanta`). Yani demo da etiketi `main.py` gibi klasör adından alıyor. Model bu bilgiyi hiç görmez; yalnızca tahminin doğru olup olmadığını söylemek için kullanılır.

```
Tahmin : Çanta
Güven  : %99.85
Gerçek : Çanta (klasör adından)
Sonuç  : DOĞRU
```

### Bir yanlış tahmini incele

Bölüm 21'de yanlış bilinen gömleği demoya verirsen:

```powershell
python tek_foto_tahmin.py "data/fashion_mnist/test/Gömlek/image_00341.png"
```

kayıtlı model de bu görüntüye **Tişört** der: Tişört %81,77, Gömlek %17,60. Model iki seçenek arasında kalmış ve bu iki seçenek, görsel olarak birbirine en çok benzeyen sınıflar. Sözlü sunumda "Model nerede ve neden yanılıyor?" sorusuna güzel bir örnek.

### `model.keras` hakkında

- `model.keras`, `main.py`'deki mimariyle (Conv2D 32 → MaxPooling2D → Flatten → Dense 64 → Dense 10) eğitilmiş bir modelin kaydı. Dosyanın içindeki bilgilere göre 29.09.2026'da Keras 3.15.1 ile kaydedilmiş. Test verisinde %90,85 doğruluk veriyor.
- Şu anki `main.py` modeli **kaydetmiyor**: Her çalıştırmada model sıfırdan eğitilir ve program bitince bellekten silinir. `main.py`'de eğittiğin modeli demoda kullanmak istersen eğitimden sonra (örneğin Bölüm 14'ün altına) şu satırı ekleyebilirsin. Bu satır mevcut `model.keras` dosyasının üzerine yazar:

```python
model.save("model.keras")
```

- Demo görüntüyü yeniden boyutlandırmaz; 28×28 olmayan bir fotoğraf verirsen model hata verir. Ayrıca model yalnızca Fashion-MNIST tarzı görüntüler gördü: siyah zemin üzerinde ortalanmış, açık renkli tek bir kıyafet. Telefonla beyaz zemin üzerinde çekilmiş bir fotoğrafı (28×28'e küçültsen bile) büyük ihtimalle yanlış bilir. Bir modelin yalnızca eğitimde gördüğü türden verilerde iyi çalışması, makine öğrenmesinin önemli bir sınırıdır.

---

## Rapor yazarken nereye bakmalısın?

| Rapor bölümü (Yönerge 18) | Programda nereye bakmalısın? | Rehberde ilgili yer |
|---|---|---|
| 1. Amaç | | Büyük resim |
| 2. Veri seti (Fashion-MNIST, 10 sınıf) | Bölüm 4-5 çıktısı (sınıf başına görüntü sayısı) | Temel kavramlar, Bölüm 1 |
| 3. Veri organizasyonu (klasör yapısı, etiketin klasör adından gelmesi) | `data/` klasörü, `load_images_from_folder` fonksiyonu | Projeyi çalıştırma, Bölüm 2-3 |
| Boyutların anlamı | Bölüm 6 çıktısı | Bölüm 6 |
| 4. Ön işleme (gri seviye, normalizasyon, kanal boyutu, karıştırma) | Bölüm 3, 8, 9 ve 10 | Bölüm 3, 8, 9, 10 |
| 5. CNN mimarisi tablosu | `model.summary()` çıktısı | Bölüm 11-12 |
| Optimizer, loss, loss seçiminin nedeni, accuracy'nin anlamı | `model.compile` | Bölüm 13 |
| 6. Eğitim sonuçları ve grafik yorumları | Eğitim geçmişi, `loss.png`, `accuracy.png` | Bölüm 14-17 |
| 7. Test sonuçları | TEST SONUÇLARI | Bölüm 18 |
| 8. Tahmin sonuçları (10 görüntü) | `predictions.png` | Bölüm 19, 21 |
| 9. Değerlendirme | Aşağıdaki tablo | |

**Değerlendirme soruları için:**

| Soru | Nereye bakmalısın? |
|---|---|
| Model test verisinde ne kadar başarılı oldu? | Bölüm 18 (Test Accuracy) ve Bölüm 20 (doğru/yanlış sayıları) |
| Eğitim ve validation accuracy arasında büyük fark var mı? | Bölüm 15'in son satırı, `accuracy.png` |
| Modelde overfitting görülüyor mu? | `loss.png`: Validation loss yükselmeye başlıyor mu? (Bölüm 16-17) |
| En çok hangi sınıflar karıştırılıyor? | Bölüm 21'deki YANLIŞ örnekler, Bölüm 22'deki dağılım |
| Başarıyı artırmak için hangi yöntemler uygulanabilir? | "Başarıyı artırmak için neler yapılabilir?" |
| Veri setinin karıştırılması neden gerekli? | Bölüm 9 |
| Test ve validation accuracy farkı nasıl açıklanabilir? | Bölüm 18 |

> Yönerge raporun "yalnızca kod çıktılarından oluşmamasını, sonuçların yorumlanmasını" istiyor. Bu rehber kavramları açıklıyor; yorumları kendi sonuçlarına bakarak kendi cümlelerinle yazmalısın.

---

## Kendini test et (sözlü sınava hazırlık)

Önce soruyu kendin cevaplamaya çalış, sonra cevabı açıp karşılaştır.

<details>
<summary>1. Bir görüntünün etiketi nereden geliyor?</summary>

Görüntünün bulunduğu klasörün adından. `load_images_from_folder` sınıf klasörlerini `CLASS_NAMES` sırasıyla okur ve her klasördeki görüntülere, o klasörün listedeki sıra numarasını etiket olarak verir (`labels.append(label)`). Örneğin `train/Çanta/` içindeki her dosyanın etiketi 8'dir. CSV ya da Excel dosyası kullanılmaz.

</details>

<details>
<summary>2. <code>CLASS_NAMES</code> listesindeki bir isim klasör adından farklı yazılsaydı ne olurdu?</summary>

`os.listdir` o klasörü bulamaz ve program `FileNotFoundError` ile durur. Listedeki isimler klasör adlarıyla, Türkçe karakterler ve boşluklar dahil, birebir aynı olmalı.

</details>

<details>
<summary>3. Piksel değerlerini neden 255'e bölüyoruz?</summary>

0-255 aralığını 0-1 aralığına getirmek için (normalizasyon). Küçük ve benzer ölçekteki girdilerle ağdaki hesaplamalar dengeli kalır, eğitim daha kararlı ve hızlı ilerler.

</details>

<details>
<summary>4. <code>(60000, 28, 28, 1)</code> şeklindeki her sayı ne anlama geliyor?</summary>

60.000 görüntü, 28 piksel yükseklik, 28 piksel genişlik ve 1 kanal (gri seviye).

</details>

<details>
<summary>5. Sondaki 1 boyutunu neden ekledik?</summary>

Conv2D katmanı her görüntüyü (yükseklik, genişlik, kanal) biçiminde bekler. Gri görüntünün tek kanalı olduğu için `np.expand_dims` ile sona boyutu 1 olan bir boyut ekledik. Değerler değişmez, yalnızca şekil değişir.

</details>

<details>
<summary>6. Eğitim verisini neden karıştırdık? Karıştırmasaydık ne olurdu?</summary>

Veriler sınıf sınıf okunduğu için sıralıydı. `validation_split=0.1` doğrulama verisini verinin son %10'undan ayırır. Her sınıfta 6.000 görüntü olduğu için son 6.000 görüntünün hepsi Bot olurdu: Model eğitimde hiç bot görmez, doğrulama sonuçları da modelin gerçek başarısını yansıtmazdı.

</details>

<details>
<summary>7. <code>model.fit</code> içindeki <code>shuffle=True</code> bu sorunu neden çözmüyor?</summary>

Çünkü Keras doğrulama verisini karıştırmadan **önce**, verinin sonundan ayırır. `shuffle=True` yalnızca geriye kalan eğitim kısmını her epoch'ta kendi içinde karıştırır.

</details>

<details>
<summary>8. Görüntüleri ve etiketleri neden aynı indekslerle karıştırdık?</summary>

Her görüntünün kendi etiketiyle eşleşmesi korunsun diye. Ayrı ayrı karıştırsaydık görüntüler yanlış etiketlerle eşleşir ve model yanlış bilgilerle eğitilirdi.

</details>

<details>
<summary>9. Conv2D katmanının çıktısı neden 26×26×32?</summary>

3×3'lük filtre 28 piksellik bir kenar boyunca, dışarı taşmadan 28 − 3 + 1 = 26 farklı konuma yerleşebilir. Her iki yönde 26 konum olduğu için her filtre 26×26'lık bir harita üretir; 32 filtre olduğu için 32 harita çıkar.

</details>

<details>
<summary>10. Conv2D katmanında neden 320 parametre var?</summary>

Her filtre 3 × 3 = 9 ağırlık ve 1 bias, yani 10 parametre içerir. 32 filtre × 10 = 320.

</details>

<details>
<summary>11. MaxPooling2D ne yapar? Parametresi neden 0?</summary>

Her 2×2'lik bölgeden en büyük değeri alarak boyutu yarıya indirir (26×26 → 13×13). Sabit bir kural uyguladığı için öğrenilecek ağırlığı yoktur.

</details>

<details>
<summary>12. Flatten neden gerekli?</summary>

Dense katmanı tek boyutlu girdi bekler. Flatten, 13×13×32'lik çıktıyı 5408 elemanlı tek boyutlu bir vektöre düzleştirir.

</details>

<details>
<summary>13. Çıkış katmanında neden 10 nöron ve softmax var?</summary>

10 sınıf olduğu için her sınıfa bir nöron var. Softmax bu 10 çıktıyı toplamı 1 olan olasılıklara çevirir; en yüksek olasılıklı sınıf tahmin olur.

</details>

<details>
<summary>14. Neden <code>sparse_categorical_crossentropy</code> kullandık?</summary>

Problem çok sınıflı sınıflandırma ve çıkış katmanı softmax olasılıkları üretiyor. Cross-entropy, doğru sınıfa verilen olasılık düştükçe büyüyen bir hata ölçüsüdür. "Sparse" versiyonu, etiketlerimiz tam sayı (0-9) olduğu için kullanılıyor; etiketler one-hot biçiminde olsaydı `categorical_crossentropy` kullanılırdı.

</details>

<details>
<summary>15. Optimizer ne işe yarar? Neden Adam?</summary>

Optimizer, loss'u azaltmak için ağırlıkların nasıl güncelleneceğini belirler. Adam her ağırlık için adım büyüklüğünü kendisi ayarladığı için çoğu problemde ayar gerektirmeden hızlı ve kararlı çalışır.

</details>

<details>
<summary>16. Epoch, batch ve step arasındaki fark ne? Ekrandaki 844 nereden geliyor?</summary>

Batch, bir güncellemede birlikte işlenen görüntü grubudur (64 görüntü). Step, tek bir batch ile yapılan bir ağırlık güncellemesidir. Epoch, eğitim verisinin tamamının bir kez işlenmesidir. 60.000 görüntünün %90'ı olan 54.000 eğitim görüntüsü ÷ 64 = 843,75; son grup eksik (48 görüntü) olduğu için 1 epoch = 844 step.

</details>

<details>
<summary>17. Validation ve test verisinin farkı ne?</summary>

Validation verisi eğitim sırasında her epoch sonunda modeli izlemek için kullanılır; eğitimle ilgili kararlar (ör. epoch sayısı) ona bakılarak verilir. Test verisi eğitim tamamen bittikten sonra bir kez kullanılır. Hiçbir karar için kullanılmadığından modelin gerçek başarısının en dürüst ölçüsüdür.

</details>

<details>
<summary>18. Overfitting nedir, grafikte nasıl anlaşılır?</summary>

Modelin genel kuralları öğrenmek yerine eğitim verisini ezberlemesidir. Training loss azalmaya devam ederken validation loss'un yükselmeye başlaması ve training ile validation accuracy arasındaki farkın açılması overfitting işaretidir.

</details>

<details>
<summary>19. "Güven: %99" tahminin kesin doğru olduğu anlamına mı gelir?</summary>

Hayır. Güven, softmax'in tahmin edilen sınıfa verdiği olasılıktır; modelin kendi tahminidir. Model yüksek güvenle de yanılabilir.

</details>

<details>
<summary>20. Tahmin sınıf dağılımına neden bakıyoruz?</summary>

Modelin bütün görüntüleri tek bir sınıfa (ya da birkaç sınıfa) tahmin edip etmediğini görmek için. Sağlıklı bir modelde tahminler sınıflara, gerçek dağılıma (her sınıfta 1.000) yakın biçimde yayılır.

</details>

<details>
<summary>21. Model özetindeki <code>None</code> ne demek?</summary>

Batch boyutu. Model aynı anda kaç görüntü işleyeceğini önceden bilmez (eğitimde 64, tek fotoğraf demosunda 1), bu yüzden o boyut boş bırakılır.

</details>

<details>
<summary>22. <code>np.random.seed(42)</code> ne işe yarıyor? Sonuçlar neden yine de her çalıştırmada biraz farklı?</summary>

NumPy'nin rastgele işlemlerini (eğitim verisinin karıştırılması, gösterilecek 10 test görüntüsünün seçimi) her çalıştırmada aynı yapar. Ama modelin başlangıç ağırlıkları ve `fit`'in epoch içi karıştırması Keras ve TensorFlow tarafından yapılır ve bu tohumdan etkilenmez; bu yüzden loss ve accuracy değerleri biraz değişir. Hepsini sabitlemek için `tf.keras.utils.set_random_seed(42)` kullanılabilir.

</details>

<details>
<summary>23. <code>tek_foto_tahmin.py</code>'de <code>expand_dims</code> neden iki kez kullanılıyor?</summary>

İlki kanal boyutunu ekler: `(28, 28)` → `(28, 28, 1)`. İkincisi batch boyutunu ekler: `(28, 28, 1)` → `(1, 28, 28, 1)`. Model görüntüleri her zaman grup hâlinde aldığı için tek bir fotoğraf da "1 fotoğraflık grup" olarak verilir.

</details>

<details>
<summary>24. Yeni bir fotoğrafı tahmin ederken neden <code>main.py</code>'deki ön işlemenin aynısını yapmak zorundayız?</summary>

Model eğitimde gri seviyeli, 0-1 aralığında, 28×28×1 biçiminde sayılar gördü. Yeni fotoğraf farklı biçimde verilirse (örneğin 255'e bölünmezse) model hiç görmediği büyüklükte sayılarla karşılaşır ve tahminleri anlamsızlaşır.

</details>

---

## Sık karşılaşılan sorunlar

| Sorun | Sebep ve çözüm |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` (ya da `PIL`, `matplotlib`) | Program, kütüphanelerin kurulu olmadığı bir Python ile çalıştırılmış. VS Code'da sağ alttan Python 3.12'yi seç ya da kütüphaneleri `python -m pip install numpy matplotlib pillow tensorflow` ile kur. |
| `FileNotFoundError: [WinError 3] Sistem belirtilen yolu bulamıyor: 'data/fashion_mnist\\train\\Tişört'` | Program proje klasörünün dışından çalıştırılmış ya da bir sınıf klasörü eksik veya adı farklı yazılmış. Terminalde proje klasörüne geç (`cd`); klasör adları `CLASS_NAMES` ile harfi harfine aynı olmalı. |
| Program ilerlemiyor, terminal bekliyor | Bir grafik penceresi açık. Pencereyi kapatınca program devam eder (toplam 3 pencere). |
| `UserWarning: Do not pass an input_shape/input_dim argument...` | Hata değil, uyarı (Bölüm 11). Model doğru kurulur. |
| Sonuçlar bir önceki çalıştırmadan biraz farklı | Normal. Modelin başlangıç ağırlıkları her çalıştırmada rastgele (Bölüm 9, "Rastgelelik tohumu"). |
| `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape...` | `data` klasörüne 28×28 olmayan bir PNG eklenmiş. `main.py` bütün görüntülerin aynı boyutta olmasını bekler. |
| `ValueError: File not found: filepath=model.keras` (`tek_foto_tahmin.py`) | Terminal proje klasöründe değil ya da `model.keras` silinmiş. Proje klasöründen çalıştır; dosya yoksa `main.py`'ye `model.save("model.keras")` ekleyip modeli yeniden eğit. |
| `tek_foto_tahmin.py` bir adımda duruyor | Enter'a basmanı bekliyor; bilerek böyle yazılmış. |

---

## Küçük sözlük

| Terim | Anlamı |
|---|---|
| Piksel | Görüntünün en küçük noktası; gri görüntüde 0-255 arası tek bir sayı |
| Gri seviye | Her pikselin yalnızca parlaklık bilgisi taşıdığı (renksiz) görüntü |
| Kanal | Bir pikseldeki değer sayısı: gri = 1, renkli (RGB) = 3 |
| Normalizasyon | Değerleri ortak bir aralığa (burada 0-1) getirme |
| Etiket (label) | Bir görüntünün doğru sınıfı (0-9 arası bir sayı) |
| Sınıf (class) | Kategori: Tişört, Pantolon, ... |
| Train / Validation / Test | Eğitim / doğrulama / test verisi |
| Epoch | Eğitim verisinin tamamının modele bir kez gösterilmesi |
| Batch | Bir ağırlık güncellemesinde birlikte işlenen görüntü grubu |
| Step | Tek bir batch ile yapılan bir güncelleme |
| Ağırlık / parametre | Modelin eğitimde ayarlanan sayıları |
| Bias | Her nörona ya da filtreye eklenen, o da öğrenilen sabit sayı |
| Loss | Tahminlerin ne kadar yanlış olduğunu gösteren sayı (küçük = iyi) |
| Accuracy | Doğru tahminlerin oranı (büyük = iyi) |
| Optimizer | Ağırlıkları loss'u azaltacak şekilde güncelleyen yöntem (Adam) |
| Learning rate | Optimizer'ın adım büyüklüğü |
| Gradyan | Bir ağırlık değişince loss'un hangi yönde ve ne kadar değişeceği |
| Backpropagation | Gradyanların çıkıştan girişe doğru hesaplanması |
| CNN | Evrişimli sinir ağı; görüntüdeki yerel desenleri filtrelerle yakalar |
| Filtre (kernel) | Görüntü üzerinde gezdirilen küçük ağırlık tablosu (burada 3×3) |
| Özellik haritası (feature map) | Bir filtrenin görüntü üzerinde gezinerek ürettiği çıktı |
| ReLU | Negatif değerleri 0 yapan aktivasyon fonksiyonu: max(0, x) |
| Pooling | Bölgeleri özetleyerek (burada en büyüğünü alarak) boyutu küçültme |
| Flatten | Çok boyutlu veriyi tek boyuta düzleştirme |
| Dense | Her nöronun önceki katmanın bütün çıktılarına bağlı olduğu katman |
| Softmax | Skorları toplamı 1 olan olasılıklara çeviren fonksiyon |
| Argmax | En büyük değerin sırası (indeksi) |
| Overfitting | Eğitim verisini ezberleme; görülmemiş verilerde başarının düşmesi |
| Underfitting | Modelin eğitim verisini bile yeterince öğrenememesi |
| Seed (tohum) | Rastgele sayı üretimini tekrarlanabilir yapan başlangıç değeri |
| Göreli yol | Çalışma klasörüne göre tanımlanan dosya yolu (ör. `data/fashion_mnist`) |
| `.keras` dosyası | Eğitilmiş bir modelin mimarisini ve ağırlıklarını saklayan dosya |
