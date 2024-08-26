# Sentiment Analysis Project

Bu proje, doğal dil işleme (NLP) teknikleri kullanarak duygu analizi yapmayı amaçlamaktadır. Çeşitli makine öğrenimi modelleri (Logistic Regression, Naive Bayes) kullanılarak metin verileri üzerinden pozitif veya negatif duygu sınıflandırması yapılmaktadır.

## Proje Hedefleri

Bu projenin temel amacı, verilen bir metnin duygusal tonunu (pozitif veya negatif) belirlemek için makine öğrenimi tekniklerini kullanarak bir model geliştirmektir. Proje, gerçek dünyada karşılaşılan metin sınıflandırma problemlerine yönelik bir çözüm sunmaktadır.

## Kullanılan Teknikler ve Kütüphaneler

Bu projede aşağıdaki Python kütüphaneleri ve teknikleri kullanılmıştır:

- **Pandas**: Veri okuma, temizleme ve işleme.
- **Scikit-learn**: Makine öğrenimi modellerini oluşturma ve değerlendirme.
- **NLTK (Natural Language Toolkit)**: Metin ön işleme adımları (stop words temizleme, lemmatizasyon vb.).
- **TextBlob**: Metin işleme ve lemmatizasyon.
- **Keras**: Derin öğrenme modelleri.
- **Tkinter**: Basit bir grafiksel kullanıcı arayüzü (GUI) oluşturma.

## Veri Seti

Veri seti, "train.tsv" adlı bir dosyadan yüklenmiştir. Bu veri seti, metinlerin duygusal tonlarını belirlemek için kullanılmıştır. Veri ön işleme adımları şunları içerir:

- Büyük-küçük harf dönüşümü
- Noktalama işaretlerinin ve sayıların kaldırılması
- Stop words'lerin çıkarılması
- Seyrek kelimelerin kaldırılması
- Lemmatizasyon

Veri seti, eğitim ve test veri kümelerine ayrılmış ve çeşitli makine öğrenimi modelleri (Logistic Regression, Naive Bayes) kullanılarak eğitilmiştir.

## Model Eğitimi ve Değerlendirme

Projede kullanılan modeller ve teknikler şunlardır:

1. **Logistic Regression**: Duygu analizi için kullanılan bir sınıflandırma algoritması.
2. **Naive Bayes**: Basit ve etkili bir sınıflandırma algoritması.
3. **Count Vectors**: Metinlerdeki kelime frekanslarını sayarak bir "count matrix" oluşturur.
4. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Metinlerdeki terim frekansını ters belge frekansı ile çarparak bir "tf-idf matrix" oluşturur.

Model performansı, cross-validation kullanılarak değerlendirilmiştir. 

## Kullanıcı Arayüzü

Proje, kullanıcıdan metin girişi alarak bu metin üzerinde duygu analizi yapabilen basit bir GUI'ye sahiptir. Tkinter kullanılarak oluşturulan bu arayüz, kullanıcının bir cümle girmesine ve analizin sonucunu görmesine olanak tanır.

--> Sentiment-Analysis-Project.py dosyasını çalıştırarak duygu analizini başlatın:

--> GUI arayüzü üzerinde cümle girişinizi yapın ve "Analiz Et" butonuna basın.

--> Model çıktıları ve doğruluk oranları terminalde görüntülenir.


LOJ N-GRAM TF-IDF Doğruluk Oranı: 0.85
LOJ CHARLEVEL Doğruluk Oranı: 0.87
Naive Bayes Count Vectors Doğruluk Oranı: 0.82
Naive Bayes Word-Level TF-IDF Doğruluk Oranı: 0.84


Katkıda bulunmak isterseniz, lütfen bir pull request oluşturun. Her türlü katkı (hata düzeltmeleri, yeni özellikler, belge güncellemeleri) memnuniyetle karşılanır.

Kaynaklar
Pandas Belgeleri
Scikit-learn Belgeleri
NLTK Belgeleri
TextBlob Belgeleri
Keras Belgeleri
