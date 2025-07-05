Mağaza İçi Müşteri Takip ve Heatmap Oluşturma Sistemi( Yol üzerinde  Kullanım )
📌 Proje Açıklaması
Bu proje, mağaza içi müşteri hareketlerini analiz etmek için geliştirilmiş bir bilgisayarlı görü sistemidir. YOLOv8 modeli kullanarak insan tespiti yapar ve müşterilerin hareketlerini takip ederek bir heatmap (ısı haritası) oluşturur.

🛠️ Teknolojiler
Python 3.8+

OpenCV (cv2)

Ultralytics YOLOv8

NumPy

Requests

📋 Özellikler
✔ Gerçek zamanlı insan tespiti ve takibi
✔ Mağaza içi yoğunluk heatmap'i oluşturma
✔ Çoklu video kaynağı desteği (dosya, webcam, URL)
✔ Otomatik örnek video indirme
✔ Detaylı hata yönetimi ve loglama

⚙️ Kurulum
Gereksinimleri yükleyin:

pip install opencv-python numpy requests ultralytics
Projeyi klonlayın veya dosyaları indirin:

git clone https://github.com/sizin-kullanici-adiniz/magaza-takip-sistemi.git
cd magiza-takip-sistemi
🚀 Kullanım
Temel Kullanım:
python main.py
Seçenekler:
Yerel video kullanmak için mall_video.mp4 veya store_video.mp4 dosyasını proje klasörüne ekleyin

Webcam kullanmak için herhangi bir video dosyası eklemeyin

Özel video yolu kullanmak için kodu düzenleyin