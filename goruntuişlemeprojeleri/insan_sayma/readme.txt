📜 Proje Tanımı
Bu proje, bilgisayar kamerası kullanarak gerçek zamanlı insan sayma sistemi geliştirir. OpenCV ve PyTorch kullanılarak hazırlanmıştır.

🛠️ Teknik Özellikler
Kamera Girişi: Standart USB webcam veya RTSP akışı

Model: Faster R-CNN with MobileNetV3 (COCO pretrained)

Çözünürlük: 1280x720 (ayarlanabilir)

FPS Gösterimi: Anlık performans metrikleri

Platform: Windows/Linux/macOS (Python 3.6+)

⚙️ Kurulum
Gereksinimleri yükleyin:
pip install opencv-python torch torchvision numpy

⚠️ Bilinen Sorunlar
İlk çalıştırmada model dosyası indirilir (~74MB)

Düşük ışıkta tespit doğruluğu azalabilir

CPU'da yüksek çözünürlüklerde FPS düşebilir

©️ Lisans
MIT License - Her türlü kullanıma açıktır
 
Kullanılan python version :3.11.0