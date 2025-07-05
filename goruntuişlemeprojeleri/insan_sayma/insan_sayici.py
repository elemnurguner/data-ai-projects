import cv2
import torch
import torchvision
import numpy as np
from torchvision.transforms import functional as F
from datetime import datetime

# Model yükleme fonksiyonu
def load_model():
    # Daha hızlı çalışan küçük bir model seçiyoruz
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model.eval()
    return model

# İnsan tespit ve sayma fonksiyonu
def detect_people(frame, model, confidence_threshold=0.7):
    # Görüntüyü modele uygun hale getir
    img_tensor = F.to_tensor(frame)
    img_tensor = img_tensor.unsqueeze_(0)
    
    # Tahmin yap
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # İnsanları filtrele (COCO sınıf 1: person)
    boxes = predictions[0]['boxes'][predictions[0]['labels'] == 1]
    scores = predictions[0]['scores'][predictions[0]['labels'] == 1]
    
    # Güven eşiğini uygula
    valid_indices = scores > confidence_threshold
    person_boxes = boxes[valid_indices]
    person_count = len(person_boxes)
    
    return person_count, person_boxes

# Ana uygulama fonksiyonu
def webcam_people_counter():
    # Modeli yükle
    model = load_model()
    
    # Webcam'i başlat (genellikle 0, bazen 1 olabilir)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Webcam açılamadı!")
        return
    
    # Pencere boyutunu ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS hesaplama için
    fps_start_time = datetime.now()
    fps_frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hata: Frame alınamadı")
                break
            
            # FPS hesapla
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps = fps_frame_count / (datetime.now() - fps_start_time).total_seconds()
                fps_start_time = datetime.now()
                fps_frame_count = 0
            
            # İnsanları tespit et
            count, boxes = detect_people(frame, model)
            
            # Tespit kutularını çiz
            for box in boxes:
                cv2.rectangle(frame, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             (0, 255, 0), 2)
            
            # Bilgileri ekrana yaz
            cv2.putText(frame, f"Insan Sayisi: {count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow('Webcam Insan Sayici', frame)
            
            # 'q' tuşu ile çıkış
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_people_counter()