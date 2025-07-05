import cv2
import numpy as np
import os
import requests
from collections import defaultdict
from ultralytics import YOLO

def download_sample_video():
    """Alternatif örnek video indirme fonksiyonu"""
    try:
        print("Örnek video indiriliyor...")
        url = "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi"
        local_path = "sample_video.avi"
        
        if not os.path.exists(local_path):
            response = requests.get(url, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
        return local_path
    except Exception as e:
        print(f"Örnek video indirme hatası: {e}")
        return None

def initialize_video_source():
    """Video kaynağını başlatma fonksiyonu"""
    # 1. Önce yerel video dosyalarını kontrol et
    video_files = ["mall_video.mp4", "store_video.mp4"]
    for video_file in video_files:
        if os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                print(f"{video_file} kullanılıyor...")
                return cap, video_file
    
    # 2. Örnek video indirmeyi dene
    sample_path = download_sample_video()
    if sample_path:
        cap = cv2.VideoCapture(sample_path)
        if cap.isOpened():
            print(f"İndirilen örnek video kullanılıyor: {sample_path}")
            return cap, sample_path
    
    # 3. Webcam'e dön
    print("Video dosyası bulunamadı, webcam kullanılıyor...")
    return cv2.VideoCapture(0), "webcam"

def main():
    # Model yükleme
    model = YOLO('yolov8n.pt')  # Nesne tespiti için YOLOv8

    # Video kaynağını başlat
    cap, video_source = initialize_video_source()
    if not cap.isOpened():
        raise RuntimeError("Video kaynağı açılamadı")

    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video Kaynağı: {video_source}")
    print(f"Çözünürlük: {width}x{height}, FPS: {fps:.2f}")

    # Heatmap için hazırlık
    heatmap = np.zeros((height, width), dtype=np.float32)
    track_history = defaultdict(lambda: [])

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video akışı sona erdi")
                break
            
            # İnsan tespiti ve takip
            results = model.track(frame, persist=True, classes=[0])  # 0: person sınıfı
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center = (int(x), int(y))
                    cv2.circle(heatmap, center, 20, (1), -1)
                    track_history[track_id].append(center)
            
            # Görselleştirme
            display_frame = frame.copy()
            heatmap_display = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_display = cv2.applyColorMap(heatmap_display.astype(np.uint8), cv2.COLORMAP_JET)
            display_frame = cv2.addWeighted(display_frame, 0.7, heatmap_display, 0.3, 0)
            
            cv2.imshow('Mağaza Takip Sistemi', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Kaynakları temizle
        cap.release()
        cv2.destroyAllWindows()
        
        # Heatmap kaydet
        try:
            final_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            final_heatmap = cv2.applyColorMap(final_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite('store_heatmap.jpg', final_heatmap)
            print("Heatmap başarıyla kaydedildi: store_heatmap.jpg")
        except Exception as e:
            print(f"Heatmap kaydetme hatası: {e}")

if __name__ == "__main__":
    main()