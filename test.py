from ultralytics import YOLO
import cv2
import numpy as np

# Load model segmentasi YOLOv8
model = YOLO("best.pt")  # Ganti dengan path ke model yang sudah dibuat
class_names = model.names  # Ambil nama label kelas dari model

# Buka file video
cap = cv2.VideoCapture('p.mp4')  # Arahkan video ke path yang sesuai

# Ambil FPS dari video agar diputar dalam kecepatan yang sama
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33  # Konversi ke delay dalam milidetik

# Hitung frame untuk skip (jika diperlukan)
count = 0

# Loop untuk membaca frame video
while True:
    ret, img = cap.read()  # Baca satu frame
    if not ret:
        break  # Keluar dari loop jika frame tidak tersedia (akhir video)

    count += 1

    # Resize frame agar tidak terlalu besar untuk diproses/dilihat
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape  # Simpan tinggi dan lebar frame

    # Prediksi dengan YOLOv8
    results = model.predict(img)

    # Loop hasil prediksi
    for r in results:
        boxes = r.boxes      # Bounding box hasil deteksi
        masks = r.masks      # Segmentasi mask hasil deteksi (None jika tidak ada)

        # Jika segmentasi tersedia
        if masks is not None:
            # Ambil data mask dan pindahkan ke CPU, ubah jadi numpy array
            masks = masks.data.cpu().numpy()

            # Loop setiap hasil mask dan box (satu objek per iterasi)
            for seg, box in zip(masks, boxes):
                # Resize mask agar cocok dengan ukuran frame
                seg = cv2.resize(seg, (w, h))

                # Konversi mask ke uint8 dan threshold untuk binerisasi
                seg_uint8 = (seg * 255).astype(np.uint8)
                _, thresh = cv2.threshold(seg_uint8, 127, 255, cv2.THRESH_BINARY)

                # Temukan kontur dari hasil segmentasi
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Loop setiap kontur untuk digambar
                for contour in contours:
                    d = int(box.cls)          # Index kelas deteksi (misalnya 0, 1, dst)
                    c = class_names[d]        # Nama kelas berdasarkan index
                    x, y, w_box, h_box = cv2.boundingRect(contour)  # Ambil posisi label

                    # Gambar kontur dengan garis merah
                    cv2.polylines(img, [contour], True, (0, 0, 255), 2)

                    # Tampilkan label kelas di atas kontur
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

    # Tampilkan hasil frame yang sudah dianotasi
    cv2.imshow('img', img)

    # Tunggu sesuai delay (agar sesuai FPS asli); tekan 'q' untuk keluar
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Setelah selesai, lepas video dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
