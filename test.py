from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('p.mp4')
count = 0

pothole_count = 0  # Inisialisasi jumlah lubang di frame ini

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)


    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu().numpy()
            for seg, box in zip(masks, boxes):
                seg = cv2.resize(seg, (w, h))
                seg_uint8 = (seg * 255).astype(np.uint8)
                _, thresh = cv2.threshold(seg_uint8, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    pothole_count += 1  # Tambah jumlah lubang
                    d = int(box.cls)
                    c = class_names[d]
                    label = f"{c} #{pothole_count}"  # Tambah nomor ke label
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, (0, 0, 255), 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Tampilkan total jumlah pothole di pojok kiri atas
    # cv2.putText(img, f"Total potholes: {pothole_count}", (20, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('memainkan video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
