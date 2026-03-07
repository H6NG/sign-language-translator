import cv2# type: ignore 

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
