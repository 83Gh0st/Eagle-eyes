import cv2
from yolo_predictions import YOLO_Pred

# Load YOLO model
try:
    yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Open webcam (0 = default, 1 = external camera)
cap = cv2.VideoCapture(0)

# Ensure the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()  # Read frame from webcam

    if not ret:
        print('Unable to read frame from webcam, retrying...')
        continue  # Retry instead of breaking immediately

    # Resize frame to match YOLO input size if necessary (e.g., 640x640)
    frame_resized = cv2.resize(frame, (640, 640))

    try:
        # Perform YOLO prediction
        pred_image = yolo.predictions(frame_resized)
    except Exception as e:
        print(f"Error during YOLO prediction: {e}")
        break  # Stop loop if YOLO fails

    # Display output
    cv2.imshow('YOLO Webcam Detection', pred_image)

    # Press 'ESC' to exit
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
