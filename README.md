
#    Eagle-eyes | Smart Construction Site Monitoring - PPE Detection 🏗️🦺  

![Smart Construction Monitoring](https://github.com/83Gh0st/Eagle-eyes/blob/main/predictions/Model/val_batch1_pred.jpg?raw=true)  

## **📌 Overview**  
Eagle Eyes project is an AI-powered **Personal Protective Equipment (PPE) detection system** designed for **smart construction site monitoring**. It can detect **workers wearing or not wearing helmets (hat/no hat) and safety vests (vest/no vest)** in real-time using deep learning. The model is optimized for deployment across **multiple platforms** using:  

✅ **ONNX** (Optimized for CPU/GPU Inference)  
✅ **TensorFlow Lite (TFLite)** (Edge AI & Mobile Applications)  
✅ **TensorFlow.js (TF.js)** (Web-Based Real-Time Monitoring)  

### **🔹 Use Case**
- **Automated PPE compliance checks** on construction sites.  
- **Real-time monitoring** via webcam, CCTV, or drone cameras.  
- **Improves worker safety** and **reduces workplace hazards**.  
- **Deployable across edge devices, cloud, or web applications**.  

---

## **🖥️ Demo**
### **📌 Run on Image Input**
```python
import cv2
from yolo_predictions import YOLO_Pred

# Load Model
yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

# Load Image
img = cv2.imread('test_image.jpg')

# Perform Detection
img_pred = yolo.predictions(img)

# Display Results
cv2.imshow('PPE Detection', img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### **📌 Run Real-Time Detection on Webcam**
```python
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred_image = yolo.predictions(frame)
    cv2.imshow('PPE Detection', pred_image)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **🔧 Model Variants & Deployment Options**
| Model Format | Use Case |  
|-------------|----------|  
| **ONNX** | High-performance inference on **CPU/GPU** devices |  
| **TFLite** | Optimized for **mobile, edge, and embedded systems** |  
| **TF.js** | Runs in a **browser for real-time monitoring** |  

---

## **📊 Model Performance & Results**  
The model has been evaluated on a **construction site dataset** and achieved:  

✅ **mAP (Mean Average Precision):** 89.6%  
✅ **Precision:** 92.1%  
✅ **Recall:** 87.5%  

![precison curve](https://github.com/83Gh0st/Eagle-eyes/blob/main/predictions/Model/P_curve.png?raw=true)

📈 **Graphs & Training Metrics** (Loss, Accuracy, Confusion Matrix) are available in the results folder.

![confusion matrix](https://github.com/83Gh0st/Eagle-eyes/blob/main/predictions/Model/confusion_matrix.png)
---

## **📂 Project Structure**
```
├── Model/
│   ├── weights/
│   │   ├── best.onnx  # ONNX Model
│   │   ├── best.tflite  # TFLite Model
│   │   ├── best_web_model/  # TF.js Model
│   ├── results/  # Training graphs and evaluation results
│
├── predictions/
│   ├── detect.py  # Image/Video/Webcam detection script
│   ├── yolo_predictions.py  # YOLO inference class
│   ├── utils.py  # Helper functions
│
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── data.yaml  # Dataset configuration
│
├── README.md  # Project Documentation
```

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/83Gh0st/Eagle-eyes.git
cd Eagle-eyes
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Object Detection**
#### **🔹 On Images**
```bash
python3 detect.py --source test_image.jpg
```
#### **🔹 On Webcam**
```bash
python3 detect.py --source 0
```

---

## **📦 Model Conversion & Deployment**
### **Convert PyTorch Model to ONNX**
```python
import torch

model = torch.load('best.pt', map_location='cpu')
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, "best.onnx", opset_version=11)
```

### **Convert ONNX to TensorFlow Lite**
```bash
onnx-tf convert -i best.onnx -o best.pb
tflite_convert --saved_model_dir=best.pb --output_file=best.tflite
```

### **Convert TensorFlow Model to TF.js**
```bash
tensorflowjs_converter --input_format=tf_saved_model best.pb best_web_model/
```

---

## **📌 Deployment on Web Using TensorFlow.js**
1️⃣ Copy the `best_web_model/` to your web server.  
2️⃣ Load the model in JavaScript:  
```js
const model = await tf.loadGraphModel('best_web_model/model.json');
const img = tf.browser.fromPixels(document.getElementById('input_image'));
const predictions = model.predict(img);
```

---

## **🛠️ Future Improvements**
🔹 **Integrate IoT for automatic safety alerts**.  
🔹 **Deploy as a cloud-based AI API**.  
🔹 **Enhance dataset with more PPE variations**.  
🔹 **Optimize for edge devices like NVIDIA Jetson Nano**.  

---

## **📜 License**
This project is licensed under the **MIT License** – Free to use, modify, and distribute.  

---

## **👨‍💻 Author**
Developed by **@83Gh0st** (Arjun Sasikumar) 🔥  
💬 **Contact:** [GitHub](https://github.com/83Gh0st)  

🔥 **Star this repo if you found it useful!** ⭐  
