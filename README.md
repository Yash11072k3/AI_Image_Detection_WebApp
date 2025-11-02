# 🚀 AI-Generated Human Face Image Detection Web App  
Detect whether an human face image is real or AI-generated using deep learning and hybrid ensemble models.

---

## 🧠 Project Overview  
This project combines multiple deep learning models — **CNN**, **ResNet**, **DIF**, **UFD**, and **PatchSelection** — to detect AI-generated human face images.  
The final prediction is made using a **weighted hybrid voting system**, providing high accuracy across various datasets.

---

## ✨ Features
- 🖼️ Upload an image to check if it’s AI-generated or real  
- ⚙️ Ensemble hybrid model with weighted voting  
- 📊 Visualized confusion matrix and per-class metrics  
- 🧾 Classification reports and model performance comparison  
- 🌐 Streamlit-based user interface  

---

## 🧩 Tech Stack
| Category | Tools Used |
|-----------|-------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Frontend | Streamlit |
| Database | SQLite |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud / Render |

---


## 🧩 Project Structure
src/
│── code/
│ ├── dataloader.py
│ ├── evaluate_hybrid.py
│ ├── inference.py
│ ├── model_definitions.py
│ ├── webapp.py
│── checkpoints/
│── newly_trained_model/
│── evaluate_hybrid.ipynb
requirements.txt
Procfile


---

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/your-username/AI_Image_Detection_WebApp.git

# Navigate into the folder
cd AI_Image_Detection_WebApp

# Install dependencies
pip install -r requirements.txt

▶️ Run the Web App
python src/code/webapp.py

or if using Streamlit:

streamlit run src/code/webapp.py
```
---

🧠 Models Used

- CNN
- ResNet18
- PatchSelection
- DIF
- UFD
- Hybrid Weighted Voting

---

📈 Results

- Average Accuracy: 98.3%
- Confusion Matrix and classification reports included

---

👨‍💻 Author

- Yashwanth Seshathri
- B.Tech Artificial Intelligence & Data Science
- 📍 Coimbatore, India
-  yashwanthyash1107@gmail.com
