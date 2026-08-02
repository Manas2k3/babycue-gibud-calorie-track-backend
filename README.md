# 🍽️ Food Detection Ensemble API with Nutrition Estimation

A FastAPI-based Food Detection API that combines predictions from **five domain-specific YOLO models** using **Weighted Boxes Fusion (WBF)** and estimates the **nutritional values** of detected food items.

The application provides a REST API, a browser-based upload interface, annotated prediction images, and nutritional analysis including calories, protein, carbohydrates, and fats.

---

## 🚀 Features

- 🍛 Multi-food detection
- 🧠 Five domain-specific YOLO models
- 🎯 Weighted Boxes Fusion (WBF)
- 🥗 Nutrition estimation
- 🔥 Calories estimation
- 🥩 Protein estimation
- 🍚 Carbohydrate estimation
- 🧈 Fat estimation
- 📦 REST API (FastAPI)
- 🌐 Swagger Documentation
- 🖼️ Browser Upload Interface
- 📊 Annotated prediction images
- ⚡ Fast inference
- 📁 Automatic output management

---

# 🏗️ System Architecture

```text
                           Input Food Image
                                  │
                                  ▼
                         Image Preprocessing
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Food Ensemble API     │
                    └─────────────────────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          ▼                       ▼                        ▼
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │ Indian Fast    │      │ Indian Food    │      │ IndianFoodNet  │
 │ Food YOLO      │      │ YOLO Model     │      │ -30 YOLO       │
 └────────────────┘      └────────────────┘      └────────────────┘
          │                       │                        │
          ▼                       ▼                        ▼
 ┌────────────────┐                                      ┌────────────────┐
 │ Soft Drinks    │                                      │ Western Food   │
 │ YOLO Model     │                                      │ YOLO Model     │
 └────────────────┘                                      └────────────────┘
          └───────────────────────┬────────────────────────┘
                                  │
                                  ▼
                 Weighted Boxes Fusion (WBF)
                                  │
                                  ▼
                 Confidence Threshold Filtering
                                  │
                                  ▼
               Nutrition Estimation Module
                                  │
                                  ▼
                  Nutrition Dataset Lookup
                                  │
                                  ▼
              Final Detection + Nutrition Result
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
        Annotated Output Images           JSON API Response
```

---

# 📂 Project Structure

```text
food_ensemble_api/
│
├── models/
│   ├── indian fast food/
│   │   └── best.pt
│   ├── indian_food/
│   │   └── best.pt
│   ├── indianfoodnet-30/
│   │   └── best.pt
│   ├── soft drinks/
│   │   └── best.pt
│   └── western food/
│       └── best.pt
│
├── pipeline/
│   ├── __init__.py
│   └── ensemble.py
│
├── data/
│   └── nutrition/
│       ├── indb_anuvaad_2024.csv
│       └── usda_fooddata_central.csv
│
├── outputs/
│
├── static/
│   └── index.html
│
├── main.py
├── nutrition.py
├── config.py
├── requirements.txt
├── README.md
├── run.bat
├── run.sh
└── .gitignore
```

---

# 🤖 Models

The project uses five specialized YOLO models.

| Model | Purpose |
|---------|----------------------------|
| Indian Fast Food | Indian snacks and street food |
| Indian Food | Traditional Indian cuisine |
| IndianFoodNet-30 | IndianFoodNet-30 dataset |
| Soft Drinks | Beverage detection |
| Western Food | Pizza, Burger, Sandwich, Pasta etc.|

---

# 🥗 Nutrition Estimation

After food detection, each detected food item is matched against the nutrition datasets.

Current datasets:

- Indian Food Composition Database (INDB Anuvaad 2024)
- USDA FoodData Central

The API estimates:

- Calories
- Protein
- Carbohydrates
- Fat

and also returns the **total nutritional summary** for the complete meal.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/food_ensemble_api.git

cd food_ensemble_api
```

Create virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Add Models

Place one trained YOLO model inside each directory.

```text
models/

indian fast food/
    best.pt

indian_food/
    best.pt

indianfoodnet-30/
    best.pt

soft drinks/
    best.pt

western food/
    best.pt
```

---

# ▶️ Run Application

Windows

```bash
run.bat
```

or

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Linux

```bash
chmod +x run.sh

./run.sh
```

---

# 🌐 API

Application

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

Health

```
http://127.0.0.1:8000/health
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Upload Interface |
| GET | `/health` | Health Check |
| GET | `/models` | Model Information |
| POST | `/predict` | Food Detection + Nutrition |
| DELETE | `/outputs/{run_id}` | Delete Prediction Results |

---

# Prediction Pipeline

1. Upload Image
2. Image Preprocessing
3. Run all five YOLO models
4. Collect raw detections
5. Weighted Boxes Fusion
6. Confidence filtering
7. Food identification
8. Nutrition lookup
9. Nutrition aggregation
10. Generate annotated images
11. Return JSON response

---

# Example Response

```json
{
  "fused_predictions": [
    {
      "class_name": "burger",
      "confidence": 0.96
    }
  ],
  "nutrition": {
    "items": [
      {
        "food": "burger",
        "calories": 295,
        "protein": 17,
        "carbohydrates": 30,
        "fat": 13
      }
    ],
    "total_calories": 295,
    "total_protein": 17,
    "total_carbohydrates": 30,
    "total_fat": 13
  }
}
```

---

# 🛠️ Tech Stack

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Ensemble Boxes
- Weighted Boxes Fusion (WBF)
- HTML
- JavaScript

---

# 🚀 Future Improvements

- ONNX deployment
- TensorRT optimization
- GPU inference
- Docker deployment
- Kubernetes deployment
- Mobile API
- Nutrition recommendation system
- Diet planning
- Meal calorie tracking

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Sudhanshu Sekhar Naik**

B.Tech Information Technology

AI/ML Engineer

**Food Detection Ensemble API using YOLO + Weighted Boxes Fusion + Nutrition Estimation**