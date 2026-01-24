# 🛡️ Network Security – Machine Learning Pipeline

An end-to-end **Machine Learning pipeline** for network security analysis, built with a **production-ready, modular architecture**.  
This project demonstrates how real-world ML systems are structured, trained, evaluated, and versioned.

---

## 🚀 Project Overview

This project focuses on detecting suspicious or malicious network behavior using classical machine learning models.  
The pipeline is fully automated and follows best practices used in **industry-grade ML systems**.

### Key Highlights
- End-to-end ML pipeline
- Config-driven architecture
- Automated artifact generation
- Hyperparameter tuning with GridSearchCV
- Clean separation of components
- Scalable and maintainable codebase

---

## 🧱 Project Architecture

```
NetworkSecurity/
│
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── constant/
│   │   └── training_pipeline.py
│   │
│   ├── utils/
│   │   └── main_utils/
│   │       └── utils.py
│   │
│   └── exception/
│       └── exception.py
│
├── Artifacts/
│   └── <timestamped pipeline outputs>
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔄 Machine Learning Pipeline Flow

### 1️⃣ Data Ingestion
- Loads raw network data
- Splits data into training and testing sets
- Stores ingested data as pipeline artifacts

### 2️⃣ Data Validation
- Validates schema
- Checks data consistency
- Performs data drift detection
- Generates drift reports

### 3️⃣ Data Transformation
- Feature preprocessing
- Saves transformed train and test datasets (`.npy`)
- Saves preprocessing object (`.pkl`)

### 4️⃣ Model Training & Selection
- Trains multiple ML models
- Performs hyperparameter tuning using GridSearchCV
- Selects best-performing model
- Saves trained model as an artifact

---

## 🤖 Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  

> Model selection is fully automated using GridSearchCV.

---

## 📊 Evaluation Metrics

- Accuracy Score  

> The pipeline can be easily extended to include:
- F1-score  
- Precision / Recall  
- ROC-AUC  

---

## 📁 Artifacts Generated

Each pipeline execution creates a **timestamped directory** inside `Artifacts/` containing:

- Ingested train & test datasets  
- Validation reports  
- Transformed datasets  
- Preprocessing object  
- Trained ML model  

This ensures full experiment traceability.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** NumPy, Pandas  
- **Pipeline Design:** Modular OOP  
- **Version Control:** Git & GitHub  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/souravkumar-cloud/NetworkSecurity.git
cd NetworkSecurity
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Pipeline
```bash
python main.py
```

---

## 📝 Optional (but strongly recommended)

Add this to your `.gitignore` file:

```gitignore
venv/
Artifacts/
*.pyc
__pycache__/
.DS_Store
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---


## 👤 Author

**Sourav Kumar**  
GitHub: [@souravkumar-cloud](https://github.com/souravkumar-cloud)

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!