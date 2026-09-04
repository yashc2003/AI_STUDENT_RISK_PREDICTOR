# 🎓 AI Student Performance Risk Prediction System

An intelligent machine learning web application that predicts academic risk levels (Low, Medium, High) for students and provides actionable, personalized intervention strategies to boost student retention and success.

---

## 📌 Features

- **🎯 Real-Time Risk Prediction**: Predicts student academic risk (`Low Risk`, `Medium Risk`, `High Risk`) using an optimized Random Forest classifier.
- **📊 Interactive Analytics Dashboard**: Built with Streamlit and Plotly for intuitive data exploration and risk factor analysis.
- **💡 Personalized Action Plans & Recommendations**: Automatically identifies key risk indicators (low attendance, backlogs, poor internal marks, etc.) and suggests tailored improvement steps.
- **📈 Model Explainability & Visualizations**: Displays feature importance plots, confusion matrices, and multi-dimensional distribution charts.
- **📁 Data Generation & Preprocessing Pipeline**: Includes synthetic dataset generation and scalable data preprocessing with standardization.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest Classifier)
- **Data Analysis & Preprocessing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 📁 Project Structure

```text
AI_STUDENT_RISK_PREDICTOR/
├── app.py                   # Streamlit web application dashboard
├── generate_dataset.py      # Script to generate synthetic student academic dataset
├── preprocess.py            # Data preprocessing & feature scaling pipeline
├── train_model.py           # Model training, evaluation & metrics logging
├── student_data.csv         # Generated student dataset
├── risk_model.pkl           # Trained Random Forest classifier model
├── scaler.pkl               # Standard scaler for input features
├── features.pkl             # Feature column names
├── confusion_matrix.png     # Model evaluation confusion matrix plot
├── feature_importance.png   # Feature importance visualization
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yashc2003/AI_STUDENT_RISK_PREDICTOR.git
cd AI_STUDENT_RISK_PREDICTOR
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Generate Data (Optional)
Generate a fresh synthetic student performance dataset:
```bash
python generate_dataset.py
```

### Train the Model
Preprocess data, train the Random Forest classifier, and generate evaluation plots:
```bash
python train_model.py
```

### Run the Web Dashboard
Launch the interactive Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the dashboard.

---

## 📊 Key Features & Indicators Used

The risk prediction model evaluates the following student features:
- **Attendance Percentage** (`%`)
- **Internal Assessment Marks** (`0 - 100`)
- **Previous Semester CGPA** (`0 - 10`)
- **Assignment Scores & Submission Rate** (`%`)
- **Daily Study Hours**
- **Active Backlogs / Arrears**
- **Average Sleep Hours**
- **Past Academic Failures**

---

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier (`n_estimators=200`, `class_weight='balanced'`)
- **Target Classes**: `Low Risk`, `Medium Risk`, `High Risk`
- Evaluated on test set with standard metrics (Accuracy, Precision, Recall, F1-Score).

---

## Project Deploy 
- https://aistudentriskpredictor-khjnsnecdramthevw5qovf.streamlit.app/

