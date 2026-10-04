# ✈️ Flight Delay Prediction System

An end-to-end **Machine Learning-based Flight Delay Prediction System** built with Python, Scikit-learn, and Streamlit.

The project uses a **Random Forest Classifier** to predict whether a flight is likely to be **Delayed** or **On Time** based on flight characteristics such as airline, route, distance, scheduled departure/arrival time, day, month, weather conditions, and airport traffic level.

The project also includes an interactive Streamlit dashboard for making predictions, exploring the dataset, viewing model performance, analyzing feature importance, and maintaining a session-based prediction history.

---

## 🚀 Project Overview

Flight delays can be influenced by several factors, including weather conditions, airport traffic, flight distance, departure time, and seasonal patterns.

This project demonstrates how Machine Learning can be used to analyze these factors and build a classification model capable of predicting flight delay status.

### Workflow

```text
Dataset Generation / Loading
          ↓
Data Cleaning
          ↓
Categorical Encoding
          ↓
Train-Test Split
          ↓
Random Forest Classification
          ↓
Model Evaluation
          ↓
Feature Importance & Confusion Matrix
          ↓
Save Trained Model
          ↓
Streamlit Dashboard
          ↓
Flight Delay Prediction
```

---

## ✨ Features

- ✈️ Flight delay prediction using Machine Learning
- 🌲 Random Forest Classification
- 📊 Interactive Streamlit dashboard
- 🧹 Automated data cleaning
- 🔤 Categorical feature encoding
- 📈 Model accuracy evaluation
- 📋 Classification report
- 🔲 Confusion matrix visualization
- 📊 Feature importance visualization
- 🎲 Random flight sample prediction
- 📁 Custom CSV dataset upload
- 🧠 Model performance dashboard
- 📝 Session-based prediction history
- ⬇️ Download trained Machine Learning model

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Random Forest | Classification model |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Pickle | Model serialization |

---

## 📂 Project Structure

```text
Flight_Delay_Prediction/
│
├── app.py
├── train_model.py
├── flights.csv
├── model.pkl
├── confusion_matrix.png
├── feature_importance.png
└── README.md
```

### File Description

**`train_model.py`**

Contains the complete Machine Learning pipeline:

- Dataset generation/loading
- Data cleaning
- Categorical encoding
- Train-test splitting
- Random Forest model training
- Model evaluation
- Feature importance visualization
- Confusion matrix generation
- Model serialization

**`app.py`**

Contains the Streamlit web application used to:

- Enter flight information
- Predict flight delays
- Explore the dataset
- View model performance
- View feature importance
- Generate random flight predictions
- Upload a custom CSV
- View prediction history

**`flights.csv`**

The generated flight dataset used by the Machine Learning pipeline.

**`model.pkl`**

Serialized trained model bundle containing:

- Random Forest model
- Label encoders
- Feature columns
- Accuracy
- Confusion matrix
- Classification report

**`feature_importance.png`**

Visualization showing the relative importance of features used by the Random Forest model.

**`confusion_matrix.png`**

Visualization showing the model's classification performance for On-Time and Delayed flights.

---

# 📊 Dataset

The project generates a synthetic dataset containing **6,000 flight records** when `flights.csv` does not already exist.

The dataset contains the following features:

| Feature | Description |
|---|---|
| `Airline` | Airline operating the flight |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Distance` | Flight distance in miles |
| `Scheduled_Departure` | Scheduled departure hour |
| `Scheduled_Arrival` | Scheduled arrival hour |
| `Day_of_Week` | Day of the week |
| `Month` | Month of the flight |
| `Weather` | Weather condition |
| `Traffic_Level` | Airport traffic level |
| `Delay` | Target variable: 0 = On Time, 1 = Delayed |

The synthetic data is generated using logical relationships between weather, traffic, departure time, seasonality, flight distance, and delay probability.

---

# 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** for binary classification.

### Target

```text
Delay

0 → On Time
1 → Delayed
```

### Input Features

```text
Airline
Source
Destination
Distance
Scheduled_Departure
Scheduled_Arrival
Day_of_Week
Month
Weather
Traffic_Level
```

Categorical variables are transformed using **Label Encoding** before training.

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

The dataset is divided into **80% training data and 20% testing data** using stratified sampling.

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

The project also generates:

```text
feature_importance.png
confusion_matrix.png
```

These visualizations are automatically created during model training.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Asifkhan629/Flight_Delay_Prediction.git
```

Move into the project directory:

```bash
cd Flight_Delay_Prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

---

# ▶️ Execution

## Step 1 — Train the Model

Run:

```bash
python train_model.py
```

The training script will:

1. Load `flights.csv` if it already exists.
2. Generate the dataset if it does not exist.
3. Clean the dataset.
4. Encode categorical variables.
5. Split the dataset into training and testing sets.
6. Train the Random Forest classifier.
7. Evaluate the model.
8. Generate the feature importance graph.
9. Generate the confusion matrix.
10. Save the trained model to `model.pkl`.

The script is designed to be executed **before launching the Streamlit application**.

---

## Step 2 — Launch the Streamlit Application

After model training is complete, run:

```bash
streamlit run app.py
```

Streamlit will start the local web application.

Open the local URL displayed in your terminal, typically:

```text
http://localhost:8501
```

---

# 🖥️ Using the Application

The dashboard contains four main sections:

### 🏠 1. Predict Delay

Enter flight information such as:

- Airline
- Source city
- Destination city
- Distance
- Scheduled departure
- Scheduled arrival
- Day of week
- Month
- Weather condition
- Airport traffic level

Then click:

```text
🔮 Predict Delay
```

The application uses the trained Random Forest model to generate the prediction.

---

### 📊 2. Dataset Insights

Explore information about the dataset, including flight statistics and delay-related information.

---

### 🧠 3. Model Performance

View Machine Learning performance metrics such as:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

The trained model and its evaluation metrics are stored together in `model.pkl` for use by the application.

---

### 📁 4. Prediction History

The application maintains a session-based history of predictions and allows the prediction results to be downloaded as a CSV file.

---

# 🎲 Random Flight Prediction

The application also provides a **Generate Random Flight** option.

This selects a random flight record from the dataset and automatically uses its characteristics for prediction, making it easier to test the application without manually entering every value.

---

# 📤 Custom Dataset

The Streamlit application allows users to optionally upload their own flight CSV dataset.

```text
Upload your own flights CSV
        ↓
Dataset loaded for the current session
        ↓
Explore / analyze the uploaded data
```

The application also provides an option to download the trained `model.pkl` file.

---

# 📌 Important Note

This project uses a **synthetically generated flight dataset** for demonstrating the Machine Learning workflow. The synthetic data is designed with logical relationships between weather, traffic, distance, departure time, and flight delays.

Therefore, the model should be considered a **Machine Learning demonstration/project**, rather than a production system for predicting real-world flight delays.

---

# 🔮 Future Improvements

Possible improvements include:

- Use real-world flight delay datasets
- Add real-time weather APIs
- Integrate live flight information
- Compare Random Forest with XGBoost, Logistic Regression and other models
- Perform hyperparameter optimization
- Add cross-validation
- Add ROC-AUC and Precision-Recall curves
- Improve feature engineering
- Deploy the application online
- Add explainable AI using SHAP
- Build an API for model predictions

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Categorical encoding
- Supervised Machine Learning
- Random Forest classification
- Model evaluation
- Data visualization
- Model serialization
- Streamlit application development
- End-to-end Machine Learning workflow

---

# 👨‍💻 Author

**Asif Khan**

GitHub:  
https://github.com/Asifkhan629

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
