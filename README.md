# 🩺 Diabetes Detection System — Ensemble Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Ensemble-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-9ACD32)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

**Explore a diabetes dataset, compare individual machine-learning models with ensemble methods, and experiment with an interactive prediction interface.**

The **Diabetes Detection System** is a Streamlit application built with Python and Scikit-learn. It uses the Pima Indians Diabetes dataset to demonstrate binary classification with six model configurations: LightGBM, K-Nearest Neighbors (KNN), Extra Trees, Logistic Regression, Soft Voting, and Stacking.

The application includes dataset exploration, feature visualizations, model evaluation, ROC curves, and an interactive prediction form. Models are tuned on training data and evaluated against a separate held-out test set.

> **Educational-use notice:** This is a portfolio and machine-learning demonstration, not a medical diagnostic tool, clinically validated risk calculator, or substitute for professional medical advice or laboratory testing. Do not use its output to make healthcare decisions.

---

## ✨ Features

### 📊 Dataset Explorer
- Preview records from the local `diabetes.csv` file.
- View the number of records and input features.
- Inspect the distribution of Outcome 0 and Outcome 1.
- Review missing-value counts after zero-placeholder handling.
- Explore descriptive statistics for the dataset.

### 🔎 Feature Insights
- Compare glucose distributions by outcome class.
- Compare BMI distributions by outcome class.
- Explore a correlation heatmap covering the input features and target.

### 🏆 Ensemble Model Lab
Compare six model configurations using the same stratified train/test split:

| Model | Role in the application |
|---|---|
| LightGBM | Gradient-boosted decision-tree classifier; hyperparameter-tuned |
| K-Nearest Neighbors | Distance-based classifier; hyperparameter-tuned |
| Extra Trees | Randomized tree ensemble; hyperparameter-tuned |
| Logistic Regression | Class-balanced baseline model |
| Soft Voting Ensemble | Combines LightGBM, KNN, and Extra Trees using predicted probabilities |
| Stacking Ensemble | Learns a second-level combination of the three base learners using Logistic Regression |

The comparison includes:
- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix and classification report for the selected model
- ROC curve comparison across all models
- Cross-validation ROC-AUC scores for the tuned base learners

### 🧪 Prediction Studio
- Enter the eight measurements used by the model.
- Select a model from the sidebar.
- Adjust the probability classification threshold.
- View the predicted dataset class and the model score for Outcome 1.
- Download a Joblib bundle containing the trained models and feature metadata.

The displayed score is not necessarily a calibrated probability or an individual clinical risk estimate.

---

## 🧠 Machine Learning Workflow

### 1. Dataset and input features

The application expects a CSV named `diabetes.csv` containing the following columns:

| Column | Description |
|---|---|
| `Pregnancies` | Number of pregnancies recorded in the dataset |
| `Glucose` | Plasma glucose measurement |
| `BloodPressure` | Diastolic blood pressure measurement |
| `SkinThickness` | Triceps skin-fold thickness measurement |
| `Insulin` | Two-hour serum insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function value |
| `Age` | Age in years |
| `Outcome` | Target class: `0` or `1` |

The CSV supplied with this project has **768 records and 9 columns**. The observed target counts in that file are 500 records for Outcome 0 and 268 for Outcome 1. These are dataset counts, not estimates of population prevalence.

### 2. Data validation and preprocessing

When the app loads the CSV, it:
1. Checks that all required columns are present.
2. Converts the expected columns to numeric values.
3. Removes rows with a missing or invalid target value.
4. Keeps records whose `Outcome` is either 0 or 1.
5. Treats zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing-value placeholders.

`Pregnancies` is not included in this zero-placeholder treatment because zero pregnancies can be a valid value.

Missing input values are imputed with the median inside each model pipeline. KNN and Logistic Regression also use `StandardScaler`; the tree-based models do not require feature scaling in this implementation.

### 3. Train/test split

The app uses an **80/20 stratified train/test split** with `random_state=42`. Stratification helps preserve the target-class proportions in both partitions.

The held-out test partition is kept separate from hyperparameter search. It is used to calculate the displayed test metrics after model training.

### 4. Hyperparameter tuning

LightGBM, KNN, and Extra Trees are tuned with `RandomizedSearchCV`:
- 12 sampled parameter combinations per base model (`n_iter=12`).
- 5-fold `StratifiedKFold` cross-validation.
- ROC-AUC as the search scoring metric.
- Fixed random seed for repeatability.

The Logistic Regression baseline is trained with class balancing but is not included in the randomized hyperparameter search.

### 5. Ensemble learning

**Soft Voting** combines the class probabilities produced by the tuned LightGBM, KNN, and Extra Trees learners. The current implementation gives the three learners equal voting weights.

**Stacking** uses those three learners as base estimators and a class-balanced Logistic Regression model as the final estimator. The final estimator learns how to combine the base-model predictions using cross-validation.

### 6. Evaluation and thresholding

The application reports accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC. A sidebar threshold controls how the model's Outcome 1 score is converted into the predicted class. Changing this threshold changes class predictions and threshold-dependent metrics; ROC-AUC is based on the underlying scores and does not change with the threshold.

The test metrics are estimates from this dataset and this particular split. They are not clinical validation results.

---

## 🧰 Technology Stack

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | CSV loading and tabular data handling |
| NumPy | Numerical operations |
| Matplotlib | Charts and visualizations |
| Seaborn | Statistical plots and heatmaps |
| Scikit-learn | Pipelines, preprocessing, tuning, ensembles, and evaluation |
| LightGBM | Gradient-boosted tree classifier |
| Joblib | Downloadable model-bundle serialization |

The exact dependency ranges are listed in [`requirements.txt`](requirements.txt).

---

## 📁 Project Structure

```text
Diabetes-Detection-System/
│
├── app.py                  # Main Streamlit application and ML workflow
├── diabetes.csv            # Local dataset used by the application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── .gitignore              # Files and folders excluded from Git
│
└── .streamlit/
    └── config.toml         # Streamlit theme and server configuration
```

The app resolves the dataset path relative to `app.py`, so it can be run from another working directory as long as `diabetes.csv` remains beside the main application file.

---

## 💻 Installation and Setup

### Prerequisites
- Python 3.10 or newer is recommended.
- pip package manager.
- The `diabetes.csv` file placed in the project root beside `app.py`.

### 1. Clone the repository

Replace the placeholder with your GitHub repository URL:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Diabetes-Detection-System
```

Or download the repository and open the extracted project folder in VS Code.

### 2. Create and activate a virtual environment

**Windows (Command Prompt / PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Git Bash:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

LightGBM is a required dependency for this application because it is used as one of the base learners in both ensemble models.

### 4. Check the dataset file

Ensure the project root contains:

```text
diabetes.csv
```

The filename must match exactly. The application displays an error if the file is missing or if required columns are absent.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local address printed in the terminal. The default is usually:

```text
http://localhost:8501
```

The first run can take longer because the app tunes three base models with randomized cross-validation, then fits the voting and stacking ensembles. Streamlit caches the loaded data and trained model resources for reuse within the session.

---

## ⚙️ Configuration and Repository Notes

- `.streamlit/config.toml` configures the dark theme and headless server mode.
- `.gitignore` excludes Python cache files, virtual environments, Streamlit secrets, notebook checkpoints, generated model artifacts, IDE settings, OS files, and logs.
- `LICENSE` contains the MIT License for this project code.
- The MIT License for the code does not automatically grant rights to redistribute the dataset. Check the dataset's original source and applicable terms before publishing the CSV in a public repository.

---

## ⚠️ Limitations and Responsible Use

- The dataset is small and historically specific. It represents women aged 21 or older from the Pima Indian population and is not representative of all people or populations.
- Results from this dataset should not be generalized to real-world clinical populations.
- Zero values in selected physiological measurement columns are interpreted as missing placeholders, based on the dataset's convention. This preprocessing assumption may not be suitable for other datasets.
- Hyperparameter search optimizes ROC-AUC on training folds; this does not guarantee the best accuracy or clinical utility.
- A single held-out split can produce variable results. Additional external validation would be needed before making claims about generalization.
- Model scores may not be calibrated probabilities.
- The app does not diagnose diabetes, recommend treatment, or replace a clinician's assessment.

**This project is for educational, experimentation, and portfolio demonstration only. It must not be used to diagnose, screen, treat, or rule out diabetes, or to make healthcare decisions. Consult a qualified healthcare professional for medical concerns.**

---

## 🔮 Possible Future Improvements

- Add repeated or nested cross-validation to obtain a more robust estimate of model performance.
- Add calibration analysis and probability calibration when appropriate.
- Compare precision-recall curves and threshold trade-offs.
- Add feature-importance and model-explanation views, with suitable caveats.
- Add an external validation dataset to assess generalization.
- Add automated tests for data validation, preprocessing, model training, and predictions.
- Add model export metadata, package versions, and reproducible training configuration.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📄 License

The project code is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full terms. Dataset rights and terms are separate and remain with the relevant dataset owner or distributor.

---

<p align="center">
  Built with Python 🐍 · Streamlit ⚡ · Ensemble Machine Learning 🤖
</p>
<p align="center">
  ⭐ If you find this project useful, consider starring the repository.
</p>
