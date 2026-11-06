#############################################################################
#
#   Importing required libraries
#
#############################################################################
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


#############################################################################
#
#   Global configuration
#
#############################################################################
APP_TITLE = "Diabetes Detection System"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"  # Keep diabetes.csv beside app.py
AUTHOR = "Shivam Ramesh Kurlekar"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 12

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

MODEL_DESCRIPTIONS = {
    "Soft Voting Ensemble": (
        "Combines LightGBM, K-Nearest Neighbors and Extra Trees. "
        "The models vote using their predicted class probabilities."
    ),
    "Stacking Ensemble": (
        "Learns how to combine LightGBM, K-Nearest Neighbors and Extra Trees "
        "using a Logistic Regression meta-model."
    ),
    "LightGBM": "Gradient-boosted decision trees with tuned hyperparameters.",
    "K-Nearest Neighbors": "Classifies a record using nearby training examples.",
    "Extra Trees": "An ensemble of randomized decision trees.",
    "Logistic Regression (Baseline)": "A simple baseline used for comparison.",
}


#############################################################################
#
#   Function name:      inject_custom_css
#   Input:              None
#   Description:        Applies a responsive, healthcare-inspired visual theme.
#   Output:             None
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def inject_custom_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #071521 0%, #102638 52%, #123b43 100%);
                color: #edf7f6;
            }
            [data-testid="stSidebar"] {
                background: #0a1c2a;
                border-right: 1px solid rgba(45, 212, 191, 0.20);
            }
            .hero-title {
                text-align: center;
                font-size: clamp(2rem, 5vw, 3.25rem);
                font-weight: 900;
                letter-spacing: -1.2px;
                background: linear-gradient(90deg, #67e8f9, #5eead4, #a7f3d0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0;
            }
            .hero-subtitle {
                text-align: center;
                color: #b7cbd2;
                font-size: 1.02rem;
                margin-top: 0.4rem;
                margin-bottom: 1.4rem;
            }
            .notice-card {
                background: rgba(74, 52, 19, 0.28);
                border: 1px solid rgba(251, 191, 36, 0.42);
                border-radius: 13px;
                padding: 15px 17px;
                color: #fde7a4;
                margin: 12px 0 18px 0;
            }
            .section-card {
                background: rgba(14, 39, 54, 0.72);
                border: 1px solid rgba(94, 234, 212, 0.15);
                border-radius: 14px;
                padding: 15px;
            }
            div[data-testid="stMetric"] {
                background: rgba(12, 44, 57, 0.72);
                border: 1px solid rgba(94, 234, 212, 0.18);
                padding: 12px;
                border-radius: 12px;
            }
            .stButton button, .stDownloadButton button {
                border-radius: 10px;
                font-weight: 700;
            }
            div[data-testid="stForm"] {
                background: rgba(12, 35, 49, 0.55);
                border: 1px solid rgba(94, 234, 212, 0.18);
                border-radius: 14px;
                padding: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


#############################################################################
#
#   Function name:      load_data
#   Input:              data_path (Path or string)
#   Description:        Loads, validates and prepares the diabetes dataset.
#   Output:             Cleaned pandas DataFrame
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data(show_spinner=False)
def load_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = [str(column).strip() for column in df.columns]

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()

    # In this dataset, zero is used as a missing-value placeholder for these
    # physiological measurements. Zero pregnancies remains a valid value.
    for column in ZERO_AS_MISSING:
        df.loc[df[column] == 0, column] = np.nan

    if df.empty:
        raise ValueError("No usable records were found in the dataset.")
    if df[TARGET].nunique() != 2:
        raise ValueError("The Outcome column must contain both 0 and 1 classes.")

    return df.reset_index(drop=True)


#############################################################################
#
#   Function name:      build_model_pipelines
#   Input:              None
#   Description:        Builds preprocessing pipelines and baseline estimators
#                       for LightGBM, KNN, Extra Trees and Logistic Regression.
#   Output:             Model dictionary and hyperparameter search spaces
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def build_model_pipelines():
    if LGBMClassifier is None:
        raise ImportError(
            "LightGBM is not installed. Run: pip install lightgbm"
        )

    lightgbm_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                    n_jobs=1,
                ),
            ),
        ]
    )

    knn_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("scaler", StandardScaler().set_output(transform="pandas")),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    extra_trees_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            (
                "classifier",
                ExtraTreesClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    baseline_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("scaler", StandardScaler().set_output(transform="pandas")),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    search_spaces = {
        "LightGBM": {
            "classifier__n_estimators": [100, 150, 220, 300],
            "classifier__learning_rate": [0.01, 0.025, 0.04, 0.06, 0.08],
            "classifier__num_leaves": [7, 11, 15, 21, 31],
            "classifier__max_depth": [-1, 3, 4, 5, 6],
            "classifier__min_child_samples": [10, 15, 20, 25, 30],
            "classifier__colsample_bytree": [0.75, 0.85, 1.0],
            "classifier__reg_lambda": [0.1, 0.5, 1.0, 3.0, 5.0],
        },
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [5, 7, 9, 11, 13, 15, 17, 19, 21, 25],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "Extra Trees": {
            "classifier__n_estimators": [200, 300, 450, 600],
            "classifier__max_depth": [None, 4, 5, 6, 8, 10],
            "classifier__min_samples_split": [2, 5, 8, 10],
            "classifier__min_samples_leaf": [2, 3, 4, 5],
            "classifier__max_features": ["sqrt", 0.8, 1.0],
        },
    }

    models = {
        "LightGBM": lightgbm_pipeline,
        "K-Nearest Neighbors": knn_pipeline,
        "Extra Trees": extra_trees_pipeline,
        "Logistic Regression (Baseline)": baseline_pipeline,
    }
    return models, search_spaces


#############################################################################
#
#   Function name:      tune_and_train_models
#   Input:              df (DataFrame)
#   Description:        Performs stratified training/test splitting, tunes
#                       individual estimators on training folds, and builds
#                       soft-voting and stacking ensemble classifiers.
#   Output:             Trained models, test data, tuning scores and split data
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_resource(show_spinner=False)
def tune_and_train_models(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_models, search_spaces = build_model_pipelines()
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tuned_models = {}
    tuning_scores = {}

    # Hyperparameter search is restricted to the training partition.
    for model_name in ["LightGBM", "K-Nearest Neighbors", "Extra Trees"]:
        search = RandomizedSearchCV(
            estimator=base_models[model_name],
            param_distributions=search_spaces[model_name],
            n_iter=SEARCH_ITERATIONS,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        tuned_models[model_name] = search.best_estimator_
        tuning_scores[model_name] = float(search.best_score_)

    tuned_models["Logistic Regression (Baseline)"] = base_models[
        "Logistic Regression (Baseline)"
    ].fit(X_train, y_train)
    tuning_scores["Logistic Regression (Baseline)"] = np.nan

    voting = VotingClassifier(
        estimators=[
            ("lightgbm", tuned_models["LightGBM"]),
            ("knn", tuned_models["K-Nearest Neighbors"]),
            ("extra_trees", tuned_models["Extra Trees"]),
        ],
        voting="soft",
        weights=[1, 1, 1],
        n_jobs=-1,
    )
    voting.fit(X_train, y_train)
    tuned_models["Soft Voting Ensemble"] = voting

    stacking = StackingClassifier(
        estimators=[
            ("lightgbm", tuned_models["LightGBM"]),
            ("knn", tuned_models["K-Nearest Neighbors"]),
            ("extra_trees", tuned_models["Extra Trees"]),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )
    stacking.fit(X_train, y_train)
    tuned_models["Stacking Ensemble"] = stacking

    return {
        "models": tuned_models,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "train_count": len(X_train),
        "test_count": len(X_test),
        "tuning_scores": tuning_scores,
    }


#############################################################################
#
#   Function name:      evaluate_model
#   Input:              model, X_test, y_test, threshold (float)
#   Description:        Evaluates a classifier using the held-out test set and
#                       a user-selected probability threshold.
#   Output:             Metrics dictionary, predictions and class-1 scores
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def evaluate_model(model, X_test, y_test, threshold=0.50):
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, scores),
    }
    return metrics, predictions, scores


#############################################################################
#
#   Function name:      prepare_prediction_input
#   Input:              sample (DataFrame)
#   Description:        Applies the same zero-placeholder treatment used for
#                       the training data to user-entered prediction values.
#   Output:             Prediction-ready DataFrame in the expected column order
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def prepare_prediction_input(sample):
    prepared = sample[FEATURES].copy()
    for column in ZERO_AS_MISSING:
        prepared.loc[prepared[column] == 0, column] = np.nan
    return prepared


#############################################################################
#
#   Function name:      plot_outcome_distribution
#   Input:              df (DataFrame)
#   Description:        Plots the number of examples in each outcome class.
#   Output:             Matplotlib figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_outcome_distribution(df):
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Outcome 0", "Outcome 1"],
        counts.values,
        color=["#38bdf8", "#2dd4bf"],
        width=0.58,
    )
    ax.set_title("Dataset Outcome Distribution")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      plot_feature_insights
#   Input:              df (DataFrame)
#   Description:        Plots glucose and BMI distributions by dataset class.
#   Output:             Matplotlib figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_feature_insights(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(
        data=df,
        x="Glucose",
        hue=TARGET,
        bins=24,
        kde=True,
        element="step",
        ax=axes[0],
    )
    axes[0].set_title("Glucose distribution")
    sns.histplot(
        data=df,
        x="BMI",
        hue=TARGET,
        bins=24,
        kde=True,
        element="step",
        ax=axes[1],
    )
    axes[1].set_title("BMI distribution")
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      plot_confusion_matrix
#   Input:              y_test, predictions
#   Description:        Creates a heatmap for a model's confusion matrix.
#   Output:             Matplotlib figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_confusion_matrix(y_test, predictions):
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Outcome 0", "Outcome 1"],
        yticklabels=["Outcome 0", "Outcome 1"],
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      plot_roc_curves
#   Input:              y_test, models (dict), X_test
#   Description:        Draws ROC curves for the trained model collection.
#   Output:             Matplotlib figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_roc_curves(y_test, models, X_test):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      create_model_download
#   Input:              models (dict), threshold (float)
#   Description:        Serializes trained estimators and metadata into a
#                       downloadable Joblib model bundle.
#   Output:             Bytes suitable for Streamlit download_button
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def create_model_download(models, threshold):
    bundle = {
        "models": models,
        "features": FEATURES,
        "target": TARGET,
        "threshold": threshold,
        "note": "Educational model bundle; not for clinical use.",
    }
    output = io.BytesIO()
    joblib.dump(bundle, output)
    output.seek(0)
    return output.getvalue()


#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Runs the interactive dashboard, model comparison,
#                       ensemble prediction workflow and model download.
#   Output:             None
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.markdown(
        "<h1 class='hero-title'>🩺 Diabetes Detection & Ensemble Lab</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='hero-subtitle'>Explore health data, compare tuned machine-learning "
        "models, and try an educational prediction demo.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='notice-card'><b>Educational use only:</b> This is a portfolio "
        "demonstration using a small historical dataset. It is not a diagnostic tool, "
        "a clinically validated risk calculator, or a substitute for medical advice "
        "and laboratory testing.</div>",
        unsafe_allow_html=True,
    )

    if LGBMClassifier is None:
        st.error("LightGBM is required. Install it with: pip install lightgbm")
        st.stop()

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            "Could not find diabetes.csv. Place your downloaded diabetes.csv file "
            "in the same folder as app.py, then restart the Streamlit application."
        )
        st.stop()
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    st.sidebar.title("⚙️ Model Controls")
    st.sidebar.caption("Models are tuned on training folds; the held-out test set is kept separate.")

    with st.spinner(
        "Tuning LightGBM, KNN and Extra Trees, then building the ensemble models... "
        "The first run may take a little while."
    ):
        training = tune_and_train_models(df)

    models = training["models"]
    X_test = training["X_test"]
    y_test = training["y_test"]

    selected_model_name = st.sidebar.selectbox(
        "Choose a prediction model",
        options=list(models.keys()),
        index=list(models.keys()).index("Soft Voting Ensemble"),
    )
    threshold = st.sidebar.slider(
        "Classification threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help=(
            "Lower thresholds usually label more records as Outcome 1. "
            "This is an experimental control, not a medical cutoff."
        ),
    )
    st.sidebar.info(
        "The threshold changes the predicted class. ROC-AUC is calculated from the "
        "underlying model scores and does not change with this slider."
    )

    results = {
        name: evaluate_model(model, X_test, y_test, threshold)
        for name, model in models.items()
    }

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Dataset Explorer",
            "🔎 Feature Insights",
            "🏆 Ensemble Model Lab",
            "🧪 Prediction Studio",
        ]
    )

    #########################################################################
    #   TAB 1: Dataset Explorer
    #########################################################################
    with tab1:
        st.subheader("Get to know the dataset")
        total = len(df)
        positive = int((df[TARGET] == 1).sum())
        negative = int((df[TARGET] == 0).sum())

        card1, card2, card3, card4 = st.columns(4)
        card1.metric("Records", f"{total:,}")
        card2.metric("Input features", len(FEATURES))
        card3.metric("Outcome 0", f"{negative:,}")
        card4.metric("Outcome 1", f"{positive:,}")

        left, right = st.columns([1.15, 1])
        with left:
            st.write("**Dataset preview**")
            st.dataframe(df.head(12), use_container_width=True)
        with right:
            st.write("**Outcome distribution**")
            figure = plot_outcome_distribution(df)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

        missing = df[FEATURES].isna().sum().rename("Missing values").to_frame()
        st.write("**Missing-value placeholders after cleaning**")
        st.dataframe(missing, use_container_width=True)
        with st.expander("View descriptive statistics"):
            st.dataframe(df.describe().T, use_container_width=True)

    #########################################################################
    #   TAB 2: Feature Insights
    #########################################################################
    with tab2:
        st.subheader("Explore patterns in the measurements")
        st.caption(
            "These charts summarize the dataset. They do not define medical thresholds "
            "or establish cause and effect."
        )
        figure = plot_feature_insights(df)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

        correlation = df[FEATURES + [TARGET]].corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="crest",
            center=0,
            ax=ax_corr,
        )
        ax_corr.set_title("Feature Correlation Overview")
        fig_corr.tight_layout()
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)

    #########################################################################
    #   TAB 3: Ensemble Model Lab
    #########################################################################
    with tab3:
        st.subheader("Compare individual learners and ensemble models")
        st.write(
            "LightGBM, KNN and Extra Trees are tuned using randomized search with "
            "stratified five-fold cross-validation on the training partition. "
            "The voting ensemble combines their scores; the stacking ensemble learns "
            "a second-level combination."
        )
        st.caption(
            f"Training records: {training['train_count']} | Held-out test records: "
            f"{training['test_count']} | Threshold: {threshold:.2f}"
        )

        table_rows = []
        for name, (metrics, _, _) in results.items():
            row = {"Model": name, **metrics}
            row["Tuning CV ROC-AUC"] = training["tuning_scores"].get(name, np.nan)
            table_rows.append(row)
        comparison = pd.DataFrame(table_rows).set_index("Model")
        st.dataframe(
            comparison.style.format("{:.3f}", na_rep="—"),
            use_container_width=True,
        )

        selected_metrics, predictions, _ = results[selected_model_name]
        st.markdown(f"### Selected model: {selected_model_name}")
        st.caption(MODEL_DESCRIPTIONS[selected_model_name])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{selected_metrics['Accuracy'] * 100:.1f}%")
        m2.metric("Balanced accuracy", f"{selected_metrics['Balanced Accuracy'] * 100:.1f}%")
        m3.metric("Outcome 1 recall", f"{selected_metrics['Recall'] * 100:.1f}%")
        m4.metric("ROC-AUC", f"{selected_metrics['ROC AUC']:.3f}")

        left, right = st.columns(2)
        with left:
            figure = plot_confusion_matrix(y_test, predictions)
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        with right:
            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Outcome 0", "Outcome 1"],
                output_dict=True,
                zero_division=0,
            )
            st.write("**Classification report**")
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.write("**ROC curve comparison**")
        roc_figure = plot_roc_curves(y_test, models, X_test)
        st.pyplot(roc_figure, use_container_width=True)
        plt.close(roc_figure)

        st.download_button(
            label="⬇️ Download trained model bundle",
            data=create_model_download(models, threshold),
            file_name="diabetes_ensemble_models.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
        st.warning(
            "The held-out test metrics are estimates from this dataset split. "
            "They are not clinical validation results. Hyperparameter search scores "
            "are training cross-validation scores and should not be presented as "
            "independent test performance."
        )

    #########################################################################
    #   TAB 4: Prediction Studio
    #########################################################################
    with tab4:
        st.subheader("Try the educational prediction demo")
        st.write(
            f"Currently selected: **{selected_model_name}**. Provide sample measurements "
            "and the model will return its predicted dataset class."
        )
        st.info(
            "Tip: If a measurement is unavailable, enter 0 for Glucose, Blood Pressure, "
            "Skin Thickness, Insulin or BMI. The preprocessing pipeline treats those "
            "dataset placeholders as missing and imputes them."
        )

        with st.form("diabetes_prediction_form"):
            left, right = st.columns(2)
            with left:
                pregnancies = st.number_input(
                    "Pregnancies", min_value=0, max_value=20, value=1, step=1,
                    help="Number of times pregnant in the dataset."
                )
                glucose = st.number_input(
                    "Glucose", min_value=0.0, max_value=250.0, value=120.0, step=1.0,
                    help="Plasma glucose measurement. Enter 0 only when the value is unavailable."
                )
                blood_pressure = st.number_input(
                    "Blood pressure", min_value=0.0, max_value=150.0, value=70.0, step=1.0,
                    help="Diastolic blood pressure measurement."
                )
                skin_thickness = st.number_input(
                    "Skin thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                    help="Triceps skin-fold thickness."
                )
            with right:
                insulin = st.number_input(
                    "Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0,
                    help="Two-hour serum insulin measurement."
                )
                bmi = st.number_input(
                    "BMI", min_value=0.0, max_value=80.0, value=32.0, step=0.1,
                    help="Body mass index."
                )
                pedigree = st.number_input(
                    "Diabetes pedigree function", min_value=0.0, max_value=3.0,
                    value=0.50, step=0.01,
                    help="Dataset-specific diabetes pedigree function value."
                )
                age = st.number_input(
                    "Age", min_value=21, max_value=100, value=33, step=1,
                    help="Age in years. The source dataset includes adult women aged 21 and over."
                )

            submitted = st.form_submit_button(
                "🧬 Analyze sample",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            sample = pd.DataFrame(
                [{
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": pedigree,
                    "Age": age,
                }],
                columns=FEATURES,
            )
            chosen_model = models[selected_model_name]
            prediction_input = prepare_prediction_input(sample)
            class_score = float(chosen_model.predict_proba(prediction_input)[0, 1])
            predicted_class = int(class_score >= threshold)

            st.divider()
            st.markdown("### Model output")
            a, b = st.columns(2)
            a.metric("Predicted dataset class", f"Outcome {predicted_class}")
            b.metric("Model score for Outcome 1", f"{class_score * 100:.1f}%")
            st.progress(min(max(class_score, 0.0), 1.0))

            if predicted_class == 1:
                st.warning(
                    "The model assigned this sample to Outcome 1 in the training dataset. "
                    "This is not a diagnosis."
                )
            else:
                st.success(
                    "The model assigned this sample to Outcome 0 in the training dataset. "
                    "This does not rule out diabetes."
                )
            st.caption(
                "The displayed score is not necessarily a calibrated probability or an "
                "individual clinical risk estimate. Do not use it to make healthcare decisions."
            )

        st.markdown(
            "<div class='notice-card'><b>Medical disclaimer:</b> This application is "
            "for education and portfolio demonstration only. It has not been clinically "
            "validated and must not be used to diagnose, screen, treat or rule out diabetes. "
            "Consult a qualified healthcare professional for medical concerns.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "Built with Python, Streamlit, LightGBM and Scikit-Learn | "
        f"Educational project by {AUTHOR}"
    )


#############################################################################
#
#   Starter of main
#
#############################################################################
if __name__ == "__main__":
    main()
