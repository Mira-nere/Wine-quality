# ✅ Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os

# ✅ Step 1: Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# ✅ Step 2: Clean data
df = df.drop_duplicates()
assert df.isnull().sum().sum() == 0, "There are missing values!"

# ✅ Step 3: Create quality label (Adjust for more balance)
df["quality_label"] = df["quality"].apply(lambda q: "low" if q <= 5 else "medium" if q == 6 else "high")

# ✅ Step 4: Normalize features (exclude label columns)
features_to_scale = df.drop(columns=["quality", "quality_label"])
scaler = MinMaxScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=features_to_scale.columns)

# ✅ Step 5: Recombine with target columns
scaled_df = pd.concat([scaled_features.reset_index(drop=True), df[["quality", "quality_label"]].reset_index(drop=True)], axis=1)

# ✅ Step 6: Prepare data for training
X = scaled_df.drop(columns=["quality", "quality_label"])
y = scaled_df["quality_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Step 7: Train or load model
MODEL_FILE = "wine_quality_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
else:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

# ✅ Step 8: Model Evaluation (Run once for debug)
if "model_evaluated" not in st.session_state:
    y_pred = model.predict(X_test)
    st.write("📋 Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "medium", "high"])
    disp.plot(cmap='Blues')
    st.pyplot(plt)
    st.session_state.model_evaluated = True

# ✅ Step 9: Streamlit App Interface
st.title("🍷 Wine Quality Prediction")
st.write("Enter wine's physicochemical properties to predict its quality level (Low, Medium, High).")

# User inputs
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
density = st.slider("Density", 0.9900, 1.0040, 0.9968)
pH = st.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

# Prediction
if st.button("Predict Quality"):
    user_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    classes = model.classes_

    st.success(f"✅ Predicted Wine Quality Category: **{prediction.upper()}**")

    st.write("🔍 Prediction Confidence:")
    prob_df = pd.DataFrame({
        "Quality": classes,
        "Probability": [f"{p*100:.2f}%" for p in probabilities]
    })
    st.table(prob_df)

    # Filter dataset by predicted label
    filtered_df = scaled_df[scaled_df["quality_label"] == prediction]

    st.markdown(f"### 📊 Characteristics of **{prediction.upper()}** Quality Wines")

    # Histogram
    st.subheader("🔍 Distribution of Features")
    features_only = filtered_df.drop(columns=["quality", "quality_label"])
    num_features = features_only.shape[1]

    fig, axes = plt.subplots(nrows=(num_features + 2) // 3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(features_only.columns):
        axes[i].hist(features_only[col], bins=15, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Feature Distributions for {prediction.upper()} Quality Wines", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    # Scatter plot: Alcohol vs Quality
    st.subheader("🍷 Alcohol vs. Wine Quality")
    fig, ax = plt.subplots()
    sns.scatterplot(data=scaled_df, x="alcohol", y="quality", hue="quality_label", palette="Set2", ax=ax)
    ax.axhline(y=filtered_df["quality"].mean(), color='red', linestyle='--', label=f'{prediction.upper()} Avg Quality')
    ax.legend()
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = filtered_df.drop(columns=["quality_label"]).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title(f"{prediction.upper()} Quality Wine Correlation Heatmap")
    st.pyplot(fig)
