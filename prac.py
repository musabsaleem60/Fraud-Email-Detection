import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

st.title('📧 Fraud Email Detection with Machine Learning')

# Load data
@st.cache_data
def load_data():
    try:
        cust = pd.read_csv("https://raw.githubusercontent.com/musabsaleem60/Fraud-Email-Detection/main/cust.csv")
        trans = pd.read_csv("https://raw.githubusercontent.com/musabsaleem60/Fraud-Email-Detection/main/trans.csv")
        merged_df = pd.merge(cust, trans, on='customerEmail', how='inner')
        merged_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)
        return merged_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Preprocess data
def preprocess_data(df):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    Q1 = df[numerical].quantile(0.25)
    Q3 = df[numerical].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (df[numerical] < lower) | (df[numerical] > upper)
    df_cleaned = df[~outliers.any(axis=1)]
    df_cleaned['Transaction_Success_Rate'] = ((df_cleaned['No_Transactions'] / (df_cleaned['No_Transactions'] + df_cleaned['transactionFailed'])) * 100).round(2)
    df_cleaned['Transaction_TotalAmount'] = df_cleaned['No_Transactions'] * df_cleaned['transactionAmount']

    imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)
    df_cleaned['Fraud'] = df_cleaned['Fraud'].astype(int)
    return df_cleaned

# Encode data
def encode_data(data):
    features = ['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']
    X = data[features]
    y = data['Fraud']
    return X, y

# Train models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    svm_model = SVC(random_state=42, probability=True, class_weight='balanced')
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    return svm_model, rf_model, X_test, y_test, svm_pred, rf_pred

# Evaluate model
def evaluate_model(y_test, preds, model, X_test):
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "Confusion Matrix": confusion_matrix(y_test, preds)
    }

# App flow
df = load_data()

if not df.empty:
    st.success("Data loaded successfully!")
    data = preprocess_data(df)
    st.write("🔍 Fraud Class Distribution")
    st.bar_chart(data['Fraud'].value_counts())

    X, y = encode_data(data)
    svm_model, rf_model, X_test, y_test, svm_pred, rf_pred = train_models(X, y)

    svm_metrics = evaluate_model(y_test, svm_pred, svm_model, X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, rf_model, X_test)

    model_choice = st.selectbox("Choose a model to view results:", ["SVM", "Random Forest"])
    metrics = svm_metrics if model_choice == "SVM" else rf_metrics

    st.subheader(f"{model_choice} Performance Metrics")
    for key, value in metrics.items():
        if key != "Confusion Matrix":
            st.write(f"{key}: {value:.2f}")

    st.write("📊 Confusion Matrix")
    st.write(pd.DataFrame(metrics["Confusion Matrix"],
                          columns=["Predicted: Not Fraud", "Predicted: Fraud"],
                          index=["Actual: Not Fraud", "Actual: Fraud"]))

    st.subheader("🔎 Compare Predictions")
    comparison_df = pd.DataFrame({
        "Actual": y_test.values,
        "SVM": svm_pred,
        "Random Forest": rf_pred
    })
    st.dataframe(comparison_df.head(10))

    st.subheader("🔐 Predict Fraud for a Customer")
    email = st.text_input("Enter Customer Email:")
    if email:
        customer_row = data[data['customerEmail'] == email]
        if not customer_row.empty:
            st.write(customer_row)
            input_data = customer_row[['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']]
            prediction = svm_model.predict(input_data) if model_choice == "SVM" else rf_model.predict(input_data)
            st.success(f"Prediction: {'FRAUD' if prediction[0] == 1 else 'NOT FRAUD'}")
        else:
            st.error("No customer found with that email.")
else:
    st.error("Failed to load data.")
