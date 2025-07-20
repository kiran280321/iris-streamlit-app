import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('iris_model.pkl')
iris = load_iris()

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title(" Iris Flower Classifier")
st.markdown("This app uses a trained **Random Forest Classifier** to predict the species of Iris flowers.")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(features)
    proba = model.predict_proba(features)
    st.success(f"Predicted Species: **{iris.target_names[prediction[0]].capitalize()}**")
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(proba, columns=iris.target_names)
    st.bar_chart(prob_df.T)

st.subheader(" Upload a CSV for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    expected_cols = iris.feature_names
    if all(col in batch_data.columns for col in expected_cols):
        batch_pred = model.predict(batch_data)
        batch_data['Predicted_Species'] = [iris.target_names[i] for i in batch_pred]
        st.write("✅ Predictions:")
        st.dataframe(batch_data)
        st.download_button("Download Predictions as CSV", data=batch_data.to_csv(index=False), file_name="iris_predictions.csv", mime="text/csv")
    else:
        st.error(f"CSV must contain columns: {expected_cols}")

st.subheader("Explore the Iris Dataset")
viz_type = st.selectbox("Select Visualization Type", ["Pairplot", "Correlation Matrix"])
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
if viz_type == "Pairplot":
    sns.pairplot(df, hue='species')
    st.pyplot()
elif viz_type == "Correlation Matrix":
    st.write("Correlation matrix of the features:")
    corr = df.iloc[:, :-1].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot()
