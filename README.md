# 🌸 Iris Streamlit App

A simple and interactive web app built with Streamlit to classify Iris flower species using a trained Random Forest model. The app allows both manual input and batch prediction via CSV, along with data visualizations.

---

## 🚀 Features

- 🌼 Predict Iris flower species based on user input
- 📂 Upload a CSV file for batch predictions
- 📊 Visualize the Iris dataset with:
  - Pairplots
  - Correlation matrix
- ⬇️ Download results as CSV

---

## 📁 Project Structure

iris-streamlit-app/
│
├── app.py                # Streamlit app
├── train_model.py        # Trains and saves the model (iris_model.pkl)
├── iris_model.pkl        # Pre-trained Random Forest model
├── sample_input.csv      # Sample CSV for batch prediction
├── requirements.txt      # Python dependencies
└── README.md             # You're reading it!

---

## 🛠 Setup Instructions

### 1. Clone the repository

git clone https://github.com/kiran280321/iris-streamlit-app.git
cd iris-streamlit-app

### 2. Install dependencies

pip install -r requirements.txt

### 3. Train the model

python train_model.py

This creates iris_model.pkl which is used for predictions.

### 4. Run the app

streamlit run app.py

The app will open in your default web browser at http://localhost:8501.

---

## 📂 Sample CSV Format

To use batch prediction, your CSV should have the following columns:

sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
6.2,3.4,5.4,2.3
5.9,3.0,4.2,1.5

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to a public GitHub repository
2. Go to https://streamlit.io/cloud
3. Click "New App" and select your repo
4. Set app.py as the main file
5. Click Deploy

---

## 🧠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📃 License

This project is licensed under the MIT License. You are free to use, modify, and share it.

---

## 🙋‍♂️ Author

Made with ❤️ by KIRAN280321
