## `README.md` — نسخه نهایی و کامل (با قابلیت معکوس)

```markdown
# Concrete Strength Predictor  
**A Streamlit-powered web app to predict concrete compressive strength using machine learning.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-orange)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-red)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This application uses a **deep neural network** to predict the **compressive strength (MPa)** of concrete based on its mixture composition and age. It supports:

- **Interactive data exploration** with regression lines and correlation matrix  
- **Upload your own CSV** for batch predictions  
- **Manual input** for real-time single-sample prediction  
- **Design mix for target strength** – Enter desired strength, get optimal mixture  
- **Downloadable results** in CSV format  

Perfect for civil engineers, researchers, and students working on concrete mix design.

---

## Features

| Feature | Description |
|--------|-------------|
| **Data Exploration** | Scatter plots with OLS regression, R², equation, histograms, box plots, and large correlation heatmap |
| **Custom CSV Upload** | Upload any CSV with the same 11/12 columns → get predictions instantly |
| **Manual Prediction** | Enter values manually and see real-time results |
| **Design Mix for Target** | Input desired strength (MPa) → get **optimal mix design** |
| **Download Results** | Export predictions as `predictions_custom.csv` |
| **No GPU Required** | Runs efficiently on CPU |

---

## Project Structure

```
siman/
├── app.py                     # Main Streamlit app
├── Data/
│   ├── train.csv              # Training data (12 columns)
│   └── test.csv               # Test data (11 columns)
├── concrete_strength_model.keras  # Trained model (auto-generated)
└── README.md
```

> **Column names** (after loading):  
> ```
> Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer,
> Coarse_Aggregate, Fine_Aggregate, Age, Cement_per_Water,
> Cement_Impurity_Factor, Cement_Moisture_Factor, [Strength]
> ```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/concrete-strength-predictor.git
cd concrete-strength-predictor
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn tensorflow-cpu plotly scipy
```

> Uses `tensorflow-cpu` and `scipy` for optimization.

---

## Run the App

```bash
streamlit run app.py
```

The app will launch at: [http://localhost:8501](http://localhost:8501)

---

## Usage

### 1. **Data Exploration**
- Select any feature and see its relationship with `Strength`
- View **regression line**, **equation**, and **R²**
- Explore distribution and correlation matrix

### 2. **Upload Custom CSV**
- Upload a CSV with the same structure as `train.csv` or `test.csv`
- Predictions appear instantly
- Download results with `Predicted_Strength` column

### 3. **Manual Prediction**
- Fill in the form with your mix design
- Click **Predict** → get strength in **MPa**
- Balloons celebrate your prediction!

### 4. **Design Mix for Target Strength** (New)
- Enter your **desired strength** (e.g., 50 MPa)
- Click **Find Optimal Mix**
- Get a **complete mix design** with:
  - Cement, Water, Age, etc.
  - Predicted strength
  - Error from target

---

## Model Details

- **Architecture**: 4-layer DNN (128 → 64 → 32 → 1)
- **Activation**: ReLU
- **Loss**: Mean Squared Error (MSE)
- **Training**: 100 epochs, Adam optimizer
- **Preprocessing**: KNN Imputation + Standard Scaling
- **Inverse Prediction**: `scipy.optimize.minimize` (L-BFGS-B) with realistic bounds

---

## Contributing

Contributions are welcome! Feel free to:
- Open issues
- Submit pull requests
- Improve UI/UX or model performance
- Add new optimization constraints (e.g., cost, CO₂)

---

## License

[MIT License](LICENSE) – Free to use, modify, and distribute.

---

## Author

**Iliya_farokhi**  
GitHub: [@1idjl](https://github.com/1idjl)  
Email: iliyaafarokhii@gmail.com

---

> **"Build smarter. Predict stronger. Design on demand."**
```
- **اسکرین‌شات از تب جدید**

بگو تا برات آماده کنم!
