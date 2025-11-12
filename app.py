# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import os
import glob
import warnings

warnings.filterwarnings("ignore")

# ==================== Page Configuration ====================
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%; background: #0099cc; color: white; border-radius: 8px; padding: 0.5rem;}
    .stDownloadButton>button {width: 100%;}
    .stSelectbox, .stNumberInput {margin-bottom: 1rem;}
    .stPlotlyChart {margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# --- TensorFlow CPU only ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], "GPU")

# --- Paths ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
MODEL_NAME = "concrete_strength_model.keras"
MODEL_PATH = MODEL_NAME

# --- Clean temp files ---
for f in glob.glob("concrete_strength_model.*"):
    if f != MODEL_NAME:
        try: os.remove(f)
        except: pass

# ==================== Column Renaming ====================
def rename_columns(df, has_strength=False):
    base = [
        "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", "Superplasticizer",
        "Coarse_Aggregate", "Fine_Aggregate", "Age", "Cement_per_Water",
        "Cement_Impurity_Factor", "Cement_Moisture_Factor"
    ]
    if has_strength:
        df.columns = base + ["Strength"]
    else:
        df.columns = base
    return df

# ==================== Imputation ====================
def impute_missing_values(df):
    df = df.copy()
    feats = df.drop(columns=["Cement_Moisture_Factor"], errors="ignore")
    knn = KNNImputer(n_neighbors=5)
    imputed = knn.fit_transform(feats)
    df_imp = pd.DataFrame(imputed, columns=feats.columns, index=df.index)
    df_imp["Cement_Moisture_Factor"] = df.get("Cement_Moisture_Factor", np.nan)
    knn2 = KNNImputer(n_neighbors=5)
    final = knn2.fit_transform(df_imp)
    return pd.DataFrame(final, columns=df_imp.columns, index=df.index)

# ==================== Model ====================
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ==================== Load Data ====================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        train = rename_columns(train, has_strength=True)
        test = rename_columns(test, has_strength=False)
        return train, test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

train_df, test_df = load_data()
st.success("Data loaded successfully.")

# ==================== Preprocessing ====================
def preprocess(df, is_train=True, scaler=None):
    df = impute_missing_values(df)
    features = [
        "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", "Superplasticizer",
        "Coarse_Aggregate", "Fine_Aggregate", "Age", "Cement_per_Water",
        "Cement_Impurity_Factor", "Cement_Moisture_Factor"
    ]
    X = df[features]
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = df["Strength"]
        return X_scaled, y, scaler, features
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, scaler, features

X_scaled, y, scaler, features = preprocess(train_df, is_train=True)

# ==================== Train / Load Model ====================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model..."):
        model = build_model(X_scaled.shape[1])
        X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)
        model.save(MODEL_PATH)
        st.success("Model trained and saved.")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded.")

# ==================== Prediction Helper ====================
def predict_strength_from_inputs(inputs, scaler, model, features):
    df_in = pd.DataFrame([inputs])
    scaled = scaler.transform(df_in[features])
    return model.predict(scaled, verbose=0)[0][0]

# ==================== SMART INVERSE SEARCH (در داده‌های واقعی) ====================
def find_optimal_mix(target_strength, scaler, model, features):
    # پیش‌بینی Strength برای همه داده‌های آموزشی
    X_scaled = scaler.transform(train_df[features])
    train_pred = model.predict(X_scaled, verbose=0).flatten()
    
    # فیلتر نزدیک‌ترین ترکیب‌ها (±10 MPa)
    mask = np.abs(train_pred - target_strength) <= 10
    if not np.any(mask):
        min_error = np.min(np.abs(train_pred - target_strength))
        st.warning(f"No mix within 10 MPa. Using closest: ±{min_error:.1f} MPa.")
        mask = np.ones(len(train_pred), dtype=bool)
    
    candidates = train_df[mask].copy()
    candidates['Predicted_Strength'] = train_pred[mask]
    candidates['Error'] = np.abs(candidates['Predicted_Strength'] - target_strength)
    
    # بهترین ترکیب (کمترین خطا)
    best_row = candidates.loc[candidates['Error'].idxmin()]
    
    optimal = best_row[features].to_dict()
    predicted = best_row['Predicted_Strength']
    
    return optimal, predicted

# ==================== Tabs ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Exploration", 
    "Upload Custom CSV", 
    "Manual Prediction", 
    "Design Mix for Target"
])

# ==================== Tab 1: Data Exploration ====================
with tab1:
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Feature", options=features, index=0)
    with col2:
        y_col = st.selectbox("Target", options=["Strength"], index=0)

    fig_scatter = px.scatter(
        train_df, x=x_col, y=y_col, color="Age",
        trendline="ols", trendline_color_override="red",
        title=f"{x_col} vs {y_col}",
        labels={x_col: x_col.replace("_", " "), y_col: "Strength (MPa)"},
        hover_data=["Cement_per_Water"]
    )
    results = px.get_trendline_results(fig_scatter).px_fit_results.iloc[0]
    slope, intercept, r2 = results.params[1], results.params[0], results.rsquared
    equation = f"y = {slope:.3f}x + {intercept:.2f} | R² = {r2:.3f}"
    fig_scatter.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=equation, showarrow=False,
        font=dict(size=14, color="red"), bgcolor="rgba(255,255,255,0.9)", bordercolor="red", borderwidth=1
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(px.histogram(train_df, x=x_col, nbins=30, title=f"Distribution of {x_col}"), use_container_width=True)
    with col4:
        st.plotly_chart(px.box(train_df, y="Strength", x="Age", title="Strength by Age", color="Age"), use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = train_df[features + ["Strength"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="equal", color_continuous_scale="RdBu", width=900, height=750)
    fig_corr.update_layout(font=dict(size=12), title_x=0.5)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== Tab 2: Upload CSV ====================
with tab2:
    st.header("Upload Your Own CSV")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            has_strength = "Strength" in user_df.columns
            user_df = rename_columns(user_df, has_strength=has_strength)
            st.dataframe(user_df.head())

            scaled, _, _ = preprocess(user_df, is_train=False, scaler=scaler)
            preds = model.predict(scaled, verbose=0).flatten()
            user_df["Predicted_Strength"] = preds
            st.success("Predictions completed!")
            st.dataframe(user_df.head(10))

            csv = user_df.to_csv(index=False).encode()
            st.download_button("Download predictions", data=csv, file_name="predictions_custom.csv", mime="text/csv")

            st.plotly_chart(px.scatter(user_df, x="Cement", y="Predicted_Strength", color="Age", size="Water", title="Your Predictions"), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== Tab 3: Manual Prediction ====================
with tab3:
    st.header("Manual Prediction")
    with st.form("manual_form"):
        cols = st.columns(3)
        inputs = {}
        for i, f in enumerate(features):
            with cols[i % 3]:
                inputs[f] = st.number_input(f.replace("_", " ").title(), value=0.0, format="%.4f", key=f)
        if st.form_submit_button("Predict"):
            pred = predict_strength_from_inputs(inputs, scaler, model, features)
            st.success(f"Predicted strength: **{pred:.2f} MPa**")
            st.balloons()

# ==================== Tab 4: Design Mix for Target ====================
with tab4:
    st.header("Design Mix for Target Strength")
    st.markdown("Enter your **desired strength** → get a **real, tested mix** from the dataset.")

    target = st.number_input("Target Strength (MPa)", min_value=15.0, max_value=75.0, value=40.0, step=1.0)

    if st.button("Find Best Real Mix"):
        with st.spinner("Searching in real data..."):
            optimal_mix, achieved = find_optimal_mix(target, scaler, model, features)
            error = abs(achieved - target)

            st.success(f"**Best real mix found!**")
            st.markdown(f"**Target**: `{target:.1f} MPa` → **Achieved**: `{achieved:.2f} MPa` (Δ `{error:.2f} MPa`)")

            mix_df = pd.DataFrame([optimal_mix]).T.round(2)
            mix_df.columns = ["Value"]
            mix_df.index.name = "Component"
            st.table(mix_df)

            st.info("This mix **exists in the original dataset** and has been experimentally tested.")
