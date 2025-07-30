import streamlit as st
import numpy as np
import pandas as pd
import joblib

FEATURES = [
    ("Utility", "The utility company or organization responsible for managing the power grid and distributing electricity."),
    ("City/Town", "The location of the solar power project (city or town)."),
    ("County", "The administrative division within the state where the solar project is situated."),
    ("Zip", "The postal code or ZIP code corresponding to the project’s location."),
    ("Developer", "The entity or organization responsible for developing and implementing the solar power project."),
    ("Metering Method", "The method used to measure and record the solar energy production (e.g., net metering, gross metering)."),
    ("Estimated PV System Size (kWdc)", "The estimated size of the photovoltaic (PV) system in kilowatts direct current (kWdc). This represents the total capacity of the solar panels."),
    ("PV System Size (kWac)", "The actual size of the PV system in kilowatts alternating current (kWac). This accounts for system losses and efficiency.")
]

cat_cols = [f[0] for f in FEATURES[:6]]
num_cols = [f[0] for f in FEATURES[6:]]

model = joblib.load("xgboost_best_model.pkl")
scaler = joblib.load("scaler.pkl")
le_dict = joblib.load("label_encoders.pkl")

try:
    enc_map = joblib.load("target_encodings.pkl")
    use_target_encoding = True
except Exception:
    enc_map = None
    use_target_encoding = False

if use_target_encoding:
    final_features = [
        "Utility", "City/Town", "County", "Zip", "Developer", "Metering Method",
        "Estimated PV System Size (kWdc)", "PV System Size (kWac)",
        "Developer_encoded", "City/Town_encoded", "County_encoded", "Metering Method_encoded", "Utility_encoded"
    ]
else:
    final_features = cat_cols + num_cols

# --- Streamlit UI ---
st.set_page_config(page_title="Solar Energy Production Predictor", layout="centered")
st.title("Solar Energy Production Prediction App")
st.write("Enter the project details below to predict annual solar energy production. All fields are required.")

user_input = {}
for name, desc in FEATURES:
    if name in cat_cols:
        options = list(le_dict[name].classes_)
        user_input[name] = st.selectbox(f"{name}", options, help=desc)
    else:
        user_input[name] = st.number_input(f"{name}", value=0.0, help=desc)

input_df = pd.DataFrame([user_input])

# Label encoding
for col in cat_cols:
    le = le_dict[col]
    # Handle unseen categories: assign -1
    if input_df.at[0, col] in le.classes_:
        input_df[col] = le.transform(input_df[col].astype(str))
    else:
        input_df[col] = -1

if use_target_encoding:
    for col in ["Developer", "City/Town", "County", "Metering Method", "Utility"]:
        val = user_input[col]
        mean_map = enc_map.get(col, {})
        global_mean = np.mean(list(mean_map.values())) if mean_map else 0
        input_df[f"{col}_encoded"] = mean_map.get(val, global_mean)

for col in final_features:
    if col not in input_df.columns:
        input_df[col] = 0  # or np.nan

input_df = input_df[final_features]

if hasattr(scaler, 'feature_names_in_'):
    numeric_features = list(scaler.feature_names_in_)
else:
    numeric_features = num_cols  

for col in numeric_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df[numeric_features] = scaler.transform(input_df[numeric_features])

if st.button("Predict"):
    prediction = model.predict(input_df.values)
    st.success(f"Predicted Annual PV Energy Production: {prediction[0]:,.2f} kWh")

    st.markdown("---")
    st.subheader("Local Interpretability: SHAP Explanation for This Prediction")
    try:
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "SHAP Value": shap_values.values[0]
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.bar_chart(shap_df.set_index("Feature"))
        st.caption("This chart shows which features most influenced this specific prediction (positive or negative impact).")
    except ImportError:
        st.warning("SHAP is not installed. To see local interpretability, please install it with: pip install shap")
    except Exception as e:
        st.warning(f"Could not compute SHAP values: {e}")

st.subheader("Model Interpretability: Feature Importances")
importances = model.feature_importances_
if len(importances) == len(input_df.columns):
    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))
    st.caption("Higher importance means the feature has a greater impact on the model's predictions.")
else:
    st.warning(f"Feature importance visualization unavailable: Model expects {len(importances)} features, but found {len(input_df.columns)} in the current data processing. Please check model and preprocessing alignment.")

st.markdown("---")
st.info("This app uses a machine learning model trained on real solar project data. For best results, use values similar to those found in the dataset.")