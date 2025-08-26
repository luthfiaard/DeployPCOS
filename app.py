import streamlit as st
import pickle
import pandas as pd

# === Load model + fitur ===
with open("final_with_features.sav", "rb") as f:
    bundle = pickle.load(f)
selected_features = bundle["features"]

st.title("Prediksi PCOS dengan Random Forest")

st.write("Masukkan data berikut untuk prediksi:")

# === Form input untuk user ===
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# === Konversi ke DataFrame sesuai urutan fitur ===
input_df = pd.DataFrame([user_input], columns=selected_features)

# === Prediksi saat tombol ditekan ===
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"💡 Hasil Prediksi: **PCOS** dengan probabilitas {probability:.2%}")
    else:
        st.info(f"💡 Hasil Prediksi: **Tidak PCOS** dengan probabilitas {probability:.2%}")



