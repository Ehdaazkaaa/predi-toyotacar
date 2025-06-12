import streamlit as st
import numpy as np
import pickle
from PIL import Image
from ocr import ocr_plate_number

# Load model & scaler
def load_models():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Styling CSS
def set_custom_style():
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            background-color: #001F3F;
            color: #FFDC00;
            font-family: 'Poppins', sans-serif;
        }

        .stTextInput, .stNumberInput, .stSelectbox, .stDateInput {
            background-color: #003366 !important;
            color: #FFDC00 !important;
        }

        .stButton>button {
            background-color: #FFDC00;
            color: #001F3F;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
        }

        h1, h2, h3, h4 {
            color: #FFDC00 !important;
        }

        .stCameraInput {
            border: 2px solid #FFDC00 !important;
            border-radius: 10px;
        }

        </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(page_title="Prediksi Mobil Toyota", page_icon="🚗", layout="centered")
    set_custom_style()
    st.title("🚗 Prediksi Harga Mobil Toyota Bekas")

    st.header("📷 Upload Gambar Mobil")
    car_image = st.camera_input("Ambil Gambar Mobil")

    st.header("🔍 Upload Plat Nomor")
    plate_image = st.camera_input("Ambil Gambar Plat")

    plate_text = ""
    if plate_image:
        img = Image.open(plate_image)
        plate_text = ocr_plate_number(img)
        st.markdown(f"**Nomor Plat Terbaca:** `{plate_text}`")

    st.header("🔧 Masukkan Detail Mobil")

    with st.form("input_form"):
        model_enc = st.number_input("Kode Model (contoh: 12)", 0, 50, 12)
        year = st.number_input("Tahun Mobil", 1990, 2025, 2018)
        mileage = st.number_input("Kilometer", 0, 300000, 40000)
        tax = st.number_input("Pajak (£)", 0, 500, 150)
        mpg = st.number_input("MPG", 0.0, 150.0, 50.0)
        engineSize = st.number_input("Ukuran Mesin (L)", 0.0, 10.0, 1.5)

        submit = st.form_submit_button("💰 Prediksi Harga")

    if submit:
        try:
            model, scaler = load_models()
            X_input = np.array([[model_enc, year, mileage, tax, mpg, engineSize]])
            X_scaled = scaler.transform(X_input)
            predicted_price = model.predict(X_scaled)[0]

            st.success(f"💸 Perkiraan Harga: £{predicted_price:,.2f}")
            if plate_text:
                st.info(f"Plat Nomor: {plate_text}")
        except Exception as e:
            st.error(f"❌ Error saat memprediksi: {e}")

if __name__ == "__main__":
    main()
