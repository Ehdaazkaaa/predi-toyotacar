import streamlit as st
import numpy as np
import pickle
from PIL import Image
from ocr.ocr import ocr_plate_number  # Pastikan ocr.py ada di dalam folder `ocr/`

# ✅ HARUS paling atas
st.set_page_config(page_title="Prediksi Harga Mobil Toyota", page_icon="🚘", layout="centered")

# ⬇️ Styling UI
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, .main {
        background-color: #001F3F;
        color: #FFDC00;
        font-family: 'Poppins', sans-serif;
    }

    .stButton>button {
        background-color: #FFDC00;
        color: #001F3F;
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 20px;
    }

    .stNumberInput>div>input, .stSelectbox>div>div {
        background-color: #003366;
        color: #FFDC00;
        border-radius: 8px;
    }

    .css-1y4p8pa img {
        border-radius: 16px;
        box-shadow: 0 0 15px #FFDC00AA;
    }
    </style>
    """, unsafe_allow_html=True)

# ⬇️ Load model2
def load_models():
    try:
        with open("models/knn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("models/le_model.pkl", "rb") as f:
            le = pickle.load(f)
        return model, scaler, le
    except FileNotFoundError as e:
        st.error("❌ File model tidak ditemukan. Pastikan file .pkl tersedia dalam folder `models/`.")
        st.stop()

# ⬇️ Main app
def main():
    local_css()
    st.title("🚘 Prediksi Harga Mobil Toyota Bekas")

    st.header("📸 Ambil Gambar Mobil")
    car_image = st.camera_input("Ambil gambar mobil")

    st.header("🏷️ Ambil Gambar Plat Nomor")
    plate_image = st.camera_input("Ambil gambar plat nomor")

    plate_text = ""
    if plate_image:
        image = Image.open(plate_image)
        plate_text = ocr_plate_number(image)
        st.markdown(f"**Nomor Plat Terbaca:** `{plate_text}`")

    st.header("🔢 Input Spesifikasi Mobil")

    # ⬇️ Ambil label encoder untuk pilihan model
    model, scaler, le = load_models()
    model_options = le.classes_.tolist()

    with st.form("input_form"):
        model_input = st.selectbox("Model", model_options)
        year = st.number_input("Tahun", 1990, 2025, 2018)
        mileage = st.number_input("Mileage (km)", 0, 500000, 40000)
        tax = st.number_input("Tax (£)", 0, 500, 150)
        mpg = st.number_input("MPG", 0.0, 100.0, 50.0)
        engineSize = st.number_input("Engine Size (L)", 0.0, 10.0, 1.5)
        submit = st.form_submit_button("💸 Prediksi Harga")

    if submit:
        try:
            model_enc = le.transform([model_input])[0]
            X_input = np.array([[model_enc, year, mileage, tax, mpg, engineSize]])
            X_scaled = scaler.transform(X_input)
            pred_price = model.predict(X_scaled)[0]

            st.success(f"💰 Perkiraan Harga Mobil: **£{pred_price:,.2f}**")
            if plate_text:
                st.info(f"Nomor Plat: `{plate_text}`")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ✅ RUN APP
if __name__ == "__main__":
    main()
