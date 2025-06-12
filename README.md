# 🚘 Prediksi Harga Mobil Toyota Bekas

Aplikasi ini memprediksi harga mobil Toyota bekas berdasarkan input data dan gambar (dari kamera).

## Cara Menjalankan

1. Simpan dataset `Toyota (1).csv` di folder project.
2. Install library:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan training:
   ```
   python train.py
   ```
4. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```

## Fitur

- Ambil gambar mobil dan plat nomor dari kamera langsung
- OCR untuk mengenali nomor plat
- UI elegan dengan warna navy-kuning dan font Poppins
- Prediksi harga dengan model KNN
