README - Food-101 Image Classification

Deskripsi:
-----------
Aplikasi klasifikasi gambar Food-101 menggunakan model TensorFlow (.h5).

Fitur:
------
1. Upload gambar tunggal atau batch
2. Prediksi kelas dari 101 jenis makanan
3. Visualisasi probabilitas prediksi
4. Download hasil prediksi CSV (untuk batch)

Persyaratan Sistem:
------------------
- Python 3.10 atau lebih tinggi
- Pip

Dependencies:
-------------
streamlit==1.26.0
tensorflow==2.16.0
numpy==1.26.1
pandas==2.1.0
plotly==5.17.0
Pillow==10.1.0

Instalasi dependencies:
----------------------
pip install -r requirements.txt

File penting:
-------------
1. `app.py`                  -> script utama Streamlit
2. `Food101_model.h5`        -> model Food-101
3. `food101_labels.txt`      -> daftar label Food-101
4. `requirements.txt`        -> daftar library + versi

Cara Menjalankan (Local):
-------------------------
1. Pastikan dependencies sudah terinstall.
2. Simpan semua file di satu folder.
3. Jalankan:

streamlit run app.py

4. Browser akan otomatis membuka dashboard Streamlit.

Catatan:
--------
- Jangan ubah nama file model atau label, path sudah diatur di `app.py`.
- Untuk batch gambar, pastikan semua gambar ada di folder atau CSV sesuai format.
- Visualisasi muncul di sidebar dan halaman utama.

Support:
--------
Jika ada error terkait package, pastikan versi sesuai requirements.txt.

