import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import logging
import streamlit as st
import tensorflow as tf
import urllib.request

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    import cv2
    logger.info("OpenCV berhasil diimpor.")
except ImportError as e:
    logger.error(f"Error mengimpor OpenCV: {e}")
    st.error("Gagal mengimpor pustaka OpenCV. Periksa log untuk detail.")

# Load model yang telah dilatih
model = load_model('models/model_klasifikasi_sampah.h5')

# Daftar label kelas
classes = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']

# Fungsi untuk memproses gambar sebelum dimasukkan ke model
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))  # Resize sesuai input model
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Header aplikasi
st.title("Aplikasi Klasifikasi Sampah")
st.markdown("### Unggah foto sampah untuk memeriksa jenisnya")
st.text("Jenis sampah yang didukung: Kaca, Kardus, Kertas, Logam, Plastik, Residu")

# Widget untuk unggah gambar
uploaded_file = st.file_uploader("Unggah gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca file yang diunggah
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Tampilkan gambar yang diunggah
    st.image(image, channels="BGR", caption="Gambar yang diunggah", use_column_width=True)
    
    # Preprocessing gambar
    processed_image = preprocess_image(image)
    
    # Prediksi gambar menggunakan model
    predictions = model.predict(processed_image)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Tampilkan hasil prediksi
    st.markdown("### Hasil Prediksi")
    st.write(f"**Jenis Sampah:** {predicted_class}")
    st.write(f"**Tingkat Kepercayaan:** {confidence:.2f}%")

    # Tambahkan visualisasi distribusi probabilitas (opsional)
    st.markdown("### Distribusi Probabilitas Kelas")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
else:
    st.info("Silakan unggah gambar sampah terlebih dahulu.")

