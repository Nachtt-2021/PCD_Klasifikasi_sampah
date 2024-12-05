import logging
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Contoh import pustaka
    import cv2
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    logger.info("Semua pustaka berhasil diimpor!")
    
    # Muat model Anda
    model = load_model("model_klasifikasi_sampah.h5")
    logger.info("Model berhasil dimuat!")

except ImportError as e:
    logger.error(f"ImportError: {e}")
    st.error("Terjadi masalah saat mengimpor pustaka. Periksa log untuk detailnya.")
except Exception as e:
    logger.error(f"Unhandled exception: {e}")
    st.error("Terjadi kesalahan. Periksa log untuk detailnya.")
