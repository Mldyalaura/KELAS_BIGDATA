import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Hewan",
    page_icon="🐾",
    layout="centered"
)

# ============================================================
# JUDUL APLIKASI
# ============================================================
st.title("🐾 Klasifikasi Citra Hewan")
st.write("Aplikasi ini dapat mengenali tiga jenis hewan: **Ikan**, **Kelinci**, dan **Ayam**.")
st.write("Upload gambar hewan di bawah ini, lalu klik tombol **Prediksi**!")

st.divider()

# ============================================================
# LOAD MODEL (hanya sekali saat aplikasi pertama dijalankan)
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_klasifikasi_hewan_cnn_v2.keras")
    return model

model = load_model()

# Nama kelas hewan (urutan harus sama persis dengan saat training)
KELAS = ["ayam", "ikan", "kelinci"]
IMG_SIZE = 160

# ============================================================
# UPLOAD GAMBAR
# ============================================================
uploaded_file = st.file_uploader(
    "📁 Pilih gambar hewan (format JPG, JPEG, atau PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang kamu upload", use_column_width=True)

    st.divider()

    # Tombol prediksi
    if st.button("🔍 Prediksi Sekarang", use_container_width=True):

        with st.spinner("Sedang menganalisis gambar..."):

            # Preprocessing gambar agar sesuai input model
            img_resized = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized) / 255.0          # normalisasi 0-1
            img_array = np.expand_dims(img_array, axis=0)      # tambah dimensi batch

            # Prediksi
            prediksi = model.predict(img_array)
            indeks_kelas = np.argmax(prediksi[0])
            nama_kelas = KELAS[indeks_kelas]
            confidence = prediksi[0][indeks_kelas] * 100

        # ============================================================
        # TAMPILKAN HASIL
        # ============================================================
        st.success(f"✅ Prediksi selesai!")

        # Emoji per kelas
        emoji_hewan = {"ayam": "🐔", "ikan": "🐟", "kelinci": "🐰"}
        emoji = emoji_hewan.get(nama_kelas, "🐾")

        st.markdown(f"## {emoji} Hasil: **{nama_kelas.upper()}**")
        st.markdown(f"### Tingkat Keyakinan: **{confidence:.2f}%**")

        st.divider()

        # Tampilkan skor semua kelas
        st.write("**Skor untuk semua kelas:**")
        for i, kelas in enumerate(KELAS):
            skor = prediksi[0][i] * 100
            em = emoji_hewan.get(kelas, "🐾")
            st.progress(
                int(skor),
                text=f"{em} {kelas.capitalize()}: {skor:.2f}%"
            )

else:
    # Tampilan saat belum ada gambar diupload
    st.info("👆 Silakan upload gambar hewan terlebih dahulu.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Dibuat menggunakan CNN + Streamlit | Praktikum Big Data")
