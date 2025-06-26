import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 0. Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Risiko Kesehatan Ibu",
    page_icon="🏥",
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="expanded"
)

# --- 1. Memuat Model, Scaler, dan LabelEncoder ---
# Hanya memuat model Decision Tree karena ini yang terbaik
@st.cache_resource # Cache resource untuk menghindari pemuatan ulang setiap kali ada interaksi
def load_assets():
    try:
        model = joblib.load('decision_tree_model.pkl')
        # st.success("Model Decision Tree berhasil dimuat.")
    except FileNotFoundError:
        st.error("File model 'decision_tree_model.pkl' tidak ditemukan. Pastikan sudah ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat model Decision Tree: {e}")
        st.stop()

    try:
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        # st.success("Scaler dan Label Encoder berhasil dimuat.")
    except FileNotFoundError:
        st.error("File 'scaler.pkl' atau 'label_encoder.pkl' tidak ditemukan. Pastikan sudah ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat scaler/label encoder: {e}")
        st.stop()

    return model, scaler, label_encoder

# Memuat aset
model, scaler, label_encoder = load_assets()
model_name = "Decision Tree" # Menetapkan nama model yang digunakan

# --- 2. Judul Aplikasi ---
st.title('🏥 Aplikasi Prediksi Risiko Kesehatan Ibu')
st.markdown(f'Aplikasi ini memprediksi tingkat risiko kesehatan ibu menggunakan model **{model_name}**.')

# --- 3. Sidebar Informasi ---
st.sidebar.header('Informasi')
st.sidebar.info(
    "**Cara Penggunaan:**\n"
    "1. Masukkan nilai untuk setiap fitur pasien di kolom 'Input Data Pasien'.\n"
    "2. Klik tombol 'Prediksi Risiko'.\n"
    "3. Lihat hasil prediksi tingkat risiko kesehatan ibu."
)
st.sidebar.write("---")
st.sidebar.markdown(f"""
    **Model yang Digunakan:** {model_name}
""")


# --- 4. Input Pengguna ---
st.header('Input Data Pasien')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Usia (Tahun)', min_value=10, max_value=60, value=25, help="Usia pasien dalam tahun.")
    systolic_bp = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=70, max_value=200, value=120, help="Tekanan darah sistolik pasien.")
    bs = st.number_input('Kadar Gula Darah (mmol/L atau mg/dL, sesuaikan unit data Anda)', min_value=1.0, max_value=20.0, value=6.0, format="%.2f", help="Kadar gula darah pasien.")

with col2:
    diastolic_bp = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=40, max_value=120, value=80, help="Tekanan darah diastolik pasien.")
    body_temp = st.number_input('Suhu Tubuh (°C)', min_value=35.0, max_value=42.0, value=37.0, format="%.1f", help="Suhu tubuh pasien.")
    heart_rate = st.number_input('Detak Jantung (bpm)', min_value=40, max_value=150, value=75, help="Detak jantung pasien per menit.")

# --- 5. Tombol Prediksi ---
if st.button('Prediksi Risiko', help="Klik untuk mendapatkan prediksi tingkat risiko."):
    # Validasi input
    if not all([age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]):
        st.warning("Mohon lengkapi semua input data.")
    else:
        # Buat DataFrame dari input pengguna
        input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                                  columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

        # --- Skalakan Input Pengguna dengan Scaler yang Sama! ---
        scaled_input_data = scaler.transform(input_data)

        st.subheader('Hasil Prediksi:')
        # --- Prediksi Menggunakan Model Decision Tree ---
        prediction_encoded = model.predict(scaled_input_data)
        prediction_proba = model.predict_proba(scaled_input_data) # Probabilitas untuk setiap kelas

        predicted_risk_level = label_encoder.inverse_transform(prediction_encoded)[0]

        if predicted_risk_level == 'High Risk':
            st.error(f"Tingkat Risiko: **{predicted_risk_level}** 🚨")
        elif predicted_risk_level == 'Mid Risk':
            st.warning(f"Tingkat Risiko: **{predicted_risk_level}** ⚠️")
        else:
            st.success(f"Tingkat Risiko: **{predicted_risk_level}** ✅")

        st.write("Probabilitas untuk setiap tingkat risiko:")
        proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
        st.dataframe(proba_df.T.rename(columns={0: 'Probabilitas'}))

        st.markdown("""
        ---
        **Catatan Penting:**
        * Aplikasi ini adalah demonstrasi dan tidak boleh digunakan sebagai pengganti diagnosis medis profesional.
        * Model ini dilatih pada data historis dan kinerjanya mungkin bervariasi pada data baru.
        """)