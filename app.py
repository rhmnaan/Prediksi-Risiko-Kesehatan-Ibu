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
# Pastikan nama file sesuai dengan yang Anda simpan!
@st.cache_resource # Cache resource untuk menghindari pemuatan ulang setiap kali ada interaksi
def load_assets():
    models = {}
    model_names = {
        'K-Nearest Neighbors': 'k-nearest_neighbors_model.pkl',
        'Gaussian Naive Bayes': 'gaussian_naive_bayes_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl'
    }

    for name, filename in model_names.items():
        try:
            models[name] = joblib.load(filename)
            # st.success(f"Model {name} berhasil dimuat.")
        except FileNotFoundError:
            st.error(f"File model '{filename}' untuk {name} tidak ditemukan. Pastikan sudah ada di folder yang sama.")
            st.stop()
        except Exception as e:
            st.error(f"Error memuat model {name} dari '{filename}': {e}")
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

    return models, scaler, label_encoder

models, scaler, label_encoder = load_assets()

# --- 2. Judul Aplikasi ---
st.title('🏥 Aplikasi Prediksi Risiko Kesehatan Ibu')
st.markdown('Aplikasi ini memprediksi tingkat risiko kesehatan ibu menggunakan beberapa model Machine Learning.')

# --- 3. Sidebar untuk Pemilihan Model dan Informasi ---
st.sidebar.header('Pengaturan Aplikasi')
selected_model_name = st.sidebar.selectbox(
    'Pilih Model untuk Prediksi Utama:',
    list(models.keys()) # Akan menampilkan 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Decision Tree'
)
st.sidebar.write("---")
st.sidebar.info(
    "**Cara Penggunaan:**\n"
    "1. Masukkan nilai untuk setiap fitur pasien di kolom 'Input Data Pasien'.\n"
    "2. Pilih model yang ingin Anda gunakan untuk prediksi utama.\n"
    "3. Klik tombol 'Prediksi Risiko'.\n"
    "4. Lihat hasil prediksi dari model yang dipilih dan perbandingan dengan model lain."
)

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

        st.subheader('Hasil Prediksi Individual Model:')
        # --- Prediksi Model yang Dipilih ---
        st.markdown(f"#### Hasil dari Model Terpilih: {selected_model_name}")
        selected_model = models[selected_model_name]
        prediction_encoded_selected = selected_model.predict(scaled_input_data)
        prediction_proba_selected = selected_model.predict_proba(scaled_input_data)

        predicted_risk_level_selected = label_encoder.inverse_transform(prediction_encoded_selected)[0]

        if predicted_risk_level_selected == 'High Risk':
            st.error(f"Tingkat Risiko: **{predicted_risk_level_selected}** 🚨")
        elif predicted_risk_level_selected == 'Mid Risk':
            st.warning(f"Tingkat Risiko: **{predicted_risk_level_selected}** ⚠️")
        else:
            st.success(f"Tingkat Risiko: **{predicted_risk_level_selected}** ✅")

        st.write("Probabilitas untuk setiap tingkat risiko:")
        proba_df_selected = pd.DataFrame(prediction_proba_selected, columns=label_encoder.classes_)
        st.dataframe(proba_df_selected.T.rename(columns={0: 'Probabilitas'}))

        st.markdown("---")

        # --- Prediksi dari Semua Model untuk Perbandingan ---
        st.subheader('Perbandingan Prediksi Antar Model:')

        comparison_results = []
        for name, model in models.items():
            prediction_encoded = model.predict(scaled_input_data)
            prediction_proba = model.predict_proba(scaled_input_data)
            predicted_risk = label_encoder.inverse_transform(prediction_encoded)[0]

            # Dapatkan probabilitas untuk kelas yang diprediksi
            predicted_class_index = prediction_encoded[0]
            confidence = prediction_proba[0][predicted_class_index]

            comparison_results.append({
                'Model': name,
                'Prediksi Risiko': predicted_risk,
                'Keyakinan (%)': f"{confidence * 100:.2f}%" # Probabilitas kelas prediksi
            })

        comparison_df = pd.DataFrame(comparison_results)
        st.dataframe(comparison_df.set_index('Model'))

        st.markdown("""
        ---
        **Catatan Penting:**
        * Aplikasi ini adalah demonstrasi dan tidak boleh digunakan sebagai pengganti diagnosis medis profesional.
        * Perbedaan prediksi antar model mungkin terjadi karena algoritma yang berbeda belajar pola data dengan cara yang unik.
        * "Keyakinan (%)" menunjukkan probabilitas model terhadap kelas yang diprediksi untuk input spesifik ini.
        * Untuk menentukan "model mana yang paling benar", Anda perlu merujuk pada metrik evaluasi model (akurasi, F1-score, dll.) yang didapat saat melatih dan menguji model di Google Colab. Model dengan kinerja evaluasi terbaik secara umum dianggap paling andal.
        """)