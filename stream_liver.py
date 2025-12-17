import pickle
import numpy as np
import streamlit as st

# ===============================
# Load scaler & model
# ===============================
scaler = pickle.load(open('scaler.sav', 'rb'))
liver_model = pickle.load(open('model_prediksi.sav', 'rb'))

# ===============================
# Judul website
# ===============================
st.title('Prediksi Penyakit Liver')

st.write("Masukkan data pasien sesuai dengan nilai medis")

# ===============================
# Membagi kolom input
# ===============================
col1, col2 = st.columns(2)

with col1:
    age = st.text_input('Input nilai Age')
with col2:
    gender = st.text_input('Input nilai Gender (Female = 1, Male = 0)')

with col1:
    tot_bilirubin = st.text_input('Input nilai Total Bilirubin')
with col2:
    direct_bilirubin = st.text_input('Input nilai Direct Bilirubin')

with col1:
    alkphos = st.text_input('Input nilai Alkaline Phosphotase')
with col2:
    sgpt = st.text_input('Input nilai SGPT')

with col1:
    sgot = st.text_input('Input nilai SGOT')
with col2:
    tot_proteins = st.text_input('Input nilai Total Proteins')

with col1:
    albumin = st.text_input('Input nilai Albumin')
with col2:
    ag_ratio = st.text_input('Input nilai A/G Ratio')

# ===============================
# Tombol prediksi
# ===============================
if st.button('Test Prediksi Liver'):
    try:
        # Konversi input ke float
        input_data = np.array([
            float(age),
            float(gender),
            float(tot_bilirubin),
            float(direct_bilirubin),
            float(alkphos),
            float(sgpt),
            float(sgot),
            float(tot_proteins),
            float(albumin),
            float(ag_ratio)
        ]).reshape(1, -1)

        # Scaling data
        input_data_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = liver_model.predict(input_data_scaled)

        # Output
        if prediction[0] == 1:
            st.error('Pasien Terkena Penyakit Liver')
        else:
            st.success('Pasien Tidak Terkena Penyakit Liver')

    except ValueError:
        st.warning('⚠️ Pastikan semua input diisi dengan angka yang valid')
