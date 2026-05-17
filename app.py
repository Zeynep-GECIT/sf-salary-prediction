import streamlit as st
import pickle
import numpy as np
import uuid
import pandas as pd
import os
from datetime import datetime

st.set_page_config(
    page_title = 'San Francisco Maaş Tahmini',
    layout = 'wide'
)
    
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f0eb;
        background-image: radial-gradient(#d4b8a0 1px, transparent 1px);
        background-size: 20px 20px;
    }
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        color: #4a3728;
    }
    h1, h2, h3 {
        color: #7d5a4f;
    }
    .stButton > button {
        background-color: #c4a882;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stTextInput > div > div > input {
        background-color: #fdf6f0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

st.title('San Francisco City Çalışan Maaş Tahmini')
st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.subheader('👤 Kişisel Bilgiler')
    name = st.text_input('Ad-Soyad')
    year = st.selectbox('Yıl', list(range(2013,2026)))
    year_type = st.selectbox('Yıl Tipi', ['Calender', 'Fiscal'])
    hours = st.number_input('Yıllık Çalışma Saati', min_value=0, max_value=100000, value=2000)

with col2:
    st.subheader('💼 İş Bilgileri')
    job_family = st.selectbox('İş Ailesi', list(mappings['job_family'].keys()))
    department = st.selectbox('Departman', list(mappings['department'].keys()))
    union = st.selectbox('Sendika', list(mappings['union'].keys()))
    org = st.selectbox('Organizasyon Grubu', list(mappings['org'].keys()))
    emp = st.selectbox('İstihdam Tipi', list(mappings['emp'].keys()))

st.subheader('💰 Mali Bilgiler')
col3, col4 = st.columns(2)
with col3:
    other_salaries = st.number_input('Diğer Maaşlar ($)', min_value=0.0, value=0.0)
with col4:
    health_and_dental = st.number_input('Sağlık ve Diş Sigortası ($)', min_value=0.0, value=0.0)

st.markdown('---')

if st.button('🔍 Maaş Tahmin Et', use_container_width=True):
    giris = np.array([[
        hours, other_salaries, health_and_dental, year,
        mappings['job_family'][job_family],
        mappings['department'][department],
        mappings['union'][union],
        mappings['org'][org],
        mappings['emp'][emp]
    ]])
    
    tahmin_log = model.predict(giris)[0]
    tahmin_gercek = np.expm1(tahmin_log)
    employee_id = str(uuid.uuid4())[:8].upper()

    st.markdown('---')
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric('💵Tahmini Maaş', f'${tahmin_gercek:,.2f}')
    with col6:
        st.metric('🪪 Çalışan ID', employee_id)
    with col7:
        st.metric('🏢 Departman', department)

    yeni_kayit = {
        'employee_id': employee_id,
        'name': name,
        'year': year,
        'year_type': year_type,
        'hours': hours,
        'other_salaries': other_salaries,
        'health_and_dental': health_and_dental,
        'job_family': job_family,
        'department': department,
        'union': union,
        'organization_group': org,
        'employment_type': emp,
        'predicted_salary': tahmin_gercek,
        'union_code': mappings['union_code'][union],
        'organization_group_code': mappings['org_code'][org],
        'job_family_code': mappings['job_family_code'][job_family],
        'data_loaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_as_of': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_salary': tahmin_gercek + other_salaries,
        'total_benefits': health_and_dental + mappings['avg_retirement'] + mappings['avg_other_benefits'],
        'total_compensation': tahmin_gercek + other_salaries + health_and_dental + mappings['avg_retirement'] + mappings['avg_other_benefits'],
        'department_code': mappings['department_code'][department],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    kayit_dosyasi = 'sf_yeni_calisanlar.csv'

    if os.path.exists(kayit_dosyasi):
        df_kayit = pd.read_csv(kayit_dosyasi)
        df_kayit = pd.concat([df_kayit, pd.DataFrame([yeni_kayit])], ignore_index=True)
    else:
        df_kayit = pd.DataFrame([yeni_kayit])

    df_kayit.to_csv(kayit_dosyasi, index=False)
    st.success('✅ Bilgiler Başarıyla Kaydedildi!')
