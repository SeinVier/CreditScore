import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Credit Score Assessor", page_icon="💳", layout="wide")
@st.cache_resource
def load_model():
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        st.error(f"File model tidak ditemukan di {model_path}. Silakan jalankan pipeline.py terlebih dahulu.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("System Prediction Credit Score")
st.markdown("---")

with st.form("credit_form"):
    st.subheader("Data Profil Finansial Nasabah")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Usia (Age)", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Pekerjaan (Occupation)", [
            "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
            "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", 
            "Accountant", "Mechanic", "Laborer"
        ])
        annual_income = st.number_input("Pendapatan Tahunan ($)", value=50000.0)
        monthly_inhand_salary = st.number_input("Gaji Bersih Bulanan ($)", value=4166.0)
        num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, value=2)
        
    with col2:
        num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, value=2)
        interest_rate = st.number_input("Suku Bunga Kartu Kredit (%)", min_value=0, value=15)
        num_of_loan = st.number_input("Jumlah Pinjaman Aktif", min_value=0, value=2)
        outstanding_debt = st.number_input("Total Sisa Hutang ($)", min_value=0.0, value=1000.0)
        total_emi_per_month = st.number_input("Cicilan per Bulan (EMI) ($)", min_value=0.0, value=150.0)
        
    with col3:
        delay_from_due_date = st.number_input("Rata-rata Telat Bayar (Hari)", min_value=0, value=5)
        num_of_delayed_payment = st.number_input("Frekuensi Telat Bayar", min_value=0, value=2)
        credit_utilization_ratio = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0.0, value=30.0)
        credit_mix = st.selectbox("Kualitas Kombinasi Kredit (Credit Mix)", ["Bad", "Standard", "Good"])
        payment_of_min_amount = st.selectbox("Hanya Bayar Tagihan Minimum?", ["Yes", "No", "NM"])
        payment_behaviour = st.selectbox("Perilaku Pembayaran", [
            "Low_spent_Small_value_payments", 
            "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments", 
            "High_spent_Large_value_payments"
        ])
    st.markdown("---")
    submitted = st.form_submit_button("Prediksi Score credit", use_container_width=True)

if submitted:
    input_dict = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Total_EMI_per_month": total_emi_per_month,
        "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Changed_Credit_Limit": None,
        "Unnamed: 0": None,
        "Type_of_Loan": None,
        "Credit_History_Age": None,
        "Monthly_Balance": None,
        "Amount_invested_monthly": None,
        "Num_Credit_Inquiries": None
    }

    input_data = pd.DataFrame([input_dict])
    
    with st.spinner("Menganalisis data nasabah..."):
        try:
            prediction = model.predict(input_data)[0]

            if prediction == "Good":
                st.success(f"### Hasil Prediksi: {prediction} Credit Score")
                st.write("Nasabah ini memiliki record finansial yang sangat baik, resiko kredit rendah.")
            elif prediction == "Standard":
                st.warning(f"### Hasil Prediksi: {prediction} Credit Score")
                st.write("Nasabah ini memiliki resiko menengah, perlu pengecekan lebih lanjut pada rasio hutang.")
            else:
                st.error(f"### Hasil Prediksi: {prediction} Credit Score")
                st.write("Nasabah ini berisiko tinggi, memiliki banyak record jejak pembayaran yang buruk.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")