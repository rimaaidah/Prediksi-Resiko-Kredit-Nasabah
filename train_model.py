import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

print("=== MEMBACA DATASET ===")

# Coba baca dataset dengan berbagai opsi agar selalu berhasil
try:
    # file .xls yang disimpan sebagai csv sering butuh skiprows=1
    df = pd.read_csv("dataset/default_credit.csv", encoding='latin1', header=0)
    # Jika kolom pertama adalah angka (bukan teks), berarti tidak ada header asli
    if df.columns[0].isdigit() or df.columns[0] == '1':
        print("→ Dataset tampak tidak punya header, membaca ulang dengan header dari baris ke-1...")
        df = pd.read_csv("dataset/default_credit.csv", encoding='latin1', header=None, skiprows=1)
        # Tambahkan nama kolom sesuai dataset Kaggle asli
        df.columns = [
            'ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month'
        ]
except Exception as e:
    print("Gagal baca CSV:", e)
    df = pd.read_excel("dataset/default_credit.csv", header=1)

print("\n5 Baris Pertama Data:")
print(df.head())
print("\nJumlah Kolom:", len(df.columns))

# Gunakan nama kolom target asli
target_col = "default.payment.next.month"

# Pisahkan fitur dan target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Bagi data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluasi
y_pred = model.predict(X_test_scaled)
print("\n=== HASIL EVALUASI MODEL ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# Simpan model
joblib.dump(model, "model_credit.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model dan Scaler berhasil disimpan!")
