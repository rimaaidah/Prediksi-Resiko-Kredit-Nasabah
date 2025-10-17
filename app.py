from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load model & scaler
model = joblib.load("model_credit.pkl")
scaler = joblib.load("scaler.pkl")

# Daftar fitur yang form kita kirim (user-friendly names)
FORM_FEATURES = [
    "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
]

# Nilai default kalau input tidak tersedia
DEFAULT_VALUE = 0.0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # kumpulkan nilai dari form untuk fitur yg kita definisikan
        values = []
        for f in FORM_FEATURES:
            v = request.form.get(f)
            if v is None or v == "":
                values.append(DEFAULT_VALUE)
            else:
                # convert ke float aman
                try:
                    values.append(float(v))
                except:
                    # jika user memilih label (misal "Laki-laki"), coba parse angka
                    # fallback ke default jika nggak bisa
                    try:
                        values.append(float(v.strip()))
                    except:
                        values.append(DEFAULT_VALUE)

        # sekarang values berisi len = len(FORM_FEATURES)
        arr = np.array(values).reshape(1, -1)  # bentuk (1, n)

        # cek berapa fitur yang scaler/model harapkan
        expected_n = getattr(scaler, "n_features_in_", None)
        if expected_n is None:
            # fallback: coba dari model jika ada attribute
            expected_n = getattr(model, "n_features_in_", arr.shape[1])

        # jika jumlah input < expected, tambahkan kolom default (0) di posisi paling kiri (mis: ID)
        if arr.shape[1] < expected_n:
            pad_count = expected_n - arr.shape[1]
            # buat padding kolom di depan dengan nilai default
            pad = np.full((1, pad_count), DEFAULT_VALUE)
            arr = np.hstack([pad, arr])
        elif arr.shape[1] > expected_n:
            # jika kebanyakan kolom (seharusnya jarang terjadi), pangkas kolom paling kiri
            arr = arr[:, -expected_n:]

        # sekarang ukurannya cocok dengan scaler
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        prob = model.predict_proba(arr_scaled)[0][1] if hasattr(model, "predict_proba") else None

        result = "💥 NASABAH BERISIKO GAGAL BAYAR" if pred == 1 else "✅ NASABAH AMAN / TIDAK BERISIKO"
        return render_template("result.html", result=result, prob=(round(prob,3) if prob is not None else "N/A"))
    except Exception as e:
        # tampilkan pesan error yang lebih ramah
        return f"Terjadi kesalahan pada proses prediksi: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
