# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import traceback # Untuk debugging
from flask_cors import CORS # Import CORS

app = Flask(__name__)
CORS(app)


# Memuat model, scaler, label encoder, dan daftar fitur saat aplikasi dimulai
try:
    model = joblib.load('model/naive_bayes_model.pkl') # Changed model file path
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    model_features = joblib.load('model/model_features.pkl')
    print("Model, scaler, label encoder, dan daftar fitur berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan 'naive_bayes_model.pkl', 'scaler.pkl', 'label_encoder.pkl', dan 'model_features.pkl' ada di direktori 'model/'.")
    print("Silakan jalankan 'model_training.py' terlebih dahulu untuk membuat file-file ini.")
    exit()

@app.route('/')
def home():
    return "API Prediksi Gangguan Tidur. Gunakan endpoint /predict untuk prediksi."

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "Permintaan harus dalam format JSON."}), 400

    data = request.json

    # Pastikan semua kunci yang diharapkan ada dalam data yang diterima
    # The list `required_keys_for_input` is illustrative and isn't actively used for validation
    # in the current code, but the input_dict creation handles missing keys gracefully.
    # required_keys_for_input = [
    #     'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    #     'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
    #     'Gender_Male',
    #     'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Lawyer',
    #     'Occupation_Nurse', 'Occupation_Sales Representative',
    #     'Occupation_Salesperson', 'Occupation_Scientist', 'Occupation_Software Engineer',
    #     'Occupation_Teacher',
    #     'BMI Category_Normal Weight', 'BMI Category_Obese', 'BMI Category_Overweight'
    # ]

    # Buat DataFrame dari data input
    try:
        # Siapkan dictionary untuk DataFrame, pastikan semua kolom yang dibutuhkan model ada
        input_dict = {}
        for feature in model_features:
            # Mengisi nilai dari data JSON jika ada, jika tidak, default ke 0
            # Ini sangat penting untuk kolom hasil one-hot encoding yang mungkin tidak selalu ada
            # di setiap request jika kategorinya tidak aktif.
            input_dict[feature] = [data.get(feature, 0)] # Ambil nilai, default 0 jika tidak ada

        # Buat DataFrame dengan urutan kolom yang benar
        input_df = pd.DataFrame(input_dict, columns=model_features)

        # Skalakan data input menggunakan scaler yang sama
        input_scaled = scaler.transform(input_df)

        # Lakukan prediksi
        prediction_encoded = model.predict(input_scaled)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({"prediction": prediction_label})

    except KeyError as e:
        return jsonify({"error": f"Missing expected key in JSON data: {e}. Please provide all required features."}), 400
    except Exception as e:
        # Tangani error tak terduga
        print(f"Error during prediction: {e}")
        print(traceback.format_exc()) # Cetak stack trace untuk debugging
        return jsonify({"error": "Terjadi kesalahan internal saat memproses permintaan.", "details": str(e)}), 500

# Endpoint untuk metrik ringkasan (bar chart)
@app.route('/metrics/summary_plot', methods=['GET'])
def get_summary_metrics_plot():
    try:
        return send_from_directory('static', 'summary_metrics_bar_chart.png')
    except FileNotFoundError:
        return jsonify({"error": "Summary metrics plot image not found. Please ensure 'summary_metrics_bar_chart.png' exists in the 'static' directory."}), 404
    except Exception as e:
        print(f"Error serving summary metrics plot: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat mencoba menampilkan plot ringkasan.", "details": str(e)}), 500

# NEW ENDPOINT: Untuk melayani full classification report (heatmap)
@app.route('/metrics/full_report_plot', methods=['GET'])
def get_full_report_plot():
    try:
        return send_from_directory('static', 'full_classification_report_heatmap.png')
    except FileNotFoundError:
        return jsonify({"error": "Full classification report heatmap not found. Please ensure 'full_classification_report_heatmap.png' exists in the 'static' directory."}), 404
    except Exception as e:
        print(f"Error serving full report plot: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Terjadi kesalahan internal saat mencoba menampilkan plot laporan lengkap.", "details": str(e)}), 500

# Removed the feature importance endpoint as Naive Bayes does not provide feature importances.
# @app.route('/metrics/future_importances', methods=['GET'])
# def get_future_importances():
#     try:
#         return send_from_directory('static', 'feature_importances.png')
#     except FileNotFoundError:
#         return jsonify({"error": "Feature importance plot not found. Please ensure 'feature_importances.png' exists in the 'static' directory."}), 404
#     except Exception as e:
#         print(f"Error serving full report plot: {e}")
#         print(traceback.format_exc())
#         return jsonify({"error": "Terjadi kesalahan internal saat mencoba menampilkan plot laporan lengkap.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)