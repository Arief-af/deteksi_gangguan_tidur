import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB # Changed from KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import traceback

# --- Fungsi untuk membuat dan menyimpan plot metrik ringkasan ---
def generate_and_save_metrics_plot(y_true, y_pred, class_names, save_path='static/summary_metrics_bar_chart.png'):
    """
    Generates a bar chart of overall model metrics (Accuracy, Precision, Recall, F1-Score)
    and saves it as an image file.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names (e.g., ['No Disorder', 'Sleep Apnea', 'Insomnia']).
        save_path (str): The file path to save the plot image.
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        precision_macro = report['macro avg']['precision']
        recall_macro = report['macro avg']['recall']
        f1_macro = report['macro avg']['f1-score']

        metrics_names = ['Accuracy', 'Precision (Macro Avg)', 'Recall (Macro Avg)', 'F1-Score (Macro Avg)']
        metrics_values = [accuracy, precision_macro, recall_macro, f1_macro]

        plt.figure(figsize=(10, 7))
        bars = plt.bar(metrics_names, metrics_values, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
        plt.ylabel('Score')
        plt.title('Overall Model Performance Metrics (Macro Average)')
        plt.ylim(0, 1.1)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Summary metrics plot saved to {save_path}")
    except Exception as e:
        print(f"Error generating summary metrics plot: {e}")
        print(traceback.format_exc())

# --- Fungsi baru untuk membuat dan menyimpan plot full classification report (heatmap) ---
def generate_and_save_full_report_plot(y_true, y_pred, class_names, save_path='static/full_classification_report_heatmap.png'):
    """
    Generates a heatmap of the full classification report (precision, recall, f1-score per class)
    and saves it as an image file.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names.
        save_path (str): The file path to save the plot image.
    """
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        report_df = pd.DataFrame(report).transpose()
        ordered_rows = class_names + ['accuracy', 'macro avg', 'weighted avg']
        report_df = report_df.reindex(ordered_rows)

        metrics_to_plot_numeric = ['precision', 'recall', 'f1-score']
        metrics_to_plot_support = ['support']

        plot_data_numeric = report_df[metrics_to_plot_numeric]
        plot_data_support = report_df[metrics_to_plot_support]

        plt.figure(figsize=(12, len(class_names) + 3))
        
        annot_data = np.full(plot_data_numeric.shape, '', dtype=object)
        for i, row_idx in enumerate(plot_data_numeric.index):
            for j, col_idx in enumerate(plot_data_numeric.columns):
                annot_data[i, j] = f"{plot_data_numeric.iloc[i, j]:.2f}"
        
        annot_support_data = np.full(plot_data_support.shape, '', dtype=object)
        for i, row_idx in enumerate(plot_data_support.index):
            for j, col_idx in enumerate(plot_data_support.columns):
                annot_support_data[i, j] = f"{int(plot_data_support.iloc[i, j]):d}"

        combined_plot_data = pd.concat([plot_data_numeric, plot_data_support], axis=1)
        combined_annot_data = np.concatenate([annot_data, annot_support_data], axis=1)

        sns.heatmap(combined_plot_data, annot=combined_annot_data, cmap='Blues', fmt="", linewidths=.5, cbar=True, annot_kws={"size": 10})
        
        plt.title('Full Classification Report Heatmap', fontsize=16)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Full classification report heatmap saved to {save_path}")
    except Exception as e:
        print(f"Error generating full report heatmap: {e}")
        print(traceback.format_exc())

# --- Fungsi baru untuk membuat dan menyimpan plot feature importances ---
# This function is not applicable for Naive Bayes as it does not inherently provide feature importances.
def generate_and_save_feature_importance_plot(feature_importances, feature_names, save_path='static/feature_importances.png'):
    """
    Generates a bar plot of feature importances and saves it as an image file.

    Args:
        feature_importances (array-like): Feature importances.
        feature_names (list): List of feature names.
        save_path (str): The file path to save the plot image.
    """
    try:
        # For Naive Bayes, there are no direct feature importances like in tree-based models.
        # This function will not be called in the Naive Bayes version, but if it were,
        # you would need to adapt it or remove it.
        print("Feature importance plot generation skipped for Naive Bayes as it does not have inherent feature importances.")
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        print(traceback.format_exc())

# --- Akhir Fungsi-fungsi Plotting ---

# --- Konstanta Global ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
# ------------------------

# Langkah 1: Muat dataset
try:
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv', keep_default_na=False)
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'Sleep_health_and_lifestyle_dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

print("\nBeberapa baris pertama dataset setelah memuat (dengan keep_default_na=False):")
print(df.head())
print("\nInformasi dataset:")
print(df.info())
print("\nJumlah nilai yang hilang per kolom (sekarang NaN mungkin lebih sedikit jika 'None' terbaca sebagai string):")
print(df.isnull().sum())

# --- PENTING: Perbaikan untuk mengubah 'None' (sekarang sebagai string) menjadi 'No Sleep Disorder' ---
print("\nNilai unik 'Sleep Disorder' sebelum penggantian:")
print(df['Sleep Disorder'].unique())

print("\nMengganti label 'None' (dan variasi) menjadi 'No Sleep Disorder' di kolom 'Sleep Disorder'...")
df['Sleep Disorder'] = df['Sleep Disorder'].astype(str).str.strip().str.lower()
none_equivalents = ['none', '', 'n/a', 'not applicable']
for val in none_equivalents:
    df['Sleep Disorder'] = df['Sleep Disorder'].replace(val, 'no sleep disorder')

print("Penggantian label selesai.")
print("\nNilai unik 'Sleep Disorder' setelah penggantian 'None' dan sebelum dropna akhir:")
print(df['Sleep Disorder'].unique())

# **Perbaikan: Menangani nilai hilang di kolom 'Sleep Disorder' (jika masih ada NaN murni)**
print("\nMenangani nilai hilang (NaN murni) di kolom 'Sleep Disorder' dengan menghapus baris yang sesuai...")
df.dropna(subset=['Sleep Disorder'], inplace=True)
print("Baris dengan nilai hilang di 'Sleep Disorder' telah dihapus.")
print(f"Jumlah baris setelah penghapusan NaN murni: {len(df)}")


# Langkah 2: Pra-pemrosesan data lanjutan

# 2.1 Menangani kolom 'Blood Pressure'
print("\nMemisahkan kolom 'Blood Pressure' menjadi 'Systolic_BP' dan 'Diastolic_BP'...")
if df['Blood Pressure'].isnull().any():
    print("Peringatan: Ada nilai NaN di kolom 'Blood Pressure'. Menghapus baris tersebut.")
    df.dropna(subset=['Blood Pressure'], inplace=True)
    print(f"Jumlah baris setelah menghapus NaN di Blood Pressure: {len(df)}")

if 'Blood Pressure' in df.columns:
    try:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop('Blood Pressure', axis=1)
        print("Kolom 'Blood Pressure' berhasil diproses.")
    except Exception as e:
        print(f"Error memproses 'Blood Pressure' (format tidak valid): {e}")
        print("Mungkin ada nilai non-numeric di Blood Pressure yang tidak '/'. Menghapus kolom.")
        df = df.drop('Blood Pressure', axis=1)
else:
    print("Kolom 'Blood Pressure' tidak ditemukan atau sudah dihapus.")


# 2.2 Mengubah kolom kategorikal menjadi numerik menggunakan One-Hot Encoding
print("\nMelakukan One-Hot Encoding untuk kolom kategorikal...")
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
categorical_cols_present = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=categorical_cols_present, drop_first=True)
print("One-Hot Encoding selesai.")

# 2.3 Mengkodekan variabel target 'Sleep Disorder'
print("\nMengkodekan variabel target 'Sleep Disorder'...")
le = LabelEncoder()
df['Sleep Disorder'] = le.fit_transform(df['Sleep Disorder'])
print(f"Mapping 'Sleep Disorder': {list(le.classes_)} -> {list(range(len(le.classes_)))}")
print("Variabel target berhasil dikodekan.")

# Langkah 3: Mendefinisikan fitur (X) dan target (y)
cols_to_drop = [col for col in ['Person ID', 'Sleep Disorder'] if col in df.columns]
X = df.drop(cols_to_drop, axis=1)
y = df['Sleep Disorder']
print("\nDimensi fitur (X):", X.shape)
print("Dimensi target (y):", y.shape)

print("\n--- Kolom yang digunakan untuk melatih model (X.columns) ---")
print(X.columns.tolist())
print("----------------------------------------------------------")

# Langkah 4: Membagi data menjadi set pelatihan dan pengujian
print(f"\nMembagi data menjadi set pelatihan ({100*(1-TEST_SIZE)}%) dan pengujian ({100*TEST_SIZE}%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Ukuran set pelatihan X_train: {X_train.shape}")
print(f"Ukuran set pengujian X_test: {X_test.shape}")
print(f"Ukuran set pelatihan y_train: {y_train.shape}")
print(f"Ukuran set pengujian y_test: {y_test.shape}")

print(f"\nDistribusi kelas di y_train sebelum oversampling: {Counter(y_train)}")

# Langkah 5: Menskalakan fitur numerik
print("\nMengskalakan fitur numerik menggunakan StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Fitur berhasil diskalakan.")

# --- BAGIAN SMOTE: Oversampling dengan SMOTE ---
print("\nMelakukan oversampling kelas minoritas dengan SMOTE pada X_train_scaled...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Distribusi kelas di y_train setelah SMOTE: {Counter(y_train_resampled)}")
print("Oversampling selesai.")
# ----------------------------------------------


# Langkah 6: Melatih dan Men-tuning model Naive Bayes Classifier
print("\nMelakukan Hyperparameter Tuning dengan GridSearchCV untuk GaussianNB...")
# Definisikan parameter grid untuk Gaussian Naive Bayes
# GaussianNB memiliki sedikit parameter untuk disetel. var_smoothing adalah salah satunya.
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100) # Parameter smoothing untuk stabilitas numerik
}

grid_search = GridSearchCV(estimator=GaussianNB(), # Changed model here
                           param_grid=param_grid,
                           cv=CV_FOLDS,
                           n_jobs=-1,
                           verbose=1,
                           scoring='f1_macro')

grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"\nBest Hyperparameters found: {grid_search.best_params_}")
print(f"Best F1-Macro Score from Cross-Validation: {grid_search.best_score_:.4f}")

model = best_model
print("\nModel terbaik berhasil ditentukan dan dilatih.")


# Langkah 7: Membuat prediksi
print("\nMembuat prediksi pada set pengujian dengan model terbaik...")
y_pred = model.predict(X_test_scaled)
print("Prediksi selesai.")

# Langkah 8: Mengevaluasi model
print("\nMengevaluasi model terbaik pada data pengujian...")
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"\nAccuracy pada Test Set: {accuracy:.4f}")
print("\nClassification Report pada Test Set:\n", class_report)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

try:
    y_prob = model.predict_proba(X_test_scaled)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    print(f"ROC AUC Score (Weighted - One-vs-Rest): {roc_auc:.4f}")
except Exception as e:
    print(f"Peringatan: Tidak dapat menghitung ROC AUC Score. Pastikan model mendukung predict_proba dan target valid: {e}")


# --- Panggilan ke fungsi plot metrik ringkasan ---
generate_and_save_metrics_plot(y_test, y_pred, list(le.classes_), save_path='static/summary_metrics_bar_chart.png')
print("Plot metrik ringkasan berhasil dibuat dan disimpan di static/summary_metrics_bar_chart.png")

# --- Panggilan ke fungsi plot full classification report ---
generate_and_save_full_report_plot(y_test, y_pred, list(le.classes_), save_path='static/full_classification_report_heatmap.png')
print("Plot full classification report (heatmap) berhasil dibuat dan disimpan di static/full_classification_report_heatmap.png")

# Opsional: Menampilkan pentingnya fitur (Feature Importance) - REMOVED FOR NAIVE BAYES
# Naive Bayes does not have feature importances
# print("\nPentingnya Fitur (Feature Importance) dari model terbaik:")
# feature_importances = pd.Series(model.feature_importances_, index=X.columns)
# print(feature_importances.sort_values(ascending=False))

# --- Panggilan ke fungsi plot feature importances --- REMOVED FOR NAIVE BAYES
# generate_and_save_feature_importance_plot(model.feature_importances_, X.columns.tolist())
# print("Plot feature importances berhasil dibuat dan disimpan di static/feature_importances.png")

# --- Menyimpan model, scaler, label encoder, dan daftar kolom ---
print("\n--- Menyimpan model, scaler, label encoder, dan daftar kolom ---")
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/naive_bayes_model.pkl') # Changed filename
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'model/model_features.pkl')

print("Model, scaler, label encoder, dan daftar fitur berhasil disimpan di folder 'model/'.")