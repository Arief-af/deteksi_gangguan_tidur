import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
        plt.ylim(0, 1.1) # Batas y ditingkatkan sedikit agar angka tidak terpotong

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

        # Convert report to a DataFrame for easier plotting, excluding 'accuracy', 'macro avg', 'weighted avg'
        # Ensure 'accuracy', 'macro avg', 'weighted avg' are included at the bottom
        report_df = pd.DataFrame(report).transpose()
        # Filter only the relevant rows (class names and averages)
        ordered_rows = class_names + ['accuracy', 'macro avg', 'weighted avg']
        report_df = report_df.reindex(ordered_rows)

        # Select only relevant columns for heatmap (precision, recall, f1-score, support)
        # Note: 'support' is integer, so fmt=".0f" or no fmt for it.
        # For precision, recall, f1-score, use ".2f"
        metrics_to_plot_numeric = ['precision', 'recall', 'f1-score']
        metrics_to_plot_support = ['support']

        plot_data_numeric = report_df[metrics_to_plot_numeric]
        plot_data_support = report_df[metrics_to_plot_support]

        plt.figure(figsize=(12, len(class_names) + 3)) # Adjust figure size based on number of classes
        
        # Create a custom annotation matrix for combined formatting
        annot_data = np.full(plot_data_numeric.shape, '', dtype=object)
        for i, row_idx in enumerate(plot_data_numeric.index):
            for j, col_idx in enumerate(plot_data_numeric.columns):
                annot_data[i, j] = f"{plot_data_numeric.iloc[i, j]:.2f}"
        
        # Add support column
        annot_support_data = np.full(plot_data_support.shape, '', dtype=object)
        for i, row_idx in enumerate(plot_data_support.index):
            for j, col_idx in enumerate(plot_data_support.columns):
                annot_support_data[i, j] = f"{int(plot_data_support.iloc[i, j]):d}" # Format as integer

        # Combine for display
        combined_plot_data = pd.concat([plot_data_numeric, plot_data_support], axis=1)
        combined_annot_data = np.concatenate([annot_data, annot_support_data], axis=1)

        sns.heatmap(combined_plot_data, annot=combined_annot_data, cmap='Blues', fmt="", linewidths=.5, cbar=True, annot_kws={"size": 10})
        
        plt.title('Full Classification Report Heatmap', fontsize=16)
        plt.yticks(rotation=0) # Ensure class names are horizontal
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Full classification report heatmap saved to {save_path}")
    except Exception as e:
        print(f"Error generating full report heatmap: {e}")
        print(traceback.format_exc())

# --- Fungsi baru untuk membuat dan menyimpan plot feature importances ---
def generate_and_save_feature_importance_plot(feature_importances, feature_names, save_path='static/feature_importances.png'):
    """
    Generates a bar plot of feature importances and saves it as an image file.

    Args:
        feature_importances (array-like): Feature importances.
        feature_names (list): List of feature names.
        save_path (str): The file path to save the plot image.
    """
    try:
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Feature importance plot saved to {save_path}")
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        print(traceback.format_exc())

# --- Akhir Fungsi-fungsi Plotting ---

# --- Konstanta Global ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3 # Jumlah lipatan untuk Cross-Validation dan GridSearchCV
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
# Daftar nilai yang dianggap setara dengan 'none' atau tidak ada gangguan
none_equivalents = ['none', '', 'n/a', 'not applicable'] # Tambahkan jika ada variasi lain
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

# Cek apakah 'Blood Pressure' masih ada setelah dropna
if 'Blood Pressure' in df.columns:
    try:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop('Blood Pressure', axis=1)
        print("Kolom 'Blood Pressure' berhasil diproses.")
    except Exception as e:
        print(f"Error memproses 'Blood Pressure' (format tidak valid): {e}")
        print("Mungkin ada nilai non-numeric di Blood Pressure yang tidak '/'. Menghapus kolom.")
        df = df.drop('Blood Pressure', axis=1) # Hapus kolom jika formatnya bermasalah
else:
    print("Kolom 'Blood Pressure' tidak ditemukan atau sudah dihapus.")


# 2.2 Mengubah kolom kategorikal menjadi numerik menggunakan One-Hot Encoding
print("\nMelakukan One-Hot Encoding untuk kolom kategorikal...")
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
# Filter kolom yang benar-benar ada di dataframe
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
# Pastikan 'Person ID' dan 'Sleep Disorder' ada sebelum mencoba drop
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


# Langkah 6: Melatih dan Men-tuning model Random Forest Classifier
print("\nMelakukan Hyperparameter Tuning dengan GridSearchCV untuk RandomForestClassifier...")
# Definisikan parameter grid
param_grid = {
    'n_estimators': [100, 200, 300], # Jumlah pohon
    'max_depth': [10, 20, None], # Kedalaman maksimum pohon
    'min_samples_split': [2, 5], # Minimal sampel untuk membelah node
    'min_samples_leaf': [1, 2], # Minimal sampel di daun
    'class_weight': ['balanced', 'balanced_subsample'] # Strategi bobot kelas
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE),
                            param_grid=param_grid,
                            cv=CV_FOLDS, # K-fold cross-validation
                            n_jobs=-1, # Gunakan semua core CPU
                            verbose=1, # Tampilkan progress
                            scoring='f1_macro') # Gunakan F1-macro untuk data imbalanced

grid_search.fit(X_train_resampled, y_train_resampled) # Latih GridSearchCV dengan data yang sudah di-resample

best_model = grid_search.best_estimator_ # Dapatkan model terbaik
print(f"\nBest Hyperparameters found: {grid_search.best_params_}")
print(f"Best F1-Macro Score from Cross-Validation: {grid_search.best_score_:.4f}")

# Gunakan model terbaik untuk evaluasi akhir
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

# Tambahan: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Tambahan: ROC AUC Score (untuk multi-class)
# Untuk ROC AUC di multi-class, kita perlu probabilitas
try:
    y_prob = model.predict_proba(X_test_scaled)
    # Gunakan 'ovr' (one-vs-rest) untuk multi-class AUC
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

# Opsional: Menampilkan pentingnya fitur (Feature Importance)
print("\nPentingnya Fitur (Feature Importance) dari model terbaik:")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))

# --- Panggilan ke fungsi plot feature importances ---
generate_and_save_feature_importance_plot(model.feature_importances_, X.columns.tolist())
print("Plot feature importances berhasil dibuat dan disimpan di static/feature_importances.png")

# --- Menyimpan model, scaler, label encoder, dan daftar kolom ---
print("\n--- Menyimpan model, scaler, label encoder, dan daftar kolom ---")
# Pastikan direktori 'model' ada
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/random_forest_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(X.columns.tolist(), 'model/model_features.pkl')

print("Model, scaler, label encoder, dan daftar fitur berhasil disimpan di folder 'model/'.")