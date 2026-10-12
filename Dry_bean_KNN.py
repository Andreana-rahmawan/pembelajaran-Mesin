import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


try:
    from IPython.display import display
except ImportError:
    def display(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string())
        else:
            print(obj)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42



# 1. MEMUAT DAN MEMAHAMI DATA


print("\n" + "=" * 70)
print("1. MEMUAT DAN MEMAHAMI DATA")
print("=" * 70)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "Dry_Bean_Dataset.xlsx")

if not os.path.exists(FILE):
    raise FileNotFoundError(
        f"File dataset tidak ditemukan:\n{FILE}\n"
        "Pastikan Dry_Bean_Dataset.xlsx berada satu folder "
        "dengan file Python."
    )

df_raw = pd.read_excel(
    FILE,
    sheet_name="Dry_Beans_Dataset"
)

print("Ukuran dataset awal (baris, kolom):", df_raw.shape)
print("\nLima data pertama:")
display(df_raw.head())

if "Class" not in df_raw.columns:
    raise KeyError(
        "Kolom target 'Class' tidak ditemukan. "
        "Periksa nama kolom pada file Excel."
    )

fitur_awal = [kolom for kolom in df_raw.columns if kolom != "Class"]

print("\nJumlah calon fitur:", len(fitur_awal))
print("Syarat minimal fitur: 10")
print("Kolom ID/nama yang dikeluarkan: tidak ada")
print("Kolom target:", "Class")
print("Daftar calon fitur:", fitur_awal)

if len(fitur_awal) < 10:
    raise ValueError(
        f"Dataset hanya memiliki {len(fitur_awal)} calon fitur. "
        "Tugas mensyaratkan minimal 10 fitur."
    )



# 2.  INFO DAN TIPE KOLOM


print("\n" + "=" * 70)
print("2. INFO DAN TIPE KOLOM")
print("=" * 70)

df = df_raw.copy()

print("\nInfo dataset sebelum koreksi:")
df.info()

# Koreksi nama kolom yang salah ketik pada dataset.
df = df.rename(columns={"AspectRation": "AspectRatio"})

#  perbarui daftar fitur setelah nama kolom dikoreksi.
# Jika tidak, daftar fitur masih menyimpan nama lama "AspectRation".
fitur_awal = [kolom for kolom in df.columns if kolom != "Class"]

# Konversi kolom fitur numerik bertipe integer menjadi float.
# Ini untuk menyeragamkan representasi tipe data, bukan syarat scaling.
kolom_integer = (
    df[fitur_awal]
    .select_dtypes(include=["int32", "int64"])
    .columns
    .tolist()
)

if kolom_integer:
    df[kolom_integer] = df[kolom_integer].astype("float64")

print("\nKolom integer yang dikonversi menjadi float:", kolom_integer)
print("\nTipe kolom setelah koreksi:")
print(df.dtypes)

# Perbarui daftar fitur setelah kemungkinan perubahan nama kolom.
fitur = [kolom for kolom in df.columns if kolom != "Class"]



# 3.  MISSING VALUES


print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

missing_per_kolom = df.isnull().sum()

print("Missing values per kolom:")
display(missing_per_kolom)

total_missing = int(missing_per_kolom.sum())
print("\nTotal missing values:", total_missing)

if total_missing == 0:
    print(
        "Kesimpulan: tidak ditemukan missing values, "
        "sehingga tidak diperlukan imputasi."
    )
else:
    print(
        "PERHATIAN: ditemukan missing values. "
        "Periksa kolom yang terdampak sebelum melanjutkan analisis."
    )

# Pemeriksaan nilai tidak wajar pada fitur numerik.
kolom_numerik = df[fitur].select_dtypes(include="number").columns

jumlah_nonpositif = int((df[kolom_numerik] <= 0).sum().sum())
jumlah_inf = int(np.isinf(df[kolom_numerik].to_numpy()).sum())

print("\nJumlah nilai <= 0 pada fitur numerik:", jumlah_nonpositif)
print("Jumlah nilai tak hingga (inf):", jumlah_inf)

if jumlah_inf > 0:
    raise ValueError(
        "Ditemukan nilai tak hingga (inf) pada fitur. "
        "Periksa dan tangani nilai tersebut sebelum scaling."
    )



# 4.  DUPLIKASI


print("\n" + "=" * 70)
print("4. DUPLIKASI DATA")
print("=" * 70)

jumlah_duplikat = int(df.duplicated().sum())

print("Jumlah baris duplikat:", jumlah_duplikat)
print("Jumlah baris sebelum penghapusan:", len(df))

if jumlah_duplikat > 0:
    print("\nContoh baris duplikat:")
    display(
        df[df.duplicated(keep=False)]
        .sort_values(by=list(df.columns))
        .head(10)
    )

df = df.drop_duplicates().reset_index(drop=True)

print("\nJumlah baris setelah penghapusan:", len(df))
print("Duplikat tersisa:", int(df.duplicated().sum()))



# 5.  KATEGORI (EJAAN DAN KAPITALISASI)


print("\n" + "=" * 70)
print("5. PEMERIKSAAN KATEGORI")
print("=" * 70)

print("Nilai unik Class sebelum dibersihkan:")
print(sorted(df["Class"].dropna().astype(str).unique()))

# Membersihkan spasi di tepi dan menyeragamkan kapitalisasi.
df["Class"] = df["Class"].astype(str).str.strip().str.upper()

print("\nNilai unik Class setelah dibersihkan:")
print(sorted(df["Class"].unique()))
print("Jumlah kategori/kelas:", df["Class"].nunique())

print("\nDistribusi kategori:")
display(df["Class"].value_counts())



# 6.  ANALISIS OUTLIER DENGAN METODE IQR


print("\n" + "=" * 70)
print("6. ANALISIS OUTLIER DENGAN METODE IQR")
print("=" * 70)


def iqr_bounds(series):
    """Menghitung batas bawah dan batas atas berdasarkan metode IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr

    return batas_bawah, batas_atas


# Analisis outlier pada fitur Area.
batas_bawah_area, batas_atas_area = iqr_bounds(df["Area"])

mask_outlier_area = (
    (df["Area"] < batas_bawah_area) |
    (df["Area"] > batas_atas_area)
)

jumlah_outlier_area = int(mask_outlier_area.sum())
persentase_outlier_area = mask_outlier_area.mean() * 100

print(f"Batas bawah Area: {batas_bawah_area:,.2f}")
print(f"Batas atas Area : {batas_atas_area:,.2f}")
print(
    f"Jumlah outlier Area: {jumlah_outlier_area} "
    f"({persentase_outlier_area:.2f}%)"
)

print("\nJumlah outlier Area berdasarkan kelas:")
display(df.loc[mask_outlier_area, "Class"].value_counts())

# Visualisasi boxplot Area keseluruhan dan per kelas.
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(
    x=df["Area"],
    ax=ax[0],
    color="#8ecae6"
)
ax[0].set_title("Boxplot Area (Keseluruhan)")
ax[0].set_xlabel("Area")

sns.boxplot(
    data=df,
    x="Class",
    y="Area",
    ax=ax[1],
    hue="Class",
    palette="Set2",
    legend=False
)
ax[1].set_title("Boxplot Area Berdasarkan Kelas")
ax[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Ringkasan jumlah outlier IQR untuk seluruh fitur numerik.
ringkasan_outlier = []

for kolom in kolom_numerik:
    batas_bawah, batas_atas = iqr_bounds(df[kolom])

    jumlah = int(
        (
            (df[kolom] < batas_bawah) |
            (df[kolom] > batas_atas)
        ).sum()
    )

    ringkasan_outlier.append((kolom, jumlah))

tabel_outlier = pd.DataFrame(
    ringkasan_outlier,
    columns=["Fitur", "Jumlah outlier IQR"]
).sort_values(
    by="Jumlah outlier IQR",
    ascending=False
)

print("\nRingkasan outlier seluruh fitur:")
display(tabel_outlier)

print(
    "\nCatatan: outlier tidak otomatis dihapus. "
    "Nilai ekstrem perlu ditinjau berdasarkan karakteristik data."
)



# 7.  ANALISIS KETIDAKSEIMBANGAN KELAS


print("\n" + "=" * 70)
print("7. ANALISIS KETIDAKSEIMBANGAN KELAS")
print("=" * 70)

distribusi_kelas = df["Class"].value_counts()
persentase_kelas = (
    distribusi_kelas / distribusi_kelas.sum() * 100
).round(2)

tabel_distribusi = pd.DataFrame({
    "Jumlah": distribusi_kelas,
    "Persentase (%)": persentase_kelas
})

display(tabel_distribusi)

rasio_kelas = distribusi_kelas.max() / distribusi_kelas.min()

print(
    f"\nRasio kelas terbesar : terkecil = "
    f"{distribusi_kelas.max()} : {distribusi_kelas.min()} "
    f"= {rasio_kelas:.2f} : 1"
)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=distribusi_kelas.index,
    y=distribusi_kelas.values,
    hue=distribusi_kelas.index,
    palette="Set2",
    legend=False
)
plt.title("Distribusi Kelas Setelah Penghapusan Duplikasi")
plt.xlabel("Kelas")
plt.ylabel("Jumlah data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 8.  PEMERIKSAAN INDIKASI DATA LEAKAGE


print("\n" + "=" * 70)
print("8. PEMERIKSAAN INDIKASI DATA LEAKAGE")
print("=" * 70)

# LabelEncoder digunakan untuk mengubah label target menjadi angka.
# Fitur input tetap berupa fitur numerik asli.
encoder_awal = LabelEncoder()
y_diagnostik = encoder_awal.fit_transform(df["Class"])

# Pemeriksaan sederhana: evaluasi setiap fitur secara individual
# menggunakan Decision Tree dangkal dan Stratified 5-Fold CV.
cv_diagnostik = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

hasil_leakage = []

for kolom in fitur:
    model_pohon = DecisionTreeClassifier(
        max_depth=4,
        random_state=RANDOM_STATE
    )

    skor = cross_val_score(
        model_pohon,
        df[[kolom]],
        y_diagnostik,
        cv=cv_diagnostik,
        scoring="accuracy"
    )

    hasil_leakage.append((
        kolom,
        skor.mean(),
        skor.std()
    ))

tabel_leakage = pd.DataFrame(
    hasil_leakage,
    columns=[
        "Fitur",
        "Rata-rata akurasi CV",
        "Standar deviasi"
    ]
).sort_values(
    by="Rata-rata akurasi CV",
    ascending=False
)

print("Hasil pengujian satu fitur secara individual:")
display(tabel_leakage)

akurasi_tertinggi_satu_fitur = tabel_leakage.iloc[0][
    "Rata-rata akurasi CV"
]

print(
    "\nAkurasi tertinggi satu fitur:",
    f"{akurasi_tertinggi_satu_fitur:.4f}"
)

print(
    "Interpretasi: hasil ini hanya pemeriksaan awal terhadap "
    "kemampuan prediksi setiap fitur secara individual. "
    "Hasil ini tidak dapat membuktikan bahwa seluruh kemungkinan "
    "data leakage tidak terjadi."
)

# Pemeriksaan korelasi antar fitur.
matriks_korelasi = df[fitur].corr(numeric_only=True)

plt.figure(figsize=(12, 9))
sns.heatmap(
    matriks_korelasi,
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Matriks Korelasi Antar Fitur")
plt.tight_layout()
plt.show()

pasangan_korelasi_tinggi = [
    (
        fitur_a,
        fitur_b,
        round(matriks_korelasi.loc[fitur_a, fitur_b], 3)
    )
    for i, fitur_a in enumerate(fitur)
    for fitur_b in fitur[i + 1:]
    if abs(matriks_korelasi.loc[fitur_a, fitur_b]) > 0.95
]

print("Pasangan fitur dengan |korelasi| > 0.95:")

if pasangan_korelasi_tinggi:
    tabel_korelasi_tinggi = pd.DataFrame(
        pasangan_korelasi_tinggi,
        columns=["Fitur A", "Fitur B", "Korelasi"]
    ).sort_values(
        by="Korelasi",
        ascending=False
    )

    display(tabel_korelasi_tinggi)
else:
    print("Tidak ditemukan pasangan fitur dengan korelasi di atas 0.95.")

print(
    "\nCatatan: korelasi tinggi antar fitur menunjukkan hubungan "
    "atau kemungkinan redundansi fitur, bukan otomatis data leakage."
)



# 9. PEMBAGIAN DATA LATIH DAN DATA UJI


print("\n" + "=" * 70)
print("9. PEMBAGIAN DATA LATIH DAN DATA UJI")
print("=" * 70)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Class"])

X = df[fitur]

print(
    "Pemetaan label:",
    dict(
        zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_)
        )
    )
)

# Stratify menjaga proporsi kelas pada data latih dan data uji.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nJumlah data latih:", X_train.shape)
print("Jumlah data uji  :", X_test.shape)

# Memastikan indeks baris tidak tumpang tindih antara train dan test.
irisan_indeks = set(X_train.index).intersection(set(X_test.index))
print("Jumlah indeks yang tumpang tindih:", len(irisan_indeks))

# Pemeriksaan yang lebih bermakna: apakah ada baris dengan nilai fitur
# identik di data latih dan data uji (duplikat lintas split)?
duplikat_lintas_split = len(
    pd.merge(
        X_train.drop_duplicates(),
        X_test.drop_duplicates(),
        how="inner"
    )
)
print(
    "Jumlah baris dengan fitur identik antara data latih dan uji:",
    duplikat_lintas_split
)



# 10.  NORMALISASI DAN STANDARISASI


print("\n" + "=" * 70)
print("10. NORMALISASI DAN STANDARISASI")
print("=" * 70)

kolom_contoh = [
    kolom for kolom in
    ["Area", "Perimeter", "Solidity", "ShapeFactor1"]
    if kolom in X_train.columns
]

print("\nStatistik sebelum scaling (data latih):")
display(
    X_train[kolom_contoh]
    .describe()
    .loc[["min", "max", "mean", "std"]]
    .round(4)
)

# Scaler di-fit hanya pada data latih.
scaler_minmax_deskriptif = MinMaxScaler()
scaler_standard_deskriptif = StandardScaler()

X_train_minmax = pd.DataFrame(
    scaler_minmax_deskriptif.fit_transform(X_train),
    columns=fitur,
    index=X_train.index
)

X_train_standard = pd.DataFrame(
    scaler_standard_deskriptif.fit_transform(X_train),
    columns=fitur,
    index=X_train.index
)

print("\nStatistik setelah normalisasi Min-Max (data latih):")
display(
    X_train_minmax[kolom_contoh]
    .describe()
    .loc[["min", "max", "mean", "std"]]
    .round(4)
)

print("\nStatistik setelah standarisasi StandardScaler (data latih):")
display(
    X_train_standard[kolom_contoh]
    .describe()
    .loc[["min", "max", "mean", "std"]]
    .round(4)
)

print(
    "\nCatatan: scaler deskriptif di atas hanya di-fit pada data latih. "
    "Untuk pemilihan model, scaling akan diuji kembali di dalam Pipeline "
    "menggunakan cross-validation."
)



# 11. PEMILIHAN SCALING DAN NILAI K DENGAN CROSS-VALIDATION


print("\n" + "=" * 70)
print("11. PEMILIHAN SCALING DAN NILAI K DENGAN CROSS-VALIDATION")
print("=" * 70)

# Semua kombinasi diuji hanya pada data latih.
# Pipeline mencegah scaler mempelajari informasi dari validation fold.
pilihan_scaling = {
    "Tanpa scaling": "passthrough",
    "Min-Max": MinMaxScaler(),
    "StandardScaler": StandardScaler()
}

pilihan_k = [3, 5, 7, 9, 11, 13, 15]

cv_model = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

hasil_perbandingan_model = []

for nama_scaling, scaler in pilihan_scaling.items():
    for nilai_k in pilihan_k:
        pipeline_model = Pipeline([
            ("scaler", scaler),
            ("knn", KNeighborsClassifier(
                n_neighbors=nilai_k,
                weights="uniform",
                metric="minkowski",
                p=2,
                n_jobs=-1
            ))
        ])

        skor_cv = cross_val_score(
            pipeline_model,
            X_train,
            y_train,
            cv=cv_model,
            scoring="accuracy",
            n_jobs=-1
        )

        hasil_perbandingan_model.append({
            "Scaling": nama_scaling,
            "Nilai K": nilai_k,
            "Akurasi CV": skor_cv.mean(),
            "Std CV": skor_cv.std()
        })

tabel_model = pd.DataFrame(hasil_perbandingan_model)
tabel_model = tabel_model.sort_values(
    by="Akurasi CV",
    ascending=False
).reset_index(drop=True)

print("Perbandingan kombinasi scaling dan nilai K:")
display(tabel_model.round(4))

print("\nAkurasi CV terbaik untuk setiap perlakuan scaling:")
display(
    tabel_model
    .loc[tabel_model.groupby("Scaling")["Akurasi CV"].idxmax()]
    .sort_values(by="Akurasi CV", ascending=False)
    .reset_index(drop=True)
    .round(4)
)

print("\nTabel ringkas akurasi CV (baris = nilai K, kolom = scaling):")
display(
    tabel_model
    .pivot(index="Nilai K", columns="Scaling", values="Akurasi CV")
    .round(4)
)

# Model dipilih berdasarkan rata-rata akurasi CV data latih.
kombinasi_terbaik = tabel_model.iloc[0]
scaling_terbaik = kombinasi_terbaik["Scaling"]
k_terbaik = int(kombinasi_terbaik["Nilai K"])

print("\nKombinasi terbaik berdasarkan cross-validation:")
print("Scaling terbaik:", scaling_terbaik)
print("Nilai K terbaik:", k_terbaik)
print(f"Akurasi CV      : {kombinasi_terbaik['Akurasi CV']:.4f}")
print(f"Standar deviasi : {kombinasi_terbaik['Std CV']:.4f}")



# 12. MELATIH MODEL KNN FINAL DAN EVALUASI DATA UJI


print("\n" + "=" * 70)
print("12. MODEL KNN FINAL DAN EVALUASI DATA UJI")
print("=" * 70)

if scaling_terbaik == "Tanpa scaling":
    scaler_final = "passthrough"
elif scaling_terbaik == "Min-Max":
    scaler_final = MinMaxScaler()
else:
    scaler_final = StandardScaler()

model_final = Pipeline([
    ("scaler", scaler_final),
    ("knn", KNeighborsClassifier(
        n_neighbors=k_terbaik,
        weights="uniform",
        metric="minkowski",
        p=2,
        n_jobs=-1
    ))
])

# Fit pada data latih; data uji hanya untuk evaluasi akhir.
model_final.fit(X_train, y_train)
y_pred = model_final.predict(X_test)

akurasi_test = accuracy_score(y_test, y_pred)

print("Algoritma       : K-Nearest Neighbors (KNN)")
print("Scaling terpilih:", scaling_terbaik)
print("Nilai K terpilih:", k_terbaik)
print("Metrik jarak    : Euclidean (Minkowski, p=2)")
print(f"\nAkurasi data uji: {akurasi_test:.4f} ({akurasi_test * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

print("\nTABEL CONFUSION MATRIX:")
print(cm_df)

fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
).plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

ax.set_title(
    f"Confusion Matrix - KNN (K={k_terbaik}, {scaling_terbaik})"
)
plt.tight_layout()
plt.show()



# 13. TAMPILAN DATA HASIL PEMROSESAN (5 TERATAS DAN 5 TERBAWAH)


print("\n" + "=" * 70)
print("13. DATA HASIL PEMROSESAN")
print("=" * 70)

# Untuk tabel representasi, scaler yang dipilih pada tahap cross-validation
# di-fit pada data latih, kemudian transform diterapkan pada seluruh baris
# hanya untuk keperluan penampilan. Data uji tidak ikut melakukan fit.
if scaling_terbaik == "Tanpa scaling":
    X_seluruh_standard = X.to_numpy()
else:
    scaler_tampilan = clone(model_final.named_steps["scaler"])
    scaler_tampilan.fit(X_train)
    X_seluruh_standard = scaler_tampilan.transform(X)

df_proses = pd.DataFrame(
    X_seluruh_standard,
    columns=fitur,
    index=df.index
)

df_proses["Class"] = df["Class"].values
df_proses["Class_encoded"] = y

print("Ukuran data hasil pemrosesan:", df_proses.shape)

print("\n=== 5 DATA TERATAS ===")
display(df_proses.head(5).round(4))

print("\n=== 5 DATA TERBAWAH ===")
display(df_proses.tail(5).round(4))






print("\n" + "=" * 70)
print("14. RINGKASAN HASIL")
print("=" * 70)

print("Dataset                 : Dry Bean Dataset")
print("Algoritma               : K-Nearest Neighbors (KNN)")
print("Jumlah data setelah     :", len(df), "baris")
print("Jumlah calon fitur      :", len(fitur))
print("Jumlah kelas            :", df["Class"].nunique())
print("Missing values          :", int(df.isnull().sum().sum()))
print("Duplikasi tersisa       :", int(df.duplicated().sum()))
print("Scaling terpilih        :", scaling_terbaik)
print("Nilai K terpilih        :", k_terbaik)
print(f"Akurasi cross-validation: {kombinasi_terbaik['Akurasi CV']:.4f}")
print(f"Akurasi data uji        : {akurasi_test:.4f}")

print(
    "\nCatatan kesimpulan: model KNN terbaik dipilih "
    "berdasarkan hasil Stratified 5-Fold Cross-Validation "
    "pada data latih. Data uji digunakan untuk evaluasi akhir."
)

print("\nProgram P2 KNN selesai dijalankan.")
