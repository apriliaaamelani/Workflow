# 🩺 Prediksi Diabetes Menggunakan MLflow & CI/CD

## 📖 Deskripsi Proyek

Proyek ini merupakan implementasi **Machine Learning** untuk melakukan prediksi diabetes dengan menerapkan praktik **MLOps** menggunakan **MLflow**, **GitHub Actions**, dan **Docker**.

Proyek ini tidak hanya berfokus pada proses pelatihan model, tetapi juga membangun pipeline machine learning yang dapat direproduksi secara otomatis, mulai dari proses training, pelacakan eksperimen, penyimpanan model, hingga pembuatan Docker image secara otomatis melalui Continuous Integration (CI).

---

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk:

- Membangun model Machine Learning untuk prediksi diabetes.
- Mencatat seluruh eksperimen menggunakan MLflow.
- Mengotomatisasi proses training menggunakan MLflow Project.
- Menerapkan Continuous Integration (CI) menggunakan GitHub Actions.
- Membangun Docker image secara otomatis dari model yang telah dilatih.
- Mempersiapkan model agar lebih mudah untuk proses deployment.

---

## 📊 Dataset

Dataset yang digunakan merupakan dataset diabetes yang telah melalui tahap preprocessing dan disimpan dalam file:

```
diabetes_preprocessing.csv
```

Dataset ini berisi sejumlah fitur numerik yang digunakan untuk memprediksi apakah seseorang menderita diabetes atau tidak.

---

## 🔄 Alur Pipeline Machine Learning

```
Dataset Diabetes
        │
        ▼
Load Dataset
        │
        ▼
Train-Test Split
        │
        ▼
Pelatihan Model
(Logistic Regression)
        │
        ▼
Evaluasi Model
(Accuracy & Confusion Matrix)
        │
        ▼
MLflow Tracking
        │
        ▼
Penyimpanan Model
        │
        ▼
GitHub Actions CI
        │
        ▼
Build Docker Image
        │
        ▼
Docker Hub
```

---

## 🛠️ Teknologi yang Digunakan

### Machine Learning

- Python
- Scikit-learn
- Logistic Regression
- Pandas
- NumPy

### Experiment Tracking

- MLflow

### Visualisasi

- Matplotlib
- Seaborn

### MLOps

- GitHub Actions
- Docker
- Docker Hub
- Conda

---

## 📂 Struktur Repository

```
.
├── .github/
│   └── workflows/
│       └── ci-mlflow.yml
├── MLProject
├── modelling.py
├── conda.yaml
├── diabetes_preprocessing.csv
├── Dockerhub.txt
└── README.md
```

---

## ⚙️ Alur Training Model

Pipeline machine learning pada proyek ini meliputi beberapa tahapan:

1. Memuat dataset hasil preprocessing.
2. Membagi dataset menjadi data training dan testing.
3. Melatih model **Logistic Regression**.
4. Menghitung nilai **Accuracy**.
5. Membuat **Confusion Matrix**.
6. Menyimpan seluruh eksperimen menggunakan MLflow.
7. Menyimpan model hasil training dalam format MLflow Model.

---

## 📈 Pelacakan Eksperimen Menggunakan MLflow

MLflow digunakan untuk mencatat seluruh proses eksperimen secara otomatis, meliputi:

- Parameter model
- Metrics
- Model hasil training
- Confusion Matrix
- Artifact
- Informasi setiap proses training (Run)

Pelacakan dilakukan menggunakan fitur:

```python
mlflow.autolog()
```

sehingga seluruh informasi eksperimen dapat terdokumentasi secara otomatis.

---

## 🔄 Continuous Integration (CI)

Proyek ini menerapkan **GitHub Actions** untuk mengotomatisasi proses Machine Learning.

Setiap perubahan yang dikirim ke branch **main** akan menjalankan pipeline secara otomatis yang meliputi:

- Checkout repository
- Setup Python
- Install seluruh dependency
- Menjalankan MLflow Project
- Upload artifact MLflow
- Build Docker Image
- Push Docker Image ke Docker Hub

Dengan demikian proses training dapat dilakukan secara konsisten tanpa perlu menjalankannya secara manual.

---

## 🐳 Docker

Model yang telah selesai dilatih kemudian dibangun menjadi Docker Image menggunakan MLflow.

Proses ini memungkinkan model dijalankan pada berbagai environment dengan konfigurasi yang sama sehingga mempermudah proses deployment.

---

## 🚀 Cara Menjalankan Proyek

### Clone Repository

```bash
git clone https://github.com/username/diabetes-mlflow-project.git
```

Masuk ke folder project

```bash
cd diabetes-mlflow-project
```

Install dependency

```bash
pip install mlflow
```

Jalankan MLflow Project

```bash
mlflow run .
```

---

## 💡 Kemampuan yang Ditunjukkan

Melalui proyek ini saya mengimplementasikan beberapa kemampuan berikut:

- Machine Learning
- Logistic Regression
- Data Preprocessing
- Model Evaluation
- MLflow Tracking
- MLflow Project
- Experiment Management
- GitHub Actions
- Continuous Integration (CI)
- Docker
- Docker Hub
- Dasar MLOps

---

## 📌 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan pada proyek ini antara lain:

- Menambahkan algoritma Machine Learning lainnya.
- Melakukan perbandingan performa beberapa model.
- Hyperparameter Tuning.
- Membangun REST API untuk serving model.
- Continuous Deployment (CD).
- Monitoring performa model setelah deployment.

---

## 👩‍💻 Penulis

**Aprilia Melani**

Machine Learning Engineer | Data Analyst | AI Enthusiast
