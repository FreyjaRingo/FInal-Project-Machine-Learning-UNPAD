# Deteksi Ujaran Kebencian Multi-Label Bahasa Indonesia

Tugas besar pengganti UTS. Proyek ini mengklasifikasikan 12 label ujaran kebencian dan bahasa kasar pada dataset Twitter Ibrohim & Budi (2019). Empat keluarga algoritma dibandingkan (Bayesian, SVM, Ensemble Tree, ANN), masing-masing dengan strategi multi-label BR dan CC, tiga kondisi penanganan imbalance (none, class weighting, MLSMOTE), dan dua mode tuning (hyperparameter tetap, atau hyperparameter kecil plus ambang per label yang dipilih lewat validasi dalam). Transformer (IndoBERTweet, IndoBERT) dipakai sebagai pembanding batas atas. Bagian 11 dan 12 notebook berisi eksperimen lanjutan dan uji lintas korpus ke IndoToxic2024.

## Isi folder

```
Deteksi_Ujaran_Kebencian_MultiLabel.ipynb   satu-satunya notebook: seluruh kode, eksperimen, dan hasil
data/raw/                                   dataset asli (otomatis diunduh bila belum ada, tidak di-push)
data/processed/data_clean.csv               data bersih + kolom id
results/folds.json                          indeks 5 fold. JANGAN dibuat ulang
results/experiments.csv                     satu baris per run (konfigurasi x fold), semua metrik
results/preds/                              skor prediksi out-of-fold per run
results/experiments_ext.csv, preds_ext/     hasil eksperimen lanjutan (bagian 11): ML-ROS, fitur karakter, transformer v2, ensembel
results/crosscorpus*.csv, crosscorpus/      hasil uji lintas korpus (bagian 12)
data/external/                              IndoToxic2024 (otomatis diunduh dari HuggingFace, tidak di-push)
results/*.csv, figures/*.png                tabel dan grafik untuk paper (dpi 300)
requirements.txt                            library yang dibutuhkan
runner/, setup_gpu.bat                      alat bantu opsional untuk menjalankan notebook di GPU laptop
```

Folder `_arsip/` dan `jobs/` tidak ikut ter-push (ada di `.gitignore`). `_arsip/` berisi versi lama (notebook terpisah, `src/`, hasil v1) dan aman dihapus.

## Cara menjalankan

1. Pasang PyTorch sesuai GPU (https://pytorch.org/get-started/locally/), lalu `pip install -r requirements.txt`.
2. Buka notebook dan jalankan semua sel dari atas (Jupyter, VS Code, atau Colab).
3. Konfigurasi ada di sel pertama bagian 1 (`RUN_GRID`, `RUN_GPU_MODELS`, `RUN_BERT`, dan lainnya).

Bagian grid dan IndoBERTweet **bisa dilanjutkan**: run yang sudah tercatat di `results/experiments.csv` dilewati. Karena hasil lengkap sudah ada di repo, menjalankan ulang notebook hanya membuat ulang EDA, tabel, dan grafik (beberapa menit). XGBoost, MLP, dan IndoBERTweet otomatis memakai CUDA bila ada. Tanpa GPU, model tersebut dilewati dan hasil yang sudah ada tetap dipakai.

## Desain eksperimen

| Komponen | Isi |
|---|---|
| Data | 12.773 tweet setelah deduplikasi (dari 13.169 baris mentah) |
| Validasi | 5-fold iterative stratification, seed 42 |
| Fitur | TF-IDF word 1-2 gram, 5.000 fitur, sublinear TF, pemilihan fitur deterministik, di-fit hanya di data latih |
| Algoritma | Complement NB, LinearSVC, XGBoost (GPU), MLP PyTorch (GPU). Random Forest hanya untuk LP |
| Strategi | BR dan CC. LP hanya pembanding baseline |
| Imbalance | none, weight, MLSMOTE + perbaikan hierarki, MLSMOTE murni (ablation) |
| Tuning | fixed, atau tuned (hyperparameter kecil + ambang per label dari 15% data latih) |
| Pembanding | IndoBERTweet (`indolem/indobertweet-base-uncased`), ambang default dan tuned |
| Uji statistik | RQ2: Wilcoxon pada F1 per label + corrected resampled t-test, koreksi Holm. RQ3: Friedman (label sebagai blok) + Nemenyi |

Data uji fold luar tidak pernah dipakai untuk memilih apa pun (fitur, resampling, hyperparameter, ambang).

## Reprodusibilitas

- `TfidfVectorizer(max_features=...)` bawaan scikit-learn memilih fitur dengan sort yang tidak stabil. Di data ini ada ratusan term dengan frekuensi sama di batas potong (fold 0: 921 term seri, hanya 202 masuk), sehingga hasil bisa berbeda antar mesin sampai 0,014 macro F1. Notebook memakai pemecah seri alfabet.
- Pencarian tetangga di MLSMOTE juga dibuat deterministik.
- Setelah perbaikan, hasil di dua mesin berbeda identik untuk 136 dari 145 run model CPU. Sisa selisih paling besar 0,001 macro F1 di level rata-rata konfigurasi (presisi floating point solver LinearSVC). Semua angka di bawah berasal dari satu mesin.

## Hasil utama (rata-rata 5 fold)

Bagian ini berisi angka untuk tim, bukan paragraf paper. Semua angka bisa dilacak ke `results/` dan ke output notebook.

### Kinerja terbaik

| Model | Konfigurasi | Macro F1 | Macro F1 minoritas | Akurasi Jaccard | Waktu latih per fold |
|---|---|---|---|---|---|
| Ensembel transformer (model akhir, bagian 11) | 3 seed IndoBERTweet + IndoBERT-p1, ambang 0,5 | 0,778 | 0,736 | 0,829 | sekitar 40 menit (GPU, 4 model) |
| IndoBERTweet v2 (bagian 11) | teks asli, 10 epoch, tuned | 0,763 | - | - | sekitar 10 menit (GPU) |
| LinearSVC kata + karakter (bagian 11) | CC, none, tuned | 0,693 | 0,643 | 0,769 | sekitar 30 detik |
| IndoBERTweet v1 (bagian 8) | weight, tuned | 0,733 | 0,680 | 0,781 | 250 detik (GPU) |
| LinearSVC | BR, none, tuned | 0,684 | 0,633 | 0,715 | 0,2 detik (+ 9 detik tuning) |
| XGBoost | CC, weight, tuned | 0,677 | 0,635 | 0,760 | 11 detik (GPU) |
| MLP | CC, MLSMOTE, fixed | 0,646 | 0,570 | 0,753 | 57 detik (GPU) |
| Complement NB | BR, weight, tuned | 0,611 | 0,506 | 0,668 | kurang dari 0,1 detik |

Label minoritas: HS_Religion, HS_Race, HS_Physical, HS_Gender, HS_Strong (IRLbl > MeanIR).

### RQ2: tuning ambang dan penanganan imbalance (`results/stats_rq2.csv`)

- **Tuning ambang per label adalah perbaikan paling konsisten.** Macro F1 naik di 21 dari 24 perbandingan fixed ke tuned. Kenaikan terbesar ada di Complement NB (+0,05 sampai +0,11) dan pada strategi BR (LinearSVC BR none +0,053, XGBoost BR none +0,046, MLP BR none +0,042). Kenaikan ini signifikan setelah koreksi Holm pada 7 perbandingan (Wilcoxon), misalnya LinearSVC BR none (p = 0,020) dan XGBoost BR none (p = 0,012). Pada CC, kenaikannya kecil (0,015 atau kurang untuk LinearSVC dan XGBoost).
- **Class weighting (tanpa tuning)** menaikkan F1 di 12 dari 12 label untuk LinearSVC BR (+0,026) dan XGBoost BR (+0,052), keduanya signifikan (Wilcoxon-Holm p = 0,012). Untuk Complement NB BR dan MLP, weighting justru menurunkan kinerja.
- **MLSMOTE dibanding tanpa penanganan tidak signifikan** untuk algoritma mana pun. Untuk XGBoost BR, MLSMOTE lebih rendah daripada weighting di 12 dari 12 label (p = 0,012). MLP adalah satu-satunya model yang cukup terbantu MLSMOTE (CC: +0,058 macro F1, +0,141 F1 minoritas), tetapi juga tidak signifikan setelah koreksi.
- **Setelah ambang di-tuning, pilihan teknik imbalance hampir tidak berpengaruh.** LinearSVC BR tuned: none 0,684, weight 0,674, MLSMOTE 0,673.

### RQ3: perbedaan antar algoritma (`results/stats_rq3_friedman.csv`, `figures/cd_diagram.png`)

- Friedman (12 label sebagai blok) signifikan di 11 dari 12 kombinasi strategi x kondisi x tuning. Satu-satunya yang tidak signifikan adalah BR, none, fixed (p = 0,082).
- LinearSVC punya peringkat rata-rata terbaik di 11 dari 12 kombinasi. Complement NB selalu terakhir atau hampir terakhir.
- Konfigurasi terbaik tiap algoritma: Friedman p < 0,001. Pada Nemenyi, IndoBERTweet berbeda signifikan dari XGBoost (p = 0,036), MLP (p = 0,001), dan Complement NB (p < 0,001), tetapi **tidak** dari LinearSVC (p = 0,615).

### Model ringan vs IndoBERTweet (`results/stats_light_vs_bert.csv`)

- IndoBERTweet unggul +0,049 macro F1 atas LinearSVC terbaik dan lebih baik di 10 dari 12 label (Wilcoxon p = 0,002, corrected t-test p = 0,002). Nemenyi lima arah tidak signifikan karena jauh lebih konservatif. Laporkan keduanya dan jelaskan perbedaannya.
- Biaya: IndoBERTweet sekitar 250 detik per fold di GPU RTX 4060, LinearSVC 0,2 detik per fold di CPU (sekitar 9 detik termasuk tuning).
- IndoBERTweet selalu berhenti di batas 4 epoch, jadi mungkin belum konvergen penuh. Ini keterbatasan, bukan hasil yang sudah dioptimalkan.

### Strategi multi-label dan hierarki label (`results/table_time_violation.csv`)

- Strategi BR menghasilkan prediksi yang melanggar hierarki label pada 22% sampai 52% tweet (misalnya HS_Gender diprediksi tanpa HS). CC untuk LinearSVC, XGBoost, dan MLP hanya 0% sampai 10%.
- Akurasi Jaccard LinearSVC: CC 0,759 vs BR 0,707. Jadi CC lebih cocok bila tujuannya label set yang utuh, sedangkan BR + tuning sedikit lebih baik di macro F1.

### MLSMOTE

- Label yang diproses (IRLbl > MeanIR 6,43): HS_Religion, HS_Race, HS_Physical, HS_Gender, HS_Strong. Rata-rata sekitar 2.100 sampel sintetis per fold.
- MLSMOTE murni menghasilkan sekitar 15% sampel sintetis yang melanggar hierarki label. Perbaikan hierarki tidak mengubah kinerja (LinearSVC BR: 0,640 vs 0,637).

### Perbandingan dengan baseline Ibrohim & Budi (2019) (`results/table_baseline_lp.csv`)

- Baseline: RFDT + LP + word unigram, akurasi Jaccard 66,12% (10-fold).
- Replikasi Random Forest + LP: 70,6%. LinearSVC + LP: 75,0%. Tertinggi keseluruhan: IndoBERTweet 79,9%. Model ringan tertinggi: XGBoost CC 76,5%.
- Perbedaan setup wajib disebut: 5-fold vs 10-fold, data sudah dideduplikasi, fitur TF-IDF 1-2 gram. Selisih angka tidak bisa diatribusikan ke satu faktor saja.

### Analisis kesalahan (`results/error_analysis_*.csv`, `results/explain_top_features_svm.csv`)

- False negative kategori minoritas paling sering diprediksi sebagai HS_Other atau tidak terdeteksi sebagai HS sama sekali. Contoh: dari 249 false negative HS_Religion, 94 tidak terdeteksi sebagai HS dan 82 diprediksi HS_Other.
- Kata pemicu LinearSVC masuk akal secara intuitif (misalnya nama etnis untuk HS_Race, kata ajakan seperti "bubarkan", "lengserkan", "usir" untuk HS_Strong). Ini juga menunjukkan risiko bias leksikal: model bisa menandai kata identitas sebagai ujaran kebencian.
- Kolom `penyebab` di `error_analysis_false_negatives.csv` diisi manual oleh tim.

## Eksperimen lanjutan (bagian 11 dan 12 notebook)

Bagian ini menjawab saran review untuk target jurnal SINTA 3 dan mencoba perbaikan dari literatur serta dari notebook tim lain (IndoBERT, protokol nested, macro F1 0,759). Semua hasil ada di `results/experiments_ext.csv`, skor prediksi di `results/preds_ext/`, dan tabel ringkasan di `results/ext_*.csv`. Tabel utama di atas tidak berubah.

### Transformer versi 2 (`results/ext_transformer_table.csv`, `results/ext_threshold_strategies.csv`)

Empat perubahan dibanding IndoBERTweet versi pertama:
1. Teks masukan memakai tweet asli dengan normalisasi ringan ala IndoBERTweet (USER jadi @USER, URL jadi HTTPURL, emoji didekode lalu diubah ke teks), bukan teks yang sudah dibersihkan untuk TF-IDF.
2. Epoch dipilih dari macro F1 validasi dalam setelah ambang di-tuning, dengan batas 6 atau 10 epoch dan patience 2. Versi pertama selalu berhenti di batas 4 epoch.
3. `pos_weight` memakai akar rasio negatif/positif yang dibatasi 10.
4. Probabilitas validasi dalam disimpan, sehingga ensembel dan pemilihan ambang bisa dilakukan tanpa data uji.

| Model | Macro F1 uji | Catatan |
|---|---|---|
| IndoBERTweet v1 (bagian 8) | 0,733 | ambang tuned |
| IndoBERT-p1 v2 | 0,744 | ambang 0,5, batas 6 epoch |
| IndoBERTweet v2, batas 6 epoch | 0,757 | ambang tuned |
| IndoBERTweet v2, batas 10 epoch | 0,763 | ambang tuned |
| 3 seed IndoBERTweet v2, batas 6 epoch | 0,765 | ensembel |
| IndoBERTweet v2 + IndoBERT-p1 v2, batas 6 epoch | 0,772 | ensembel, ambang 0,5 |
| **3 seed IndoBERTweet + IndoBERT-p1, batas 10 epoch** | **0,778 ± 0,009** | **model akhir, ambang 0,5** |

Model akhir: macro F1 0,778, micro F1 0,825, F1 minoritas 0,736, akurasi Jaccard 0,829, subset accuracy 0,745. F1 per label: HS 0,883, Abusive 0,925, HS_Individual 0,794, HS_Group 0,748, HS_Religion 0,744, HS_Race 0,781, HS_Physical 0,641, HS_Gender 0,682, HS_Other 0,827, HS_Weak 0,777, HS_Moderate 0,695, HS_Strong 0,834.

- Model akhir dipilih dari validasi dalam, bukan dari skor uji. Caranya: untuk tiap konfigurasi dan tiap strategi ambang (0,5, tuned, tuned yang ditarik ke 0,5, rata-rata bootstrap), ambang dipilih di separuh validasi dalam lalu dinilai di separuh lainnya (2-fold, diulang 5 kali). Konfigurasi dan strategi dengan nilai tertinggi menjadi model akhir.
- Ambang tuned penuh hampir selalu kalah dari ambang 0,5 pada transformer. Validasi dalam hanya sekitar 1.500 tweet, sehingga ambang per label untuk label jarang ikut overfit. Pada model klasik, tuning ambang tetap membantu (lihat RQ2).
- Model akhir lebih baik dari IndoBERTweet v1 di 12 dari 12 label (+0,044 macro F1, Wilcoxon-Holm p = 0,0015, corrected t-test p = 0,005) dan dari LinearSVC terbaik kata + karakter (+0,085, 12 dari 12 label, corrected t-test p < 0,001). Label tersulit tetap HS_Physical (0,641) dan HS_Gender (0,682). Angka p final ada di `results/ext_stats_final_model.csv`.
- Selisih model akhir dengan ensembel dua model 6 epoch hanya +0,006 dan tidak signifikan pada corrected t-test (p sekitar 0,38). Artinya perbaikan transformer sudah jenuh di setup ini: tambahan seed, epoch, dan anggota ensembel memberi kenaikan yang lebih kecil dari variasi antar fold.
- Ensembel beberapa seed dari model yang sama hanya sedikit menambah skor. Kenaikan lebih besar datang dari menggabungkan dua model pretrained yang berbeda.
- Batas 10 epoch membantu IndoBERTweet (0,757 menjadi 0,763) tetapi tidak membantu IndoBERT-p1, yang epoch terbaiknya rata-rata 4 sampai 5.
- Variasi antar seed IndoBERTweet sekitar 0,01 macro F1. Selisih yang lebih kecil dari itu sebaiknya tidak ditafsirkan.
- Biaya: satu IndoBERTweet sekitar 7 sampai 12 menit per fold di RTX 4060. Model akhir terdiri dari empat model, jadi biaya latih dan inferensinya sekitar empat kali lipat. Bila biaya penting, ensembel dua model (0,772) atau IndoBERTweet tunggal 10 epoch (0,763) adalah pilihan yang lebih hemat.

### Post-processing hierarki label (`results/ext_postproc_hierarchy.csv`)

Aturan anotasi (HS = 0 berarti semua HS_* = 0, tepat satu target, tepat satu tingkat, minimal satu kategori) diterapkan ke skor tersimpan dari 106 konfigurasi tanpa training ulang. Varian utama (topdown) ditetapkan sebelum hasil dilihat.
- Akurasi Jaccard naik di 99 dari 106 konfigurasi. Kenaikan terbesar ada di strategi BR (Complement NB +0,091, MLP +0,060, XGBoost +0,055, LinearSVC +0,051), karena BR paling sering melanggar hierarki.
- Macro F1 naik di 41 dari 106 konfigurasi. Untuk BR model klasik naik sedikit (Complement NB +0,035, MLP +0,027, LinearSVC +0,009), sedangkan untuk CC dan transformer hampir tidak berubah atau turun sedikit (sekitar -0,005). Tidak ada perubahan macro F1 yang signifikan setelah koreksi Holm.
- Artinya, post-processing ini murah dan berguna bila tujuannya label set yang utuh (misalnya untuk moderasi konten), tetapi bukan cara menaikkan skor per label.

### ML-ROS dan fitur karakter (`results/ext_stats_mlros_features.csv`)

- ML-ROS (oversampling acak multi-label, 20% data latih) juga tidak membantu. LinearSVC BR tuned: tanpa resampling 0,684, ML-ROS 0,676, MLSMOTE 0,673. Karena dua cara oversampling yang berbeda sama-sama gagal, kesimpulan bahwa oversampling tidak cocok untuk fitur TF-IDF di data ini tidak bergantung pada implementasi MLSMOTE.
- Fitur kata + karakter (char_wb 3-5 gram) menaikkan LinearSVC sedikit: CC 0,693 dan BR 0,692, dibanding fitur kata saja 0,684. Pada XGBoost, fitur karakter tidak membantu (0,67 sampai 0,68).
- Tabel precision dan recall label minoritas per kondisi ada di `results/ext_minority_precision_recall.csv`.

### Uji lintas korpus ke IndoToxic2024 (RQ4, `results/crosscorpus_results.csv`)

Model dilatih di seluruh data Ibrohim & Budi lalu diuji di IndoToxic2024 versi IndoDiscourse v2 (sekitar 25 ribu teks setelah agregasi mayoritas anotator). Pemetaan label: HS ke `toxicity`, HS ke `identity_attack`, Abusive ke `profanity_obscenity`.

| Model | F1 HS ke toxicity |
|---|---|
| IndoBERTweet v2 (6 epoch) | 0,394 |
| IndoBERTweet v2 10 epoch (3 seed) | 0,375 sampai 0,395 |
| **Model akhir (3 seed IndoBERTweet + IndoBERT-p1, 10 epoch)** | **0,387** |
| Ensembel IndoBERTweet + IndoBERT-p1 (6 epoch) | 0,355 |
| IndoBERT-p1 v2 (6 dan 10 epoch) | 0,329 dan 0,340 |
| Complement NB | 0,236 |
| LinearSVC kata + karakter | 0,224 |
| LinearSVC kata | 0,210 |

- Semua model turun tajam dari F1 HS sekitar 0,83 sampai 0,88 di data asli. Transformer lebih tahan sekitar 0,13 sampai 0,18 F1 dibanding model klasik.
- Model akhir (precision 0,342, recall 0,444) setara dengan IndoBERTweet tunggal terbaik. Ensembel 6 epoch lebih rendah karena lebih konservatif: memprediksi positif pada 7% teks, sedangkan prevalensi toxicity 8,5%. Variasi antar seed IndoBERTweet di korpus ini sekitar 0,02 F1, jadi selisih antar transformer di tabel ini tidak perlu ditafsirkan berlebihan.
- Untuk HS ke identity_attack, semua model rendah (0,08 sampai 0,17) karena definisinya lebih sempit dari HS Ibrohim.
- 15,8% token dan 80% jenis kata di data 2024 tidak ada di data latih. Kata baru yang paling sering muncul berkaitan dengan topik 2023-2024 (misalnya "hamas", "gibran", "netanyahu").
- Abusive ke profanity hampir tidak bisa dinilai karena prevalensinya hanya 1% dan definisinya berbeda.
- Ini uji generalisasi lintas korpus, bukan uji temporal murni. Korpus 2024 berbeda tahun, platform, pedoman anotasi, dan definisi label.

## Catatan untuk penulisan paper

1. **Hipotesis awal soal MLSMOTE tidak didukung.** Jangan menulis seolah MLSMOTE unggul. Temuan yang didukung data: tuning ambang per label adalah cara termurah dan paling konsisten untuk label minoritas, class weighting membantu model linear dan tree tanpa tuning, dan MLSMOTE hanya membantu MLP.
2. **Usulan framing:** perbandingan penanganan imbalance di tiga level (data: MLSMOTE, algoritma: class weighting, keputusan: tuning ambang) pada 12 label, ditambah peran Classifier Chain dalam menjaga konsistensi hierarki label.
3. **Posisi transformer:** setelah diperbaiki (bagian 11), ensembel transformer unggul sekitar 0,09 macro F1 atas LinearSVC terbaik di 12 dari 12 label, dengan biaya komputasi ribuan kali lipat. Perbaikan transformer sudah jenuh: kenaikan terakhir lebih kecil dari variasi antar fold. Untuk cerita paper, yang penting adalah cara model dipilih (validasi dalam saja, termasuk strategi ambang), bukan angka 0,778 itu sendiri.
4. **Tuning ambang punya dua sisi.** Pada model klasik, tuning ambang adalah perbaikan paling konsisten. Pada transformer, ambang yang dipilih di validasi dalam kecil justru overfit, dan ambang 0,5 lebih baik. Ini bisa menjadi satu paragraf pembahasan.
5. **RQ4 (lintas korpus):** semua model turun tajam di IndoToxic2024. Transformer lebih tahan daripada model klasik, tetapi skor terbaik di data asli tidak menjamin skor terbaik di korpus lain. Sebut sebagai uji lintas korpus, bukan uji temporal.
6. **Keterbatasan yang perlu ditulis:** data hanya Twitter 2018. Hyperparameter transformer hanya sebagian yang diuji (epoch, seed, model). Variasi antar seed sekitar 0,01 macro F1. Label tidak sepenuhnya independen sebagai blok uji statistik. Pemetaan label ke IndoToxic2024 adalah asumsi. Kamus alay bawaan punya beberapa pemetaan keliru (misalnya "neg" menjadi "tawar").
7. **Yang belum dicoba** (bila ada waktu): task-adaptive pretraining (TAPT), asymmetric loss, dan IndoBERT-large. Kodenya sudah ada di notebook, tetapi belum dijalankan karena hasil terakhir menunjukkan kenaikan yang semakin kecil.

## Lisensi dan sitasi data

Dataset berlisensi CC BY-NC-SA 4.0. Wajib mensitasi:

Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter. *Proceedings of the Third Workshop on Abusive Language Online*, 46-57. https://aclanthology.org/W19-3506/

Referensi metode: Charte dkk. (2015) MLSMOTE, *Knowledge-Based Systems* 89. Sechidis dkk. (2011) iterative stratification. Demšar (2006) Friedman dan Nemenyi. Nadeau & Bengio (2003) corrected resampled t-test. Koto dkk. (2021) IndoBERTweet.
