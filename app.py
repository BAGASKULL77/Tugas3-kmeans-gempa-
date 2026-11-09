"""Dashboard Streamlit — Clustering Gempa Bumi Global dengan K-Means.

Jalankan:  streamlit run app.py
Folder artifacts/ dibuat otomatis dari global_earthquakes_seismic.csv bila belum ada
(pipeline sama dengan notebook Clustering_KMeans_Gempa.ipynb, lihat clustering_pipeline.py).
"""
import json
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (diperlukan untuk plot 3D)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from clustering_pipeline import DatasetNotFound, build_artifacts

st.set_page_config(page_title="Clustering Gempa Bumi (K-Means)", page_icon="🌋", layout="wide")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PALET = sns.color_palette("tab10", 10)

FEATURE_LABELS = {
    "depth": "depth — Kedalaman (km)",
    "mag": "mag — Magnitudo",
    "nst": "nst — Jumlah stasiun perekam",
    "dmin": "dmin — Jarak stasiun terdekat (derajat)",
    "gap": "gap — Celah azimut stasiun (derajat)",
    "rms": "rms — Residual waktu tempuh (detik)",
    "sig": "sig — Skor signifikansi",
    "latitude": "latitude — Lintang",
    "longitude": "longitude — Bujur",
}
LABEL_ID = {
    "depth": ("Relatif Dalam", "Dangkal"),
    "mag": ("Magnitudo Lebih Besar", "Magnitudo Lebih Kecil"),
    "nst": ("Stasiun Banyak", "Stasiun Sedikit"),
    "dmin": ("Stasiun Terdekat Jauh", "Stasiun Terdekat Dekat"),
    "gap": ("Celah Azimut Lebar", "Celah Azimut Sempit"),
}
CATEGORY_COLS = {
    "focal_depth_regime": "Rezim kedalaman fokus",
    "magnitude_severity_class": "Kelas keparahan magnitudo",
    "tectonic_boundary_zone": "Zona batas tektonik",
    "tsunami_hazard_potential": "Potensi bahaya tsunami",
}
CORR_COLS = ["depth", "mag", "nst", "dmin", "gap", "rms", "sig", "latitude", "longitude", "tsunami"]


# --------------------------------------------------------------------------------------
# Pemuatan data & komputasi (di-cache)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Menyiapkan hasil clustering (hanya saat pertama kali, sekitar 1 menit)...")
def load_all():
    """Baca artifacts/ bila sudah ada; jika belum (mis. saat deploy), buat otomatis dari dataset mentah."""
    csv_path = ARTIFACTS_DIR / "dataset_hasil_clustering.csv"
    meta_path = ARTIFACTS_DIR / "clustering_meta.json"
    if csv_path.exists() and meta_path.exists():
        df = pd.read_csv(csv_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta
    return build_artifacts(BASE_DIR, ARTIFACTS_DIR)


try:
    df, meta = load_all()
except DatasetNotFound as e:
    st.error(f"{e}\n\nAlternatif: jalankan notebook `Clustering_KMeans_Gempa.ipynb` hingga selesai, "
             "lalu upload folder `artifacts/` ke repository.")
    st.stop()
FEATURES = meta["features"]
LOG_FEATURES = meta["log_features"]
SEED, N_INIT = meta["seed"], meta["n_init"]
K_DEFAULT = meta["k_final"]
eval_df = pd.DataFrame(meta["eval"])


@st.cache_data(show_spinner=False)
def get_scaled():
    """Transformasi log1p pada fitur miring + standarisasi Z-score (sama persis dengan notebook)."""
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


@st.cache_data(show_spinner="Menjalankan K-Means...")
def fit_kmeans(k: int):
    Xs = get_scaled()
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Xs)
    urutan = pd.Series(df["depth"].values).groupby(km.labels_).mean().sort_values().index
    remap = {lama: baru for baru, lama in enumerate(urutan)}
    labels = np.array([remap[l] for l in km.labels_])
    sil = silhouette_samples(Xs, labels)
    return {
        "labels": labels,
        "inertia": float(km.inertia_),
        "silhouette": float(sil.mean()),
        "sil_samples": sil,
        "dbi": float(davies_bouldin_score(Xs, labels)),
    }


@st.cache_data(show_spinner=False)
def get_pca():
    pca = PCA(n_components=2, random_state=SEED).fit(get_scaled())
    return pca.transform(get_scaled()), pca.explained_variance_ratio_, pca.components_


@st.cache_resource(show_spinner="Membuat pairplot...")
def make_pairplot():
    sample = df[FEATURES].sample(1500, random_state=SEED)
    g = sns.pairplot(sample, corner=True, plot_kws=dict(s=8, alpha=0.4, color="#3B6FB6"), diag_kws=dict(color="#3B6FB6"))
    return g.fig


def nama_klaster(zrow, ambang=0.4, top=2):
    urut = sorted(((f, zrow[f]) for f in FEATURES), key=lambda t: -abs(t[1]))
    bagian = [LABEL_ID[f][0 if z > 0 else 1] for f, z in urut[:top] if abs(z) >= ambang]
    return ", ".join(bagian) if bagian else "Profil Mendekati Rata-rata"


def tafsir_silhouette(s: float) -> str:
    if s > 0.70:
        return "struktur klaster **kuat**"
    if s > 0.50:
        return "struktur klaster **cukup baik**"
    if s > 0.25:
        return "struktur klaster **lemah** (klaster agak tumpang tindih)"
    return "**tidak ada struktur klaster yang tegas** (klaster saling tumpang tindih)"


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Sidebar: opsi interaktif
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Interaktif")
K = st.sidebar.slider("Jumlah klaster (K)", min_value=2, max_value=10, value=K_DEFAULT,
                      help=f"K bawaan dari notebook = {K_DEFAULT}. K-Means dijalankan ulang secara langsung.")
st.sidebar.markdown("**Fitur untuk visualisasi**")
VIS_FEATURES = [c for c in FEATURE_LABELS if c in df.columns]
fx = st.sidebar.selectbox("Sumbu X", VIS_FEATURES, index=VIS_FEATURES.index("depth"), format_func=lambda c: FEATURE_LABELS[c])
fy = st.sidebar.selectbox("Sumbu Y", VIS_FEATURES, index=VIS_FEATURES.index("mag"), format_func=lambda c: FEATURE_LABELS[c])
fz = st.sidebar.selectbox("Sumbu Z (plot 3D)", VIS_FEATURES, index=VIS_FEATURES.index("nst"), format_func=lambda c: FEATURE_LABELS[c])
logx = st.sidebar.checkbox("Skala log sumbu X", value=fx in LOG_FEATURES)
logy = st.sidebar.checkbox("Skala log sumbu Y", value=fy in LOG_FEATURES)
st.sidebar.caption("Fitur yang dipakai K-Means: " + ", ".join(FEATURES) + ".")

res = fit_kmeans(K)
labels = res["labels"]
Xs = get_scaled()
dfc = df.copy()
dfc["cluster"] = labels

z_prof = pd.DataFrame(Xs, columns=FEATURES).assign(cluster=labels).groupby("cluster").mean()
names = {c: f"C{c} · {nama_klaster(z_prof.loc[c])}" for c in z_prof.index}
dfc["cluster_name"] = dfc["cluster"].map(names)
sizes = dfc["cluster"].value_counts().sort_index()
cmap = {c: PALET[c % len(PALET)] for c in range(K)}

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("🌋 Clustering Gempa Bumi Global Menggunakan K-Means")
st.markdown(
    f"Proyek ini mengelompokkan **{len(df):,} kejadian gempa bumi global** ({meta['date_min']} s.d. {meta['date_max']}, "
    "magnitudo ≥ 4,5; sumber katalog USGS) berdasarkan karakteristik seismiknya — **kedalaman, magnitudo, jumlah stasiun perekam, "
    "jarak stasiun terdekat, dan celah azimut** — menggunakan algoritma **K-Means**. Gunakan panel kiri untuk mengubah jumlah klaster (K) "
    "dan fitur yang divisualisasikan."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah Klaster (K)", K, delta="K bawaan notebook" if K == K_DEFAULT else f"bawaan notebook: {K_DEFAULT}", delta_color="off")
m2.metric("Silhouette Score", f"{res['silhouette']:.4f}")
m3.metric("Inertia (SSE)", f"{res['inertia']:,.0f}")
m4.metric("Davies-Bouldin", f"{res['dbi']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Ringkasan Dataset", "📊 Distribusi & Korelasi", "🗺️ Hasil Clustering", "📈 Elbow & Silhouette", "🧩 Karakteristik Klaster",
])

# ======================================================================================
# TAB 1 — Ringkasan dataset
# ======================================================================================
with tab1:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baris awal", f"{meta['raw_rows']:,}")
    c2.metric("Kolom awal", meta["raw_cols"])
    c3.metric("Baris setelah pembersihan", f"{meta['clean_rows']:,}")
    c4.metric("Fitur clustering", len(FEATURES))
    c5.metric("Rentang waktu", f"{meta['date_min'][:7]} – {meta['date_max'][:7]}")

    dihapus = sum(n for _, n in meta["cleaning_log"])
    st.markdown("**Tahap pembersihan data**")
    st.dataframe(pd.DataFrame(meta["cleaning_log"], columns=["Tahap pembersihan", "Baris dihapus"]), hide_index=True, width="stretch")
    st.markdown(
        f"**Analisis:** dari {meta['raw_rows']:,} baris awal, {meta['clean_rows']:,} baris ({meta['clean_rows'] / meta['raw_rows']:.2%}) "
        f"dipertahankan; {dihapus} baris dibuang karena bukan gempa tektonik atau memiliki nilai pengukuran jaringan yang tidak terukur "
        "(`nst = gap = dmin = 0` sekaligus). Tidak ada baris yang dihapus sebagai outlier karena gempa besar dan gempa dalam adalah peristiwa "
        "nyata yang justru ingin dikelompokkan."
    )

    st.markdown("**Fitur yang dipertimbangkan untuk clustering**")
    st.dataframe(pd.DataFrame(meta["feature_roles"], columns=["Fitur", "Status", "Alasan"]), hide_index=True, width="stretch")

    st.markdown("**Statistik deskriptif (data bersih)**")
    desc = df[FEATURES + ["rms", "sig"]].describe().T.round(3)
    st.dataframe(desc, width="stretch")
    pct_dangkal = (df["depth"] < 70).mean() * 100
    st.markdown(
        f"**Analisis:** median kedalaman hanya **{df['depth'].median():.1f} km** sedangkan rata-ratanya **{df['depth'].mean():.1f} km** dan nilai "
        f"maksimum **{df['depth'].max():.0f} km** — ciri distribusi miring ke kanan; **{pct_dangkal:.0f}%** gempa terjadi di kedalaman < 70 km (dangkal). "
        f"Magnitudo berkisar {df['mag'].min():.1f}–{df['mag'].max():.1f} dengan median {df['mag'].median():.1f}, artinya sebagian besar gempa berukuran "
        "sedang dan gempa besar jarang terjadi. Perbedaan skala antar kolom yang sangat besar (misalnya `nst` hingga ratusan vs `mag` di kisaran 4–8) "
        "menjadi alasan seluruh fitur distandarisasi sebelum clustering."
    )

# ======================================================================================
# TAB 2 — Distribusi & korelasi
# ======================================================================================
with tab2:
    st.subheader("Distribusi Fitur")
    fig, axes = plt.subplots(2, len(FEATURES), figsize=(3.4 * len(FEATURES), 6.2))
    for j, col in enumerate(FEATURES):
        sns.histplot(df[col], bins=40, ax=axes[0, j], color="#3B6FB6")
        axes[0, j].set_title(col, fontweight="bold")
        axes[0, j].set_xlabel("")
        sns.boxplot(y=df[col], ax=axes[1, j], color="#8FB3E0", fliersize=2)
        axes[1, j].set_ylabel("")
    plt.tight_layout()
    show(fig)
    skew = df[FEATURES].skew().sort_values(ascending=False)
    miring = [f"`{c}` ({v:.2f})" for c, v in skew.items() if v > 1]
    st.markdown(
        f"**Analisis:** fitur dengan kemiringan (skewness) tertinggi adalah `{skew.index[0]}` ({skew.iloc[0]:.2f}). "
        f"Fitur yang sangat miring ke kanan (skewness > 1): {', '.join(miring) if miring else 'tidak ada'}; sebagian besar nilainya kecil dengan ekor panjang "
        f"ke nilai besar (boxplot menunjukkan banyak titik di atas whisker). Karena itu `{', '.join(LOG_FEATURES)}` ditransformasi log1p sebelum standarisasi "
        "agar ekor panjang tidak mendominasi jarak Euclidean pada K-Means. Titik-titik di atas whisker bukan kesalahan data, melainkan gempa dalam atau gempa besar yang nyata."
    )

    st.subheader("Korelasi Antar Fitur")
    corr = df[CORR_COLS].corr()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
    plt.tight_layout()
    show(fig)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["a", "b", "r"]
    pairs["abs"] = pairs["r"].abs()
    top = pairs.sort_values("abs", ascending=False).head(3)
    kalimat = "; ".join(f"`{r.a}`–`{r.b}` (r = {r.r:.2f})" for r in top.itertuples())
    st.markdown(
        f"**Analisis:** tiga pasangan dengan korelasi terkuat: {kalimat}. `sig` hampir identik dengan fungsi `mag`, sehingga tidak dipakai sebagai fitur "
        "agar magnitudo tidak terhitung ganda. Gempa yang lebih besar direkam lebih banyak stasiun (`nst` naik) dengan celah azimut lebih sempit (`gap` turun). "
        "Korelasi `depth` dan `mag` hampir nol, artinya kedalaman dan kekuatan gempa membawa informasi yang berbeda — kombinasi yang cocok untuk membentuk klaster. "
        "`rms` nyaris tidak berkorelasi dengan fitur lain sehingga dikeluarkan dari clustering."
    )

    if st.checkbox("Tampilkan pairplot fitur clustering (sampel 1.500 gempa)"):
        show(make_pairplot())
        st.markdown(
            "**Analisis:** pada scatter antar-fitur tidak tampak kelompok yang terpisah jelas; titik-titik membentuk satu awan kontinu dengan kepadatan "
            "tertinggi di magnitudo rendah dan kedalaman dangkal. Ini menjelaskan mengapa silhouette score bernilai rendah: K-Means di sini berfungsi "
            "sebagai segmentasi karakteristik gempa, bukan pemisahan kelompok yang benar-benar terpisah."
        )

# ======================================================================================
# TAB 3 — Hasil clustering
# ======================================================================================
with tab3:
    st.subheader(f"Scatter Plot Klaster: {fx} vs {fy}  (K = {K})")
    if fx == fy:
        st.warning("Pilih dua fitur yang berbeda pada panel kiri agar scatter plot informatif.")
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(K):
        m = labels == c
        ax.scatter(dfc[fx].values[m], dfc[fy].values[m], s=8, alpha=0.5, color=cmap[c], label=names[c])
    if logx and (dfc[fx] > 0).all():
        ax.set_xscale("log")
    if logy and (dfc[fy] > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel(FEATURE_LABELS[fx])
    ax.set_ylabel(FEATURE_LABELS[fy])
    ax.legend(markerscale=2.5, fontsize=8)
    plt.tight_layout()
    show(fig)

    # analisis dinamis scatter
    mean_x = dfc.groupby("cluster")[fx].mean()
    mean_y = dfc.groupby("cluster")[fy].mean()
    idx = np.random.RandomState(0).choice(len(dfc), size=min(3000, len(dfc)), replace=False)
    tx = np.log1p(dfc[fx].values) if (logx and (dfc[fx] >= 0).all()) else dfc[fx].values
    ty = np.log1p(dfc[fy].values) if (logy and (dfc[fy] >= 0).all()) else dfc[fy].values
    if fx != fy:
        X2 = StandardScaler().fit_transform(np.column_stack([tx, ty]))
        sil2 = silhouette_score(X2[idx], labels[idx])
        teks_sil2 = (f"Jika hanya dua fitur ini yang dilihat, silhouette-nya {sil2:.3f} → {tafsir_silhouette(sil2)} pada tampilan 2D ini; "
                     "klaster sebenarnya dibentuk dari lima fitur sekaligus sehingga pemisahannya tidak selalu tampak pada satu pasang sumbu.")
    else:
        teks_sil2 = ""
    st.markdown(
        f"**Analisis:** pada sumbu `{fx}`, klaster dengan rata-rata tertinggi adalah **{names[int(mean_x.idxmax())]}** ({mean_x.max():.2f}) dan terendah "
        f"**{names[int(mean_x.idxmin())]}** ({mean_x.min():.2f}). Pada sumbu `{fy}`, tertinggi **{names[int(mean_y.idxmax())]}** ({mean_y.max():.2f}) "
        f"dan terendah **{names[int(mean_y.idxmin())]}** ({mean_y.min():.2f}). {teks_sil2}"
    )

    st.subheader("Proyeksi PCA 2D")
    Xp, evr, comps = get_pca()
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for c in range(K):
        m = labels == c
        ax.scatter(Xp[m, 0], Xp[m, 1], s=8, alpha=0.5, color=cmap[c], label=f"C{c}")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.legend(markerscale=2.5, fontsize=8)
    plt.tight_layout()
    show(fig)
    load1 = pd.Series(comps[0], index=FEATURES).abs().sort_values(ascending=False)
    load2 = pd.Series(comps[1], index=FEATURES).abs().sort_values(ascending=False)
    st.markdown(
        f"**Analisis:** dua komponen utama merangkum **{evr.sum():.0%}** varians kelima fitur (PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%}), jadi gambar ini cukup mewakili "
        f"struktur data. PC1 didominasi `{load1.index[0]}` dan `{load1.index[1]}`, sedangkan PC2 oleh `{load2.index[0]}` dan `{load2.index[1]}`. "
        "Klaster tampak bersebelahan dan saling bersinggungan tanpa celah — sejalan dengan silhouette yang rendah; batas klaster adalah garis pemisah K-Means "
        "pada awan data yang kontinu."
    )

    st.subheader("Sebaran Geografis Klaster")
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for c in range(K):
        m = labels == c
        ax.scatter(dfc["longitude"].values[m], dfc["latitude"].values[m], s=6, alpha=0.55, color=cmap[c], label=names[c])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Bujur (longitude)")
    ax.set_ylabel("Lintang (latitude)")
    ax.legend(markerscale=3, fontsize=8, loc="lower left")
    plt.tight_layout()
    show(fig)
    zona = pd.crosstab(dfc["cluster"], dfc["tectonic_boundary_zone"], normalize="index") * 100
    dom = {c: (zona.loc[c].idxmax(), zona.loc[c].max()) for c in zona.index}
    st.markdown(
        "**Analisis:** peta memperlihatkan gempa mengikuti jalur batas lempeng — terutama Cincin Api Pasifik, punggung tengah samudra, dan sabuk Alpide–Himalaya. "
        "Klaster **tidak** dibentuk dari koordinat, tetapi pola geografisnya tetap bermakna: "
        + "; ".join(f"C{c}: {z} ({p:.0f}%)" for c, (z, p) in dom.items())
        + ". Klaster dengan banyak gempa dalam cenderung terkonsentrasi di zona subduksi, sedangkan klaster stasiun sedikit/gap lebar banyak muncul di wilayah samudra "
        "yang jauh dari jaringan seismograf."
    )

    if st.checkbox("Tampilkan scatter plot 3D (sampel 3.000 gempa)"):
        samp = dfc.sample(3000, random_state=SEED)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for c in range(K):
            s = samp[samp["cluster"] == c]
            ax.scatter(s[fx], s[fy], s[fz], s=8, alpha=0.6, color=cmap[c], label=f"C{c}")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_zlabel(fz)
        ax.legend()
        plt.tight_layout()
        show(fig)
        st.markdown(
            f"**Analisis:** tampilan 3D menambah dimensi `{fz}` pada pasangan `{fx}`–`{fy}`. Jika klaster yang bertumpuk di 2D mulai terpisah pada sumbu ketiga, "
            "berarti `" + fz + "` ikut berperan membedakan klaster. Putar mental sumbu-sumbu ini dengan mengganti pilihan fitur di panel kiri."
        )

# ======================================================================================
# TAB 4 — Elbow & silhouette
# ======================================================================================
with tab4:
    st.subheader("Elbow Method dan Silhouette Score")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(eval_df["K"], eval_df["inertia"], "o-", color="#1F4E9C", lw=2)
    axes[0].axvline(meta["k_elbow"], color="#D9534F", ls="--", label=f"Titik siku (jarak terjauh dari garis) K = {meta['k_elbow']}")
    axes[0].axvline(K, color="#2E8B57", ls=":", lw=2, label=f"K dipilih = {K}")
    axes[0].set(title="Elbow Method (Inertia)", xlabel="Jumlah klaster (K)", ylabel="Inertia (SSE)")
    axes[0].legend(fontsize=8)
    axes[1].plot(eval_df["K"], eval_df["silhouette"], "s-", color="#C9722B", lw=2)
    axes[1].axvline(K, color="#2E8B57", ls=":", lw=2, label=f"K dipilih = {K}")
    axes[1].set(title="Silhouette Score", xlabel="Jumlah klaster (K)", ylabel="Silhouette")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(list(eval_df["K"]))
    plt.tight_layout()
    show(fig)
    k_sil_all = int(eval_df.loc[eval_df["silhouette"].idxmax(), "K"])
    drop_first = (eval_df["inertia"].iloc[0] - eval_df["inertia"].iloc[1]) / eval_df["inertia"].iloc[0] * 100
    drop_last = (eval_df["inertia"].iloc[-2] - eval_df["inertia"].iloc[-1]) / eval_df["inertia"].iloc[-2] * 100
    st.markdown(
        f"**Analisis:** inertia turun terus saat K bertambah — dari {eval_df['inertia'].iloc[0]:,.0f} (K = 2) menjadi {eval_df['inertia'].iloc[-1]:,.0f} "
        f"(K = {int(eval_df['K'].iloc[-1])}). Penurunan terbesar terjadi di awal ({drop_first:.1f}% dari K = 2 ke 3) dan makin kecil di akhir ({drop_last:.1f}%), "
        f"namun kurvanya melandai mulus tanpa siku yang tajam; metode jarak-terjauh-dari-garis menunjuk K = {meta['k_elbow']}. Silhouette tertinggi ada pada K = {k_sil_all} "
        f"({eval_df['silhouette'].max():.3f}) dan cenderung turun seiring K naik. Notebook memakai K = {K_DEFAULT} (silhouette tertinggi untuk K ≥ 3, selisih kecil dari K = 2) "
        "agar segmentasi lebih informatif. Karena semua nilai silhouette berada di sekitar 0,2, pemilihan K bersifat pertimbangan interpretasi, bukan batas yang tegas — itulah gunanya slider K."
    )
    tbl = eval_df.rename(columns={"inertia": "Inertia (SSE)", "silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin"})
    st.dataframe(tbl.round(4), hide_index=True, width="stretch")

    st.subheader(f"Silhouette per Klaster (K = {K})")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y_lower = 10
    for c in range(K):
        v = np.sort(res["sil_samples"][labels == c])
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(v)), 0, v, color=cmap[c], alpha=0.8)
        ax.text(-0.04, y_lower + 0.5 * len(v), f"C{c}", fontsize=9)
        y_lower += len(v) + 10
    ax.axvline(res["silhouette"], color="red", ls="--", label=f"Rata-rata = {res['silhouette']:.3f}")
    ax.set(xlabel="Nilai silhouette", ylabel="Anggota klaster (terurut)")
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    show(fig)
    sil_c = pd.Series(res["sil_samples"]).groupby(labels).mean()
    neg = (res["sil_samples"] < 0).mean() * 100
    st.markdown(
        f"**Analisis:** silhouette rata-rata K = {K} adalah **{res['silhouette']:.3f}** → {tafsir_silhouette(res['silhouette'])}. Klaster paling kompak: "
        f"**{names[int(sil_c.idxmax())]}** ({sil_c.max():.3f}); paling tumpang tindih: **{names[int(sil_c.idxmin())]}** ({sil_c.min():.3f}). "
        f"Sebanyak **{neg:.1f}%** gempa memiliki silhouette negatif, yaitu lebih dekat ke klaster lain daripada klaster sendiri (berada di perbatasan). "
        f"Inertia {res['inertia']:,.0f} adalah jumlah kuadrat jarak ke centroid pada ruang fitur terstandarisasi; nilainya baru bermakna jika dibandingkan antar K."
    )

# ======================================================================================
# TAB 5 — Karakteristik klaster
# ======================================================================================
with tab5:
    st.subheader(f"Ringkasan Karakteristik Setiap Klaster (K = {K})")
    PROFIL_COLS = FEATURES + ["rms", "sig", "tsunami"]
    stat = st.radio("Statistik", ["Rata-rata (mean)", "Median"], horizontal=True)
    agg = dfc.groupby("cluster")[PROFIL_COLS].mean() if stat.startswith("Rata") else dfc.groupby("cluster")[PROFIL_COLS].median()
    tabel = agg.copy()
    tabel.insert(0, "nama klaster", [names[c] for c in tabel.index])
    tabel.insert(1, "jumlah", sizes)
    tabel.insert(2, "%", (100 * sizes / len(dfc)).round(1))
    st.dataframe(tabel.round(2), width="stretch")
    glob = dfc[PROFIL_COLS].mean() if stat.startswith("Rata") else dfc[PROFIL_COLS].median()
    st.caption("Kolom `tsunami` = proporsi gempa yang memicu peringatan tsunami. Rata-rata global: "
               + ", ".join(f"{k} = {v:.2f}" for k, v in glob.items() if k in FEATURES))
    besar = sizes.idxmax()
    kecil = sizes.idxmin()
    st.markdown(
        f"**Analisis:** klaster terbesar adalah **{names[besar]}** ({100 * sizes[besar] / len(dfc):.1f}% data) dan terkecil **{names[kecil]}** "
        f"({100 * sizes[kecil] / len(dfc):.1f}%). Nama klaster dibuat otomatis dari dua fitur yang paling menyimpang dari rata-rata global (|z| ≥ 0,4). "
        "Perbandingan mean dan median berguna untuk melihat pengaruh ekor panjang: jika mean jauh di atas median (misalnya pada `depth`), klaster tersebut berisi "
        "sebagian gempa sangat dalam yang menarik rata-ratanya."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Profil klaster (z-score rata-rata fitur)**")
        zt = z_prof.copy()
        zt.index = [names[c] for c in zt.index]
        fig, ax = plt.subplots(figsize=(7, 1.2 + 0.7 * K))
        sns.heatmap(zt, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        show(fig)
    with col_b:
        fbar = st.selectbox("Fitur untuk bar chart", PROFIL_COLS, format_func=lambda c: FEATURE_LABELS.get(c, c))
        fig, ax = plt.subplots(figsize=(7, 1.2 + 0.7 * K + 1.2))
        vals = agg[fbar]
        ax.bar([f"C{c}" for c in vals.index], vals.values, color=[cmap[c] for c in vals.index])
        ax.axhline(glob[fbar], color="k", ls="--", lw=1, label=f"global = {glob[fbar]:.2f}")
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(f"{stat.split()[0]} {fbar}")
        ax.legend()
        plt.tight_layout()
        show(fig)
    zc, zf = z_prof.abs().stack().sort_values(ascending=False).index[0]
    zv = z_prof.loc[zc, zf]
    st.markdown(
        f"**Analisis:** pada heatmap, warna merah berarti di atas rata-rata global dan biru di bawahnya. Penyimpangan terbesar ada pada fitur `{zf}` di klaster "
        f"**{names[zc]}** (z = {zv:+.2f}). Bar chart di sebelahnya memperlihatkan nilai fitur pilihan per klaster dibandingkan garis rata-rata/median global "
        f"(`{fbar}`: tertinggi pada C{vals.idxmax()}, terendah pada C{vals.idxmin()})."
    )

    st.markdown("---")
    st.subheader("Komposisi Kategori per Klaster (validasi eksternal)")
    cat = st.selectbox("Kolom kategori", list(CATEGORY_COLS), format_func=lambda c: CATEGORY_COLS[c])
    ct = pd.crosstab(dfc["cluster"], dfc[cat], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.6 * K))
    ct.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", width=0.75)
    ax.set_yticklabels([f"C{c}" for c in ct.index])
    ax.invert_yaxis()
    ax.set_xlabel("% anggota klaster")
    ax.set_xlim(0, 100)
    ax.legend(title=CATEGORY_COLS[cat], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    show(fig)
    dom_kat = ct.idxmax(axis=1)
    st.markdown(
        "**Analisis:** kolom kategori ini tidak dipakai untuk membentuk klaster, sehingga dapat dipakai untuk mengecek apakah klaster bermakna secara geofisika. "
        + " ".join(f"C{c}: **{dom_kat[c]}** ({ct.loc[c, dom_kat[c]]:.0f}%)." for c in ct.index)
        + " Klaster yang didominasi satu kategori menandakan karakteristik yang konsisten; klaster dengan komposisi campuran menandakan batas yang lebih kabur."
    )

    st.markdown("---")
    st.subheader("Interpretasi Tiap Klaster")
    g = dfc[FEATURES].mean()
    for c in range(K):
        sub = dfc[dfc["cluster"] == c]
        dom_r = sub["focal_depth_regime"].value_counts(normalize=True)
        with st.expander(f"{names[c]} — {len(sub):,} gempa ({100 * len(sub) / len(dfc):.1f}%)", expanded=(c == 0)):
            st.markdown(
                f"- Kedalaman rata-rata **{sub['depth'].mean():.1f} km** (global {g['depth']:.1f}), magnitudo **{sub['mag'].mean():.2f}** (global {g['mag']:.2f}).\n"
                f"- Stasiun perekam **{sub['nst'].mean():.0f}** (global {g['nst']:.0f}); dmin **{sub['dmin'].mean():.1f}°** (global {g['dmin']:.1f}); "
                f"gap **{sub['gap'].mean():.0f}°** (global {g['gap']:.0f}).\n"
                f"- Rezim kedalaman dominan: **{dom_r.index[0]}** ({100 * dom_r.iloc[0]:.0f}%); proporsi peringatan tsunami: **{100 * sub['tsunami'].mean():.1f}%**."
            )

    unduh = dfc[["id", "place", "latitude", "longitude"] + FEATURES + ["cluster", "cluster_name"]]
    st.download_button("⬇️ Unduh data dengan label klaster (CSV)", unduh.to_csv(index=False).encode("utf-8"),
                       file_name=f"gempa_klaster_K{K}.csv", mime="text/csv")
