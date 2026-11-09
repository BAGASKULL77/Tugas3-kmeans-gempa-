"""Pipeline clustering K-Means gempa bumi — versi skrip dari notebook Clustering_KMeans_Gempa.ipynb.

Dipakai oleh app.py agar folder `artifacts/` otomatis dibuat saat aplikasi dijalankan/di-deploy
(misalnya di Streamlit Community Cloud), tanpa harus menjalankan notebook lebih dulu.
Logika di sini SAMA dengan Langkah 2-7 pada notebook.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

CSV_NAME = "global_earthquakes_seismic.csv"
PARQUET_NAME = "global_earthquakes_seismic.parquet"

SEED = 42
N_INIT = 20
FEATURES = ["depth", "mag", "nst", "dmin", "gap"]
LOG_FEATURES = ["depth", "dmin", "nst"]
ZERO_COLS = ["nst", "gap", "dmin"]
K_RANGE = list(range(2, 11))
K_MIN_PRAKTIS = 3
K_OVERRIDE = None

ORDINAL_MAPS = {
    "magnitude_severity_class": {"moderate": 0, "strong": 1, "major": 2, "great_catastrophic": 3},
    "focal_depth_regime": {"shallow_crustal": 0, "intermediate_subduction": 1, "deep_mantle_slab": 2},
    "tsunami_hazard_potential": {"inland_negligible": 0, "moderate_coastal_alert": 1, "critical_tsunamigenic": 2},
}
ENC_NAMES = {"magnitude_severity_class": "severity_enc", "focal_depth_regime": "depth_regime_enc",
             "tsunami_hazard_potential": "tsunami_hazard_enc"}

LABEL_ID = {
    "depth": ("Relatif Dalam", "Dangkal"),
    "mag": ("Magnitudo Lebih Besar", "Magnitudo Lebih Kecil"),
    "nst": ("Stasiun Banyak", "Stasiun Sedikit"),
    "dmin": ("Stasiun Terdekat Jauh", "Stasiun Terdekat Dekat"),
    "gap": ("Celah Azimut Lebar", "Celah Azimut Sempit"),
}

PERAN_FITUR = [
    ["depth", "DIPAKAI", "Kedalaman hiposenter (km): pembeda utama jenis gempa dangkal vs dalam."],
    ["mag", "DIPAKAI", "Magnitudo: ukuran kekuatan gempa."],
    ["nst", "DIPAKAI", "Jumlah stasiun seismik yang merekam: indikator seberapa jelas gempa terekam."],
    ["dmin", "DIPAKAI", "Jarak horizontal ke stasiun terdekat (derajat): menggambarkan kerapatan jaringan di sekitar episentrum."],
    ["gap", "DIPAKAI", "Celah azimut stasiun (derajat): kualitas geometri jaringan pemantau."],
    ["sig", "TIDAK", "Skor signifikansi turunan magnitudo (korelasi dengan mag = 0,80) -> redundan, akan menggandakan bobot magnitudo."],
    ["rms", "TIDAK", "Residual waktu tempuh: nyaris tidak berkorelasi dengan fitur lain (|r| < 0,15) sehingga hanya menambah derau."],
    ["latitude / longitude", "TIDAK", "Koordinat geografis: jarak Euclidean pada derajat tidak valid di bola bumi. Dipakai untuk peta, bukan untuk clustering."],
    ["tsunami", "TIDAK", "Variabel biner (0/1) dan sangat jarang (<1%): dipakai untuk profil klaster."],
    ["confidence_score", "TIDAK", "Anotasi kualitas data (hanya 2 nilai), bukan pengukuran fisik."],
]


class DatasetNotFound(FileNotFoundError):
    pass


def find_dataset(base_dir: Path) -> Path:
    """Cari file dataset (CSV, lalu Parquet) di folder proyek, subfolder data/, atau folder induk."""
    base_dir = Path(base_dir)
    for name in (CSV_NAME, PARQUET_NAME):
        for folder in (base_dir, base_dir / "data", base_dir.parent):
            p = folder / name
            if p.exists():
                return p
    for name in (CSV_NAME, PARQUET_NAME):
        found = list(base_dir.rglob(name))
        if found:
            return found[0]
    raise DatasetNotFound(
        f"File {CSV_NAME} (atau {PARQUET_NAME}) tidak ditemukan di {base_dir}. "
        "Pastikan file dataset ikut di-upload ke repository GitHub, satu folder dengan app.py."
    )


def _knee_point(ks, ys):
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ys - ys.min()) / (ys.max() - ys.min())
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]) / np.hypot(y[-1] - y[0], x[-1] - x[0])
    return int(ks[np.argmax(d)])


def _nama_klaster(zrow, ambang=0.4, top=2):
    urut = sorted(((f, zrow[f]) for f in FEATURES), key=lambda t: -abs(t[1]))
    bagian = [LABEL_ID[f][0 if z > 0 else 1] for f, z in urut[:top] if abs(z) >= ambang]
    return ", ".join(bagian) if bagian else "Profil Mendekati Rata-rata"


def run_pipeline(data_path: Path):
    """Jalankan seluruh pipeline; kembalikan (df_hasil, meta)."""
    data_path = Path(data_path)
    df_raw = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    df_raw["waktu"] = pd.to_datetime(df_raw["time"], unit="ms")

    # --- Langkah 3.1: pembersihan data
    df = df_raw.copy()
    cleaning_log = []

    def hapus(data, mask, alasan):
        cleaning_log.append([alasan, int(mask.sum())])
        return data.loc[~mask].copy()

    df = hapus(df, df["type"] != "earthquake", "Bukan gempa tektonik (type = landslide)")
    df = hapus(df, (df[ZERO_COLS] == 0).all(axis=1), "Nilai tak terukur (nst = gap = dmin = 0)")
    df = hapus(df, df.duplicated(subset="id"), "ID duplikat")
    df = hapus(df, df.duplicated(subset=["time", "latitude", "longitude", "depth", "mag"]),
               "Baris duplikat (waktu, lokasi, kedalaman, magnitudo)")
    df = df.reset_index(drop=True)

    # --- Langkah 3.3: encoding
    for col, mapping in ORDINAL_MAPS.items():
        df[ENC_NAMES[col]] = df[col].map(mapping)
    df = pd.concat([
        df,
        pd.get_dummies(df["tectonic_boundary_zone"], prefix="zone", dtype=int),
        pd.get_dummies(df["magType"], prefix="magType", dtype=int),
    ], axis=1)

    # --- Langkah 3.4: log1p + standarisasi
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    X_scaled = StandardScaler().fit_transform(X)
    for j, c in enumerate(FEATURES):
        df["z_" + c] = X_scaled[:, j]

    # --- Langkah 4: pilih K
    hasil = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X_scaled)
        hasil.append({
            "K": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X_scaled, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, km.labels_)),
        })
    eval_df = pd.DataFrame(hasil)
    k_elbow = _knee_point(eval_df["K"], eval_df["inertia"])
    kandidat = eval_df[eval_df["K"] >= K_MIN_PRAKTIS]
    k_sil = int(kandidat.loc[kandidat["silhouette"].idxmax(), "K"])
    k_final = K_OVERRIDE or k_sil

    kmeans = KMeans(n_clusters=k_final, n_init=N_INIT, random_state=SEED).fit(X_scaled)
    urutan = df.groupby(kmeans.labels_)["depth"].mean().sort_values().index
    remap = {lama: baru for baru, lama in enumerate(urutan)}
    df["cluster"] = np.array([remap[l] for l in kmeans.labels_])

    # --- Langkah 5: evaluasi & penamaan
    labels = df["cluster"].values
    centroids = np.vstack([X_scaled[labels == c].mean(axis=0) for c in range(k_final)])
    inertia = float(((X_scaled - centroids[labels]) ** 2).sum())
    sil_global = float(silhouette_samples(X_scaled, labels).mean())
    z_profil = df.groupby("cluster")[["z_" + c for c in FEATURES]].mean()
    z_profil.columns = FEATURES
    nama = {c: f"C{c} · {_nama_klaster(z_profil.loc[c])}" for c in z_profil.index}
    df["cluster_name"] = df["cluster"].map(nama)

    from sklearn.feature_selection import f_classif
    F, _ = f_classif(X_scaled, labels)
    top_features = pd.Series(F, index=FEATURES).sort_values(ascending=False).index.tolist()

    kolom_simpan = (["id", "waktu", "place", "latitude", "longitude", "depth", "mag", "magType", "sig", "nst", "dmin",
                     "rms", "gap", "tsunami", "confidence_score", "magnitude_severity_class", "focal_depth_regime",
                     "tsunami_hazard_potential", "tectonic_boundary_zone"] + list(ENC_NAMES.values())
                    + [c for c in df.columns if c.startswith(("zone_", "magType_"))] + ["z_" + c for c in FEATURES])
    df_hasil = df[kolom_simpan + ["cluster", "cluster_name"]].copy()

    meta = {
        "features": FEATURES,
        "log_features": LOG_FEATURES,
        "k_final": int(k_final),
        "k_elbow": int(k_elbow),
        "k_range": K_RANGE,
        "seed": SEED,
        "n_init": N_INIT,
        "eval": eval_df.round(6).to_dict(orient="records"),
        "raw_rows": int(len(df_raw)),
        "raw_cols": int(df_raw.shape[1] - 1),
        "clean_rows": int(len(df)),
        "cleaning_log": [[a, int(n)] for a, n in cleaning_log],
        "date_min": f"{df['waktu'].min():%Y-%m-%d}",
        "date_max": f"{df['waktu'].max():%Y-%m-%d}",
        "feature_roles": PERAN_FITUR,
        "top_features": top_features,
        "silhouette_final": sil_global,
        "inertia_final": inertia,
    }
    return df_hasil, meta


def build_artifacts(base_dir: Path, artifacts_dir: Path):
    """Buat artifacts/dataset_hasil_clustering.csv dan clustering_meta.json dari dataset mentah."""
    artifacts_dir = Path(artifacts_dir)
    df_hasil, meta = run_pipeline(find_dataset(base_dir))
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        df_hasil.to_csv(artifacts_dir / "dataset_hasil_clustering.csv", index=False)
        with open(artifacts_dir / "clustering_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # server read-only: hasil tetap dikembalikan dan dipakai langsung dari memori
    return df_hasil, meta
