import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
)

from utils import (
    clean_commodity_series,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_dataset,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Input Parameter Model — KomoditasAI",
    page_icon="⚙️",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input parameter",
    title=" Input Parameter Model",
    caption=(
        "Konfigurasi pembagian data, feature engineering, "
        "dan tuning Random Forest sebelum model dijalankan."
    ),
)


# ============================================================
# DATASET
# ============================================================

df, date_column, commodity_column = require_dataset()

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=True,
)

df[commodity_column] = clean_commodity_series(
    df,
    commodity_column,
)

df = (
    df
    .dropna(
        subset=[
            date_column,
            commodity_column,
        ]
    )
    .sort_values(date_column)
)

harga = (
    df
    .set_index(date_column)[commodity_column]
    .astype(float)
    .sort_index()
)

harga = harga[
    ~harga.index.duplicated(
        keep="last"
    )
]


# ============================================================
# GLOBAL CONFIG
# ============================================================

RANDOM_STATE = 42

END_DATE = harga.index.max()

MIN_LAG = 1
MAX_LAG = 30

DEFAULT_TRAIN_END_DATE = pd.Timestamp(
    "2025-12-31"
)

RF_ITERATIONS_BY_PROFILE = {
    "Fast": 15,
    "Balanced": 24,
    "Thorough": 40,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_time_series_features(
    series: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """
    Membuat fitur time series tanpa data leakage.

    Seluruh rolling statistics menggunakan shift(1)
    sehingga hanya menggunakan informasi sebelum waktu t.
    """

    s = series.astype(float).copy()

    frame = pd.DataFrame(
        index=s.index
    )

    frame["target"] = s

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in range(
        1,
        max_lag + 1,
    ):
        frame[
            f"lag_{lag}"
        ] = s.shift(lag)

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    shifted = s.shift(1)

    for window in [
        3,
        5,
        7,
        10,
        14,
        21,
        30,
    ]:

        frame[
            f"roll_mean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        frame[
            f"roll_std_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )

        frame[
            f"roll_min_{window}"
        ] = (
            shifted
            .rolling(window)
            .min()
        )

        frame[
            f"roll_max_{window}"
        ] = (
            shifted
            .rolling(window)
            .max()
        )

    # --------------------------------------------------------
    # Exponential Moving Average
    # --------------------------------------------------------

    frame["ewm_mean_5"] = (
        shifted
        .ewm(
            span=5,
            adjust=False,
        )
        .mean()
    )

    frame["ewm_mean_14"] = (
        shifted
        .ewm(
            span=14,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    frame["diff_1"] = (
        s.shift(1)
        - s.shift(2)
    )

    frame["diff_5"] = (
        s.shift(1)
        - s.shift(6)
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    idx = pd.DatetimeIndex(
        frame.index
    )

    frame["day_of_week"] = (
        idx.dayofweek
    )

    frame["day_of_month"] = (
        idx.day
    )

    frame["month"] = (
        idx.month
    )

    frame["quarter"] = (
        idx.quarter
    )

    frame["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * idx.dayofyear
        / 365.25
    )

    frame["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * idx.dayofyear
        / 365.25
    )

    return (
        frame
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


def valid_tscv(
    n_samples,
    requested_splits,
):
    """
    Menentukan jumlah fold TimeSeriesSplit
    yang valid berdasarkan jumlah observasi.
    """

    n_splits = min(
        requested_splits,
        max(
            2,
            n_samples // 60,
        ),
    )

    n_splits = min(
        n_splits,
        n_samples - 1,
    )

    return TimeSeriesSplit(
        n_splits=n_splits
    )


# ============================================================
# SECTION 01
# DATASET AKTIF
# ============================================================

st.markdown(
    "### 01 · Dataset Aktif"
)

st.caption(
    "Ringkasan dataset yang digunakan "
    "sebagai dasar proses pemodelan."
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Komoditas",
        commodity_column,
    )


with m2:

    st.metric(
        "Observasi",
        f"{len(harga):,}".replace(
            ",",
            ".",
        ),
    )


with m3:

    st.metric(
        "Tanggal Mulai",
        harga.index.min().strftime(
            "%d %b %Y"
        ),
    )


with m4:

    st.metric(
        "Tanggal Akhir",
        harga.index.max().strftime(
            "%d %b %Y"
        ),
    )


# ============================================================
# SECTION 02
# TRAIN TEST SPLIT
# ============================================================

st.markdown(
    "### 02 · Pembagian Data Train & Test"
)

with st.container(border=True):

    split_col, summary_col = st.columns(
        [1.25, 1],
        gap="large",
    )

    # --------------------------------------------------------
    # SPLIT METHOD
    # --------------------------------------------------------

    with split_col:

        split_method = st.radio(
            "Metode Pembagian Data",
            [
                "Persentase",
                "Tanggal (Advanced)",
            ],
            horizontal=True,
        )

        # ----------------------------------------------------
        # PERCENTAGE
        # ----------------------------------------------------

        if split_method == "Persentase":

            train_ratio = st.slider(
                "Proporsi Data Train (%)",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
            )

            test_ratio = (
                100
                - train_ratio
            )

            split_index = int(
                len(harga)
                * train_ratio
                / 100
            )

            train_series = (
                harga
                .iloc[:split_index]
                .copy()
            )

            test_series = (
                harga
                .iloc[split_index:]
                .copy()
            )

            split_note = (
                f"Data dibagi secara kronologis "
                f"{train_ratio}% train dan "
                f"{test_ratio}% test."
            )

        # ----------------------------------------------------
        # DATE ADVANCED
        # ----------------------------------------------------

        else:

            train_end_date = st.date_input(
                "Tanggal Akhir Data Train",
                value=min(
                    DEFAULT_TRAIN_END_DATE.date(),
                    harga.index.max().date(),
                ),
                min_value=harga.index.min().date(),
                max_value=harga.index.max().date(),
            )

            train_series = (
                harga.loc[
                    harga.index
                    <= pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            test_series = (
                harga.loc[
                    harga.index
                    > pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            total = len(harga)

            train_ratio = round(
                len(train_series)
                / total
                * 100,
                1,
            )

            test_ratio = round(
                len(test_series)
                / total
                * 100,
                1,
            )

            split_note = (
                "Data dibagi berdasarkan "
                "tanggal yang dipilih."
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        len(train_series) == 0
        or len(test_series) == 0
    ):

        st.error(
            "Pembagian data menghasilkan "
            "train atau test kosong. "
            "Silakan ubah rasio atau tanggal."
        )

        st.stop()

    effective_train_end = (
        train_series.index.max()
    )

    effective_test_start = (
        test_series.index.min()
    )

    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    with summary_col:

        st.caption(
            "RINGKASAN PEMBAGIAN"
        )

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Train",
                f"{len(train_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        with m2:

            st.metric(
                "Test",
                f"{len(test_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        st.caption(
            f"Train: "
            f"{effective_train_end.strftime('%d %b %Y')}"
        )

        st.caption(
            f"Test mulai: "
            f"{effective_test_start.strftime('%d %b %Y')}"
        )


st.info(
    split_note
)


# ============================================================
# SPLIT VISUALIZATION
# ============================================================

with st.expander(
    "📈 Lihat Visualisasi Pembagian Data",
    expanded=False,
):

    fig, ax = plt.subplots(
        figsize=(13, 4)
    )

    ax.plot(
        train_series.index,
        train_series,
        label="Train",
        linewidth=1.3,
    )

    ax.plot(
        test_series.index,
        test_series,
        label="Test",
        linewidth=1.3,
    )

    ax.axvline(
        effective_test_start,
        color="red",
        linestyle="--",
        linewidth=1.3,
    )

    ax.set_title(
        "Pembagian Data Train dan Test"
    )

    ax.set_xlabel(
        "Tanggal"
    )

    ax.set_ylabel(
        f"Harga {commodity_column}"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


st.write("")


st.markdown("### 03 · Konfigurasi Machine Learning")

svr_col, rf_col = st.columns(
    2,
    gap="medium",
)


# ------------------------------------------------------------
# SVR
# ------------------------------------------------------------

with svr_col:

    with st.container(border=True):

        
        st.markdown(
            "#### Cross Validation"
        )

        cv_splits = st.slider(
            "TimeSeriesSplit (Fold)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="cv_split_rf",
        )

        effective_cv = min(
            cv_splits,
            max(
                2,
                len(train_series) // 60,
            ),
        )

        effective_cv = min(
            effective_cv,
            len(train_series) - 1,
        )

        st.metric(
            "Effective Fold",
            effective_cv,
        )

        st.caption(
            "TimeSeriesSplit digunakan agar "
            "proses validasi tetap mengikuti "
            "urutan waktu."
        )


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

with rf_col:

    with st.container(border=True):

        st.markdown(
            "#### Random Forest"
        )

        rf_profile = st.radio(
            "Search Profile",
            [
                "Fast",
                "Balanced",
                "Thorough",
            ],
            index=1,
            horizontal=True,
            key="rf_profile",
        )

        rf_iterations = (
            RF_ITERATIONS_BY_PROFILE[
                rf_profile
            ]
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Search Iteration",
                rf_iterations,
            )

        with c2:

            st.metric(
                "CV Fold",
                effective_cv,
            )

        st.caption(
            "RandomizedSearchCV digunakan "
            "untuk mencari kombinasi hyperparameter "
            "Random Forest terbaik."
        )

st.write("")


# ============================================================
# RUN MODEL
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin:10px 0 14px 0;
    ">
        <div style="
            font-size:13px;
            color:#747784;
            margin-bottom:8px;
        ">
            Semua konfigurasi sudah siap
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


jalankan = st.button(
    "▶  Jalankan Model",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MODEL EXECUTION
# ============================================================

if jalankan:

    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    params = {
        "commodity_column": commodity_column,
        "date_column": date_column,
        "train_end": effective_train_end,
        "test_start": effective_test_start,
        "train_size": len(train_series),
        "test_size": len(test_series),
        "max_lag": MAX_LAG,
        "rf_iterations": rf_iterations,
        "rf_profile": rf_profile,
        "cv_splits": effective_cv,
        "random_state": RANDOM_STATE,
    }

    st.session_state.model_params = params


    # ========================================================
    # PREPARATION
    # ========================================================

    with st.spinner(
        "Mempersiapkan data dan menjalankan "
        "tuning Random Forest..."
    ):

        # ----------------------------------------------------
        # SUPERVISED DATA
        # ----------------------------------------------------

        supervised = (
            make_time_series_features(
                harga,
                max_lag=MAX_LAG,
            )
        )

        X_all = (
            supervised
            .drop(
                columns="target"
            )
        )

        y_all = (
            supervised["target"]
        )

        # ----------------------------------------------------
        # TRAIN TEST MASK
        # ----------------------------------------------------

        train_mask = (
            X_all.index
            <= effective_train_end
        )

        test_mask = (
            (
                X_all.index
                >= effective_test_start
            )
            & (
                X_all.index
                <= END_DATE
            )
        )

        X_train = (
            X_all
            .loc[train_mask]
            .copy()
        )

        y_train = (
            y_all
            .loc[train_mask]
            .copy()
        )

        X_test = (
            X_all
            .loc[test_mask]
            .copy()
        )

        y_test = (
            y_all
            .loc[test_mask]
            .copy()
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            len(X_train) < 100
            or len(X_test) < 10
        ):

            st.error(
                f"Data supervised tidak cukup. "
                f"Train={len(X_train)}, "
                f"Test={len(X_test)}"
            )

            st.stop()


    # ========================================================
    # RANDOM FOREST TUNING
    # ========================================================

    st.markdown(
        "### 🌲 Tuning Random Forest"
    )

    rf_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # RANDOM FOREST SEARCH SPACE
    # --------------------------------------------------------

    rf_search_space = {

        "n_estimators": randint(
            300,
            1001,
        ),

        "max_depth": [
            None,
            5,
            8,
            12,
            16,
            24,
            32,
        ],

        "min_samples_split": randint(
            2,
            16,
        ),

        "min_samples_leaf": randint(
            1,
            10,
        ),

        "max_features": [
            "sqrt",
            "log2",
            0.5,
            0.75,
            1.0,
        ],

        "bootstrap": [
            True
        ],
    }


    # --------------------------------------------------------
    # RANDOMIZED SEARCH
    # --------------------------------------------------------

    rf_search = RandomizedSearchCV(

        estimator=rf_model,

        param_distributions=(
            rf_search_space
        ),

        n_iter=rf_iterations,

        scoring=(
            "neg_root_mean_squared_error"
        ),

        cv=valid_tscv(
            len(X_train),
            effective_cv,
        ),

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=True,

        verbose=0,
    )


    # --------------------------------------------------------
    # FIT RANDOM FOREST
    # --------------------------------------------------------

    with st.spinner(
        "Melakukan tuning Random Forest..."
    ):

        start = time.time()

        rf_search.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time() - start
        ) / 60


    best_rf = (
        rf_search.best_estimator_
    )


    # ========================================================
    # TUNING RESULT
    # ========================================================

    st.success(
        f"Tuning Random Forest selesai "
        f"({elapsed:.2f} menit)"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Best CV RMSE",
            f"{-rf_search.best_score_:.3f}",
        )


    with c2:

        st.metric(
            "Jumlah Iterasi",
            rf_iterations,
        )


    # ========================================================
    # BEST PARAMETERS
    # ========================================================

    with st.expander(
        "Lihat Best Parameter Random Forest",
        expanded=False,
    ):

        best_rf_params = (
            rf_search.best_params_
        )

        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "n_estimators",
                f"{best_rf_params['n_estimators']:,}".replace(
                    ",",
                    ".",
                ),
            )


        with c2:

            max_depth = (
                best_rf_params[
                    "max_depth"
                ]
            )

            st.metric(
                "Max Depth",
                (
                    "Unlimited"
                    if max_depth is None
                    else str(max_depth)
                ),
            )


        with c3:

            st.metric(
                "Max Features",
                str(
                    best_rf_params[
                        "max_features"
                    ]
                ),
            )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Min Samples Split",
                str(
                    best_rf_params[
                        "min_samples_split"
                    ]
                ),
            )


        with c2:

            st.metric(
                "Min Samples Leaf",
                str(
                    best_rf_params[
                        "min_samples_leaf"
                    ]
                ),
            )


        with c3:

            st.metric(
                "CV RMSE",
                f"{-rf_search.best_score_:.3f}",
            )


    # ========================================================
    # SAVE MODEL RESULT
    # ========================================================

    st.session_state.model_result = {

        "best_rf": best_rf,

        "rf_best_params":
            rf_search.best_params_,

        "rf_best_score":
            -rf_search.best_score_,

    }


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.markdown(
        "### 🔎 Hasil Konfigurasi Random Forest"
    )

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Max Lag",
            MAX_LAG,
        )


    with c2:

        st.metric(
            "CV Fold",
            effective_cv,
        )


    with c3:

        st.metric(
            "Train",
            f"{len(X_train):,}".replace(
                ",",
                ".",
            ),
        )


    with c4:

        st.metric(
            "Test",
            f"{len(X_test):,}".replace(
                ",",
                ".",
            ),
        )


    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    st.markdown(
        "#### 📋 Ringkasan Feature Engineering"
    )

    feature_summary = pd.DataFrame(
        {
            "Kategori": [
                "Lag Features",
                "Rolling Mean",
                "Rolling Std",
                "Rolling Min",
                "Rolling Max",
                "Exponential Moving Average",
                "Difference",
                "Calendar Features",
            ],
            "Fitur": [
                "Lag 1–30",
                "Window 3, 5, 7, 10, 14, 21, 30",
                "Window 3, 5, 7, 10, 14, 21, 30",
                "Window 3, 5, 7, 10, 14, 21, 30",
                "Window 3, 5, 7, 10, 14, 21, 30",
                "EWM 5 dan 14",
                "Diff 1 dan Diff 5",
                "Day, Month, Quarter, "
                "Day of Week, Sin/Cos",
            ],
        }
    )

    st.dataframe(
        feature_summary,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # SAVE PREPROCESSING
    # ========================================================

    st.session_state.model_data = {

        "harga": harga,

        "train_series":
            train_series,

        "test_series":
            test_series,

        "X_train":
            X_train,

        "y_train":
            y_train,

        "X_test":
            X_test,

        "y_test":
            y_test,

        # Disimpan sebagai max_lag,
        # bukan optimal lag dari SVR.
        "max_lag":
            MAX_LAG,

        # Compatibility key apabila
        # halaman berikutnya masih
        # membaca optimal_lag.
        "optimal_lag":
            MAX_LAG,

        "effective_train_end":
            effective_train_end,

        "effective_test_start":
            effective_test_start,

        "best_rf":
            best_rf,

        "rf_best_params":
            rf_search.best_params_,

        "rf_best_score":
            -rf_search.best_score_,

    }


    st.success(
        "Konfigurasi dan model Random Forest "
        "berhasil disimpan."
    )