import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    evaluate_prediction,
    format_rupiah,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_trained_model,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Output Model — KomoditasAI",
    page_icon="📈",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / output",
    title="Hasil Forecasting",
    caption=(
        "Evaluasi performa Random Forest, "
        "prediksi harga, dan analisis risiko komoditas."
    ),
)


# ============================================================
# LOAD TRAINING RESULT
# ============================================================

trained, data, params = require_trained_model()


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

best_rf = trained["best_rf"]


# ============================================================
# LOAD DATA
# ============================================================

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

train_series = data["train_series"]
test_series = data["test_series"]

harga = data["harga"]


# ============================================================
# LOAD PARAMETERS
# ============================================================

cv_splits = params.get(
    "cv_splits",
    5,
)

random_state = params.get(
    "random_state",
    42,
)

commodity_name = params.get(
    "commodity_column",
    "Komoditas",
)


# ============================================================
# 1. DATASET & MODEL OVERVIEW
# ============================================================

st.markdown(
    "### 📄 Dataset & Model Overview"
)

with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Komoditas",
            commodity_name,
            "Dataset aktif",
        )

    with c2:

        st.metric(
            "Data Train",
            f"{len(train_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c3:

        st.metric(
            "Data Test",
            f"{len(test_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c4:

        st.metric(
            "Model Forecasting",
            "Random Forest",
            "1 Model",
        )


st.write("")


# ============================================================
# 2. RANDOM FOREST FORECASTING & EVALUATION
# ============================================================

st.markdown(
    "### 2. 📊 Evaluasi Performa Random Forest"
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

with st.spinner(
    "Menghasilkan prediksi Random Forest..."
):

    rf_test_pred = pd.Series(
        best_rf.predict(
            X_test
        ),
        index=X_test.index,
    )


st.success(
    "✓ Prediksi Random Forest berhasil dilakukan."
)


# ============================================================
# EVALUATION METRICS
# ============================================================

rf_metric = evaluate_prediction(
    y_test,
    rf_test_pred,
    y_train,
    "Random Forest",
)


with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "RMSE",
            f"{rf_metric['RMSE']:.2f}",
        )

    with c2:

        st.metric(
            "MAE",
            f"{rf_metric['MAE']:.2f}",
        )

    with c3:

        st.metric(
            "MAPE",
            f"{rf_metric['MAPE (%)']:.2f}%",
        )

    with c4:

        st.metric(
            "R²",
            f"{rf_metric['R2']:.4f}",
        )


st.write("")

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.markdown(
    "#### 📈 Aktual vs Prediksi Random Forest"
)

prediction_result = pd.DataFrame(
    {
        "Actual": y_test,
        "Random Forest": rf_test_pred,
    }
)


with st.container(border=True):

    col1, col2, col3 = st.columns(
        [1, 8, 1]
    )

    with col2:

        fig, ax = plt.subplots(
            figsize=(10, 5),
            dpi=120,
        )

        ax.plot(
            prediction_result.index,
            prediction_result["Actual"],
            linewidth=2.8,
            label="Actual",
        )

        ax.plot(
            prediction_result.index,
            prediction_result["Random Forest"],
            linewidth=2.0,
            label="Random Forest",
        )

        ax.set_xlabel(
            "Tanggal"
        )

        ax.set_ylabel(
            f"Harga {commodity_name}"
        )

        ax.set_title(
            "Aktual vs Prediksi Random Forest"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


# ============================================================
# PREDICTION DETAIL
# ============================================================

with st.expander(
    "📋 Lihat Detail Hasil Prediksi"
):

    detail_prediction = pd.DataFrame(
        {
            "Tanggal": y_test.index,
            "Actual": y_test.values,
            "Prediksi Random Forest":
                rf_test_pred.values,
        }
    )

    detail_prediction[
        "Error"
    ] = (
        detail_prediction["Actual"]
        - detail_prediction[
            "Prediksi Random Forest"
        ]
    )

    detail_prediction[
        "Absolute Error"
    ] = (
        detail_prediction[
            "Error"
        ].abs()
    )

    st.dataframe(
        detail_prediction,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 3. VALUE AT RISK
# ============================================================

st.markdown(
    "### 3. ⚠️ Analisis Risiko"
)


# ============================================================
# ABSOLUTE ERROR
# ============================================================

rf_absolute_error = (
    y_test
    - rf_test_pred
).abs()


# ============================================================
# VAR CALCULATION
# ============================================================

var90 = rf_absolute_error.quantile(
    0.90
)

var95 = rf_absolute_error.quantile(
    0.95
)

var99 = rf_absolute_error.quantile(
    0.99
)


# ============================================================
# VAR HORIZON
# ============================================================

var_table = pd.DataFrame(
    {
        "Hari": range(
            1,
            6,
        )
    }
)


var_table["VaR 90%"] = (
    var90
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 95%"] = (
    var95
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 99%"] = (
    var99
    * np.sqrt(
        var_table["Hari"]
    )
)


# ============================================================
# VAR SUMMARY
# ============================================================

with st.container(border=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "VaR 90% · 1 Hari",
            format_rupiah(
                var90
            ),
        )

    with c2:

        st.metric(
            "VaR 95% · 1 Hari",
            format_rupiah(
                var95
            ),
        )

    with c3:

        st.metric(
            "VaR 99% · 1 Hari",
            format_rupiah(
                var99
            ),
        )


st.write("")


# ============================================================
# VAR BY HORIZON
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### VaR Berdasarkan Horizon"
    )

    var_display = var_table.copy()

    var_display[
        "VaR 90%"
    ] = var_display[
        "VaR 90%"
    ].apply(
        format_rupiah
    )

    var_display[
        "VaR 95%"
    ] = var_display[
        "VaR 95%"
    ].apply(
        format_rupiah
    )

    var_display[
        "VaR 99%"
    ] = var_display[
        "VaR 99%"
    ].apply(
        format_rupiah
    )

    st.dataframe(
        var_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VAR CHART
# ============================================================

st.markdown(
    "#### 📈 Perkembangan Risiko 1–5 Hari"
)

with st.container(border=True):

    col1, col2, col3 = st.columns(
        [1, 8, 1]
    )

    with col2:

        fig, ax = plt.subplots(
            figsize=(9, 5),
            dpi=120,
        )

        ax.plot(
            var_table["Hari"],
            var_table["VaR 90%"],
            marker="o",
            linewidth=2.5,
            label="VaR 90%",
        )

        ax.plot(
            var_table["Hari"],
            var_table["VaR 95%"],
            marker="o",
            linewidth=2.5,
            label="VaR 95%",
        )

        ax.plot(
            var_table["Hari"],
            var_table["VaR 99%"],
            marker="o",
            linewidth=2.5,
            label="VaR 99%",
        )

        ax.set_xlabel(
            "Horizon Risiko (Hari)"
        )

        ax.set_ylabel(
            "Nilai VaR (Rp)"
        )

        ax.set_xticks(
            var_table["Hari"]
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


# ============================================================
# RISK INTERPRETATION
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### 💡 Interpretasi Risiko"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Estimasi VaR 95% · 1 Hari",
            format_rupiah(
                var95
            ),
        )

    with c2:

        st.metric(
            "Estimasi VaR 95% · 5 Hari",
            format_rupiah(
                var_table.iloc[-1][
                    "VaR 95%"
                ]
            ),
        )

    st.write(
        "Estimasi VaR dihitung berdasarkan distribusi "
        "absolute error pada prediksi Random Forest "
        "dan diperluas berdasarkan horizon menggunakan "
        "pendekatan square-root-of-time."
    )