import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy import stats

from utils import (
    clean_commodity_series,
    format_id,
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
    page_title="Analisis Deskriptif",
    page_icon="📊",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / analisis deskriptif",
    title="Analisis Deskriptif",
    caption=(
        "Ringkasan statistik dan visualisasi pola data harga "
        "sebelum pemodelan dilakukan."
    ),
)


# ============================================================
# VALIDASI & PEMBERSIHAN DATA
# ============================================================

df, date_column, commodity_column = require_dataset()

working_df = df[[date_column, commodity_column]].copy()

working_df[date_column] = pd.to_datetime(
    working_df[date_column],
    dayfirst=True,
    errors="coerce",
)

working_df = (
    working_df
    .dropna(subset=[date_column])
    .sort_values(date_column)
)

st.session_state.df = working_df

df[commodity_column] = clean_commodity_series(
    df,
    commodity_column,
)

df = df.dropna(subset=[commodity_column])

harga = df[commodity_column]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_lag_data(
    series,
    n_lags=1,
):
    """
    Membentuk dataset lag harga
    untuk visualisasi hubungan harga t-1 dan harga t.
    """

    frame = pd.DataFrame({
        "y": series
    })

    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)

    return frame.dropna()


def format_rupiah(value):
    """
    Format angka menjadi format Rupiah.
    """

    return f"Rp {format_id(value, 0)}"


# ============================================================
# METRIK RINGKAS
# ============================================================

mean_val = harga.mean()

std_val = harga.std()

skew_val = stats.skew(harga)

kurt_val = stats.kurtosis(
    harga,
    fisher=False,
)


# ============================================================
# RATA-RATA PERUBAHAN BULANAN
# ============================================================

plot_df = (
    df
    .sort_values(date_column)
    .copy()
)

monthly = (
    plot_df
    .set_index(date_column)[commodity_column]
    .resample("MS")
    .mean()
)

monthly_change = (
    monthly
    .pct_change()
    .mean()
    * 100
)

if pd.isna(monthly_change):

    mean_delta = "-"

elif monthly_change > 0:

    mean_delta = (
        f"▲ {monthly_change:.2f}% / bulan"
    )

elif monthly_change < 0:

    mean_delta = (
        f"▼ {abs(monthly_change):.2f}% / bulan"
    )

else:

    mean_delta = "Tidak berubah"


# ============================================================
# VOLATILITAS
# ============================================================

cv = std_val / mean_val

if cv < 0.10:

    volatility_delta = "▼ Rendah"
    volatility_color = "inverse"

elif cv < 0.20:

    volatility_delta = "■ Sedang"
    volatility_color = "off"

else:

    volatility_delta = "▲ Tinggi"
    volatility_color = "normal"


# ============================================================
# SKEWNESS
# ============================================================

if abs(skew_val) < 0.50:

    skew_delta = "● Simetris"
    skew_color = "normal"

elif skew_val > 0:

    skew_delta = "▶ Miring ke kanan"
    skew_color = "inverse"

else:

    skew_delta = "◀ Miring ke kiri"
    skew_color = "off"


# ============================================================
# KURTOSIS
# ============================================================

if kurt_val < 3:

    kurt_delta = "▼ Platykurtic"
    kurt_color = "inverse"

elif kurt_val <= 3.5:

    kurt_delta = "● Mesokurtic"
    kurt_color = "off"

else:

    kurt_delta = "▲ Leptokurtic"
    kurt_color = "normal"


# ============================================================
# SECTION: RINGKASAN
# ============================================================

st.markdown("### Ringkasan Data")

st.caption(
    "Indikator utama yang menggambarkan karakteristik "
    "distribusi dan perubahan harga komoditas."
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Rata-rata Harga",
        format_rupiah(mean_val),
        mean_delta,
        delta_color="off",
    )


with m2:

    st.metric(
        "Volatilitas",
        format_id(std_val, 1),
        volatility_delta,
        delta_color=volatility_color,
    )


with m3:

    st.metric(
        "Skewness",
        f"{skew_val:.2f}",
        skew_delta,
        delta_color=skew_color,
    )


with m4:

    st.metric(
        "Kurtosis",
        f"{kurt_val:.2f}",
        kurt_delta,
        delta_color=kurt_color,
    )


# ============================================================
# SECTION: TREN HARGA
# ============================================================

st.markdown("### 📈 Tren Harga Historis")

st.caption(
    "Pergerakan harga berdasarkan seluruh periode "
    "data yang tersedia."
)

with st.container(border=True):

    plot_df = (
        df
        .sort_values(date_column)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df[date_column],
            y=plot_df[commodity_column],
            mode="lines",
            name=commodity_column,
            line=dict(
                color="#FF4B4B",
                width=1.8,
            ),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.07)",
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Harga: Rp %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        height=340,
        yaxis_title="Rp/kg",
        xaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E5E6EA",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F0F1F4",
        zeroline=False,
        linecolor="#E5E6EA",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SECTION: DISTRIBUSI HARGA
# ============================================================

st.markdown("### 📊 Distribusi Harga")

st.caption(
    "Statistik deskriptif dan bentuk distribusi "
    "harga komoditas."
)

col1, col2 = st.columns(
    2,
    gap="medium",
)


# ============================================================
# STATISTIK DESKRIPTIF
# ============================================================

with col1:

    with st.container(border=True):

        st.markdown(
            "#### Tabel Statistik Deskriptif"
        )

        stat_table = pd.DataFrame(
            {
                "Statistik": [
                    "Mean",
                    "Median",
                    "Std. Deviasi",
                    "Minimum",
                    "Maksimum",
                    "Skewness",
                    "Kurtosis",
                ],
                "Nilai": [
                    f"{mean_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.median():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{std_val:,.1f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.min():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.max():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{skew_val:.2f}",
                    f"{kurt_val:.2f}",
                ],
            }
        )

        st.dataframe(
            stat_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# HISTOGRAM
# ============================================================

with col2:

    with st.container(border=True):

        st.markdown(
            "#### Histogram Harga"
        )

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=df[commodity_column],
                nbinsx=8,
                marker_color="#FF7A6E",
                marker_line_color="#FFFFFF",
                marker_line_width=1,
            )
        )

        fig_hist.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            height=280,
            xaxis_title="Rp/kg",
            yaxis_title="Frekuensi",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        fig_hist.update_xaxes(
            showgrid=False,
            linecolor="#E5E6EA",
        )

        fig_hist.update_yaxes(
            showgrid=True,
            gridcolor="#F0F1F4",
            zeroline=False,
            linecolor="#E5E6EA",
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )


# ============================================================
# SECTION: SCATTER PLOT
# ============================================================

st.markdown("### 📈 Hubungan Harga Lag-1")

st.caption(
    "Hubungan antara harga pada periode t-1 "
    "dan harga pada periode t."
)


# Membentuk data lag-1
lag_data = build_lag_data(
    harga,
    n_lags=1,
)


col1, col2, col3 = st.columns([1, 8, 1])


with col2:

    fig_scatter, ax = plt.subplots(
        figsize=(9, 5),
        dpi=120,
    )

    x = lag_data["lag_1"]
    y = lag_data["y"]

    ax.scatter(
        x,
        y,
        alpha=0.45,
    )

    # Garis tren
    coef = np.polyfit(
        x,
        y,
        1,
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    yy = np.polyval(
        coef,
        xx,
    )

    ax.plot(
        xx,
        yy,
        linewidth=2,
    )

    ax.set_xlabel("Harga t-1")
    ax.set_ylabel("Harga t")

    ax.grid(
        alpha=0.20,
    )

    fig_scatter.tight_layout()

    st.pyplot(
        fig_scatter,
        use_container_width=True,
    )

    plt.close(fig_scatter)