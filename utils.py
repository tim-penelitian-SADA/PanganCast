import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
from pathlib import Path

# ======================================================================
# SESSION STATE
# ======================================================================

# Nilai default seluruh session_state yang dipakai lintas halaman.
SESSION_DEFAULTS = {
    "df": None,
    "original_df": None,
    "dataset_name": None,
    "date_column": None,
    "commodity_column": None,
    "analysis_range": None,
    "model_result": None,
    "model_data": None,
    "model_params": None,
}


def init_session_state():
    """Pastikan seluruh key session_state sudah terdaftar sebelum dipakai halaman mana pun."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ======================================================================
# STYLING (CSS GLOBAL)
# ======================================================================

_CUSTOM_CSS = """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

html{
    font-size:14px;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{
    background:#FFFFFF;
    color:#31333F;
    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


/* ==========================================================
   STREAMLIT HEADER
========================================================== */

header[data-testid="stHeader"]{
    background:#FFFFFF !important;
    border-bottom:1px solid #E6E6E9;
}

[data-testid="stToolbar"]{
    background:#FFFFFF !important;
}


/* ==========================================================
   MAIN CONTAINER
========================================================== */

.block-container{
    max-width:1400px;
    padding:1rem 2rem 1.5rem;
}


/* ==========================================================
   TYPOGRAPHY
========================================================== */

h1{
    font-size:42px !important;
    font-weight:800 !important;
    color:#31333F;
    margin-bottom:.5rem;
}

h2{
    font-size:32px !important;
    font-weight:700 !important;
    color:#31333F;
    margin-top:1.4rem;
    margin-bottom:.6rem;
}

h3{
    font-size:24px !important;
    font-weight:700 !important;
    color:#31333F;
}

h4{
    font-size:18px !important;
    font-weight:600 !important;
    color:#31333F;
}


/* ==========================================================
   HOMEPAGE HERO
========================================================== */

.homepage-hero{
    position:relative;
    overflow:hidden;

    min-height:690px;

    background:
        radial-gradient(
            circle at 78% 22%,
            rgba(46,173,104,.13),
            transparent 25%
        ),
        radial-gradient(
            circle at 12% 88%,
            rgba(0,140,149,.07),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #FFFFFF 0%,
            #FBFEFC 52%,
            #F5FBF7 100%
        );

    border:1px solid #DDEBE2;
    border-radius:28px;

    padding:65px 70px 45px;

    box-shadow:
        0 8px 30px rgba(38,75,52,.045);
}


/* ==========================================================
   DECORATIVE GRID
========================================================== */

.homepage-hero::before{
    content:"";

    position:absolute;

    top:45px;
    left:45px;

    width:150px;
    height:120px;

    opacity:.55;

    background-image:
        radial-gradient(
            #8DD9A9 1.6px,
            transparent 1.6px
        );

    background-size:24px 24px;

    mask-image:
        linear-gradient(
            135deg,
            black 0%,
            transparent 90%
        );

    -webkit-mask-image:
        linear-gradient(
            135deg,
            black 0%,
            transparent 90%
        );
}


/* ==========================================================
   RIGHT GREEN GLOW
========================================================== */

.homepage-hero::after{
    content:"";

    position:absolute;

    width:430px;
    height:430px;

    right:-170px;
    top:110px;

    border-radius:50%;

    background:
        radial-gradient(
            circle,
            rgba(46,173,104,.13) 0%,
            rgba(46,173,104,.055) 38%,
            transparent 72%
        );

    pointer-events:none;
}


/* ==========================================================
   HERO CONTENT
========================================================== */

.hero-content{
    position:relative;
    z-index:5;

    text-align:center;

    max-width:1100px;

    margin:0 auto;
}


/* ==========================================================
   HERO LOGO
========================================================== */

.hero-logo{
    position:relative;

    width:155px;
    height:155px;

    margin:0 auto 22px;

    display:flex;
    align-items:center;
    justify-content:center;

    background:transparent;

    border:none;
    border-radius:0;

    overflow:visible;

    z-index:5;
}


/* glow di belakang logo */

.hero-logo::before{
    content:"";

    position:absolute;

    width:125px;
    height:125px;

    border-radius:50%;

    background:
        radial-gradient(
            circle,
            rgba(46,173,104,.16),
            rgba(46,173,104,.05) 45%,
            transparent 72%
        );

    filter:blur(10px);

    z-index:-1;
}


/* logo */

.hero-logo-image{
    width:155px;
    height:155px;

    object-fit:contain;

    display:block;

    background:transparent;

    filter:
        drop-shadow(
            0 12px 18px
            rgba(46,173,104,.14)
        );
}



/* ==========================================================
   KICKER
========================================================== */

.hero-kicker{
    position:relative;

    display:inline-flex;

    align-items:center;
    justify-content:center;

    padding:8px 17px;

    margin-bottom:22px;

    border-radius:30px;

    background:
        linear-gradient(
            135deg,
            #F0FBF4,
            #E7F8ED
        );

    border:1px solid #D2ECDD;

    color:#249B5B;

    font-size:12px;
    font-weight:750;

    letter-spacing:.8px;

    text-transform:uppercase;

    box-shadow:
        0 4px 12px
        rgba(46,173,104,.05);
}


/* ==========================================================
   HERO TITLE
========================================================== */

.hero-title{
    font-size:43px;

    font-weight:800;

    line-height:1.18;

    letter-spacing:-1.4px;

    color:#26352D;

    margin:0 auto;

    max-width:1000px;
}


/* ==========================================================
   HERO ACCENT
========================================================== */

.hero-accent{
    width:95px;
    height:5px;

    border-radius:10px;

    background:
        linear-gradient(
            90deg,
            #008C95 0%,
            #2EAD68 50%,
            #E7F8ED 100%
        );

    margin:28px auto 26px;

    box-shadow:
        0 4px 10px
        rgba(46,173,104,.15);
}


/* ==========================================================
   HERO AUTHORS
========================================================== */

.hero-authors{
    display:flex;

    align-items:center;
    justify-content:center;

    gap:12px;

    font-size:18px;

    font-weight:600;

    color:#4B5A51;
}


/* ==========================================================
   AUTHOR ICON
========================================================== */

.hero-author-icon{
    width:40px;
    height:40px;

    border-radius:50%;

    display:flex;
    align-items:center;
    justify-content:center;

    background:
        linear-gradient(
            135deg,
            #EAF8EF,
            #DDF4E6
        );

    border:1px solid #CDE9D7;

    color:#238D53;

    font-size:21px;

    box-shadow:
        0 5px 14px
        rgba(46,173,104,.10);
}


/* ==========================================================
   EXTRA DECORATIVE ELEMENT
========================================================== */

.homepage-hero .hero-content::after{
    content:"";

    position:absolute;

    width:8px;
    height:8px;

    right:-130px;
    top:120px;

    border-radius:50%;

    background:#D9A441;

    box-shadow:
        26px 45px 0 #2EAD68,
        -35px 80px 0 #008C95,
        55px 105px 0 #8DD9A9;

    opacity:.55;
}


/* ==========================================================
   BOTTOM DECORATIVE LINE
========================================================== */

.homepage-hero{
    isolation:isolate;
}

.homepage-hero .hero-content{
    padding-bottom:25px;
}

.homepage-hero .hero-content::marker{
    display:none;
}


/* ==========================================================
   SMALL SCREEN
========================================================== */

@media(max-width:900px){

    .homepage-hero{
        padding:50px 30px 35px;
        min-height:620px;
    }

    .hero-logo{
        width:110px;
        height:110px;
    }

    .hero-logo-image{
        width:110px;
        height:110px;
    }

    .hero-title{
        font-size:31px;
    }

    .hero-authors{
        font-size:13px;
    }

    .hero-content::before,
    .hero-content::after{
        display:none;
    }
}

/* ==========================================================
   HERO AUTHOR ICON
========================================================== */

.hero-author-icon{
    width:38px;
    height:38px;

    border-radius:50%;

    background:
        linear-gradient(
            135deg,
            #F0FAF2,
            #F0FAF2
        );

    color:#FFFFFF;

    display:flex;

    align-items:center;

    justify-content:center;

    font-size:23px;

    border:1px solid #DCEFE0;

    box-shadow:
        0 5px 12px rgba(46,155,97,.15);
}


/* ==========================================================
   DECORATIVE BOTTOM WAVE
========================================================== */

.hero-wave{
    position:absolute;

    left:-5%;
    bottom:-5px;

    width:110%;
    height:150px;

    z-index:1;

    opacity:.9;
}

.hero-wave svg{
    width:100%;
    height:100%;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{
    width:245px !important;

    background:#F6F8F7;

    border-right:1px solid #E2E8E4;
}

section[data-testid="stSidebar"] > div{
    background:#F6F8F7;
}

section[data-testid="stSidebarContent"]{
    padding:18px 16px;
}


/* Hide default Streamlit multipage navigation */

[data-testid="stSidebarNav"]{
    display:none !important;
}

[data-testid="stSidebarNavItems"]{
    display:none !important;
}


/* ==========================================================
   SIDEBAR BRAND
========================================================== */

.sidebar-brand{
    display:flex;

    align-items:center;

    gap:11px;

    margin-bottom:28px;
}


.sidebar-brand-text{
    line-height:1.25;
}


.sidebar-brand-title{
    font-size:15px;

    font-weight:750;

    color:#29352F;
}


.sidebar-brand-subtitle{
    font-size:11px;

    color:#737C76;

    margin-top:3px;
}


/* ==========================================================
   SIDEBAR BRAND LOGO
========================================================== */

.sidebar-brand-logo{
    width:52px;
    height:52px;

    border-radius:14px;

    display:flex;

    align-items:center;

    justify-content:center;

    background:
        linear-gradient(
            135deg,
            #2E9B61,
            #4DBB6A
        );

    color:#FFFFFF;

    font-size:22px;

    font-weight:800;

    box-shadow:
        0 8px 18px rgba(46,155,97,.15);
}


/* ==========================================================
   SIDEBAR TITLE
========================================================== */

.sidebar-title{
    font-size:15px;

    font-weight:700;

    color:#29352F;

    margin:0 0 10px 2px;
}


/* ==========================================================
   PAGE LINK
========================================================== */

div[data-testid="stPageLink"]{
    margin-bottom:4px;
}

div[data-testid="stPageLink"] a{
    display:flex;

    align-items:center;

    min-height:40px;

    padding:8px 12px;

    border-radius:11px;

    font-size:13px;

    font-weight:500;

    color:#424D47;

    transition:
        background .18s ease,
        color .18s ease;
}


div[data-testid="stPageLink"] a:hover{
    background:#E8F3EA;

    color:#2E9B61;
}


div[data-testid="stPageLink"][aria-current="page"] a{
    background:#E3F1E6;

    color:#2E9B61;

    font-weight:650;
}


/* Bullet */

div[data-testid="stPageLink"] a::before{
    content:"•";

    color:#9BA8A0;

    margin-right:9px;

    font-size:11px;
}


div[data-testid="stPageLink"][aria-current="page"] a::before{
    color:#2E9B61;
}


/* ==========================================================
   SIDEBAR DIVIDER
========================================================== */

.sidebar-divider{
    border-top:1px solid #DDE5E0;

    margin:20px 0 14px;
}


/* ==========================================================
   SIDEBAR FOOTER
========================================================== */

section[data-testid="stSidebar"] .stCaption{
    font-size:11px !important;

    color:#858F89 !important;
}


/* ==========================================================
   PAGE HEADER
========================================================== */

.page-breadcrumb{
    font-size:11px;

    color:#929A95;

    margin-bottom:5px;
}


.page-title{
    font-size:32px;

    font-weight:800;

    letter-spacing:-.5px;

    color:#29352F;

    margin:0;
}


.page-caption{
    font-size:13px;

    color:#747C77;

    margin-top:5px;
}


/* ==========================================================
   CARD
========================================================== */

.card{
    background:#FFFFFF;

    border:1px solid #E1E7E3;

    border-radius:16px;

    padding:22px;

    box-shadow:
        0 3px 12px rgba(46,155,97,.025);
}


.stMarkdown .card h3{
    font-size:18px !important;

    font-weight:700 !important;

    margin:0 0 10px !important;

    color:#29352F !important;
}


.stMarkdown .card p{
    font-size:13px !important;

    line-height:1.7 !important;

    color:#626D67 !important;
}


/* ==========================================================
   STREAMLIT CONTAINER
========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid #E1E7E3;

    border-radius:16px;

    background:#FFFFFF;
}


/* ==========================================================
   INFO / WARNING / ERROR / SUCCESS
========================================================== */

div[data-testid="stInfo"],
div[data-testid="stWarning"],
div[data-testid="stError"],
div[data-testid="stSuccess"]{

    border-radius:12px;

    border:1px solid #E1E7E3;

    padding:.75rem 1rem;

    font-size:13px;
}


/* ==========================================================
   METRIC
========================================================== */

[data-testid="stMetric"]{
    background:#FFFFFF;

    border:1px solid #E1E7E3;

    border-radius:14px;

    padding:16px;
}


[data-testid="stMetricLabel"]{
    font-size:12px !important;

    color:#777F7A !important;
}


[data-testid="stMetricValue"]{
    font-size:25px !important;

    font-weight:750 !important;

    color:#29352F !important;
}


[data-testid="stMetricDelta"]{
    font-size:12px !important;
}


[data-testid="stMetricDelta"] svg{
    display:none;
}


/* ==========================================================
   BUTTON
========================================================== */

.stButton button{
    border-radius:10px;

    border:1px solid #DCE5DF;

    font-size:13px;

    font-weight:600;

    padding:.48rem 1rem;

    transition:
        .18s ease;
}


.stButton button:hover{
    border-color:#4DBB6A;

    color:#2E9B61;

    background:#F0FAF2;
}


/* ==========================================================
   PRIMARY BUTTON
========================================================== */

.stButton button[kind="primary"]{
    background:#2E9B61;

    border-color:#2E9B61;

    color:#FFFFFF;
}


.stButton button[kind="primary"]:hover{
    background:#247F4F;

    border-color:#247F4F;

    color:#FFFFFF;
}


/* ==========================================================
   INPUT
========================================================== */

.stTextInput label,
.stSelectbox label,
.stNumberInput label,
.stDateInput label,
.stRadio label,
.stCheckbox label{
    font-size:13px;

    font-weight:600;

    color:#454F49;
}


.stTextInput input,
.stNumberInput input{
    font-size:13px;

    border-radius:9px;
}


.stSelectbox div[data-baseweb="select"]{
    font-size:13px;
}


/* ==========================================================
   INPUT FOCUS
========================================================== */

.stTextInput input:focus,
.stNumberInput input:focus{

    border-color:#4DBB6A !important;

    box-shadow:
        0 0 0 1px #4DBB6A !important;
}


/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"]{
    font-size:13px;
}


/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"]{
    font-size:13px;

    padding:9px 16px;
}


button[data-baseweb="tab"][aria-selected="true"]{
    color:#2E9B61 !important;
}


/* ==========================================================
   SLIDER
========================================================== */

div[data-baseweb="slider"] div[role="slider"]{
    background:#2E9B61 !important;
}


div[data-baseweb="slider"] > div > div{
    background:#DCEFE0 !important;
}


/* ==========================================================
   CHECKBOX
========================================================== */

[data-testid="stCheckbox"] label{
    color:#454F49;
}


/* ==========================================================
   RADIO
========================================================== */

div[role="radiogroup"] label{
    color:#454F49;
}


/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{
    width:6px;
    height:6px;
}


::-webkit-scrollbar-track{
    background:transparent;
}


::-webkit-scrollbar-thumb{
    background:#D5DED8;

    border-radius:10px;
}


::-webkit-scrollbar-thumb:hover{
    background:#B8C9BD;
}


/* ==========================================================
   HIDE STREAMLIT COMMUNITY CLOUD BOTTOM-RIGHT PROFILE
========================================================== */

/* Target utama tombol profile */

[data-testid="stDecoration"],
button[data-testid="baseButton-stDecoration"],
button[aria-label="View profile"],
div[data-testid="stDecoration"],
span[data-testid="stDecoration"],
a[data-testid="stDecoration"],
button[data-testid="stDecoration"]{

    display:none !important;

    visibility:hidden !important;

    opacity:0 !important;

    pointer-events:none !important;
}


/* Fallback container */

div.css-1cpxqw2,
div.css-1v0mbdj,
div[data-testid="stBottomRightContainer"]{

    display:none !important;
}


/* Fallback tambahan */

body > div[data-testid="stDecoration"]{

    display:none !important;
}


/* Target seluruh elemen fixed di area kanan bawah */

div[data-testid="stDecoration"] button,
div[data-testid="stDecoration"] a,
div[data-testid="stDecoration"] span{

    display:none !important;

    visibility:hidden !important;

    opacity:0 !important;

    pointer-events:none !important;
}


/* ==========================================================
   SMALL SCREEN
========================================================== */

@media(max-width:900px){

    .homepage-hero{

        padding:50px 30px 35px;

        min-height:620px;
    }


    .hero-title{

        font-size:31px;
    }


    .hero-logo{

        width:95px;
        height:95px;

        border-radius:24px;
    }


    .hero-authors{

        font-size:13px;

        flex-wrap:wrap;
    }


    .sidebar-brand-title{

        font-size:14px;
    }

}

</style>
"""

# path halaman & label navigasi (urutan sesuai konsep tampilan awal)
NAV_ITEMS = [
    ("homepage.py", "Homepage"),
    ("pages/input_data.py", "Input Dataset"),
    ("pages/analisis_desk.py", "Analisis Deskriptif"),
    ("pages/input_params.py", "Input Parameter"),
    ("pages/output.py", "Output"),
]


def inject_custom_css():
    """Suntikkan CSS global (card, sidebar, page link, info box, container)."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar PanganCast."""

    with st.sidebar:

        # ======================================================
        # LOGO
        # ======================================================
        logo_path = Path("logo.png")

        logo_base64 = base64.b64encode(
            logo_path.read_bytes()
        ).decode() 

        st.html(f"""
        <div style="
            display:flex;
            align-items:center;
            gap:10px;
            margin-bottom:28px;
        ">

            <div style="
                width:48px;
                height:48px;
                flex-shrink:0;
                display:flex;
                justify-content:center;
                align-items:center;
                background:transparent;
            ">
                <img
                    src="data:image/png;base64,{logo_base64}"
                    style="
                        width:48px;
                        height:48px;
                        object-fit:contain;
                        display:block;
                        background:transparent;
                    "
                >
            </div>

            <div style="
                line-height:1.25;
            ">

                <div style="
                    font-size:15px;
                    font-weight:750;
                    color:#31333F;
                ">
                    PanganCast
                </div>

                <div style="
                    font-size:11px;
                    color:#737687;
                    margin-top:3px;
                ">
                    Prediksi Harga Pangan Akurat
                </div>

            </div>

        </div>
        """)

        # ======================================================
        # NAVIGATION
        # ======================================================

        st.markdown(
            '<div class="sidebar-title">Navigasi</div>',
            unsafe_allow_html=True,
        )

        for path, label in NAV_ITEMS:
            st.page_link(path, label=label)

        # ======================================================
        # FOOTER
        # ======================================================

        st.markdown(
            '<div class="sidebar-divider"></div>',
            unsafe_allow_html=True,
        )

        st.caption("© 2026 • PanganCast Dashboard")

def page_header(breadcrumb: str, title: str, caption: str = ""):
    """Header konsisten untuk tiap halaman: breadcrumb, judul, dan sub-judul."""
    st.markdown(f"`{breadcrumb}`")
    st.title(title)
    if caption:
        st.caption(caption)


def setup_page(page_title: str, page_icon: str, breadcrumb: str, title: str, caption: str = ""):
    """
    Satu pemanggilan untuk seluruh boilerplate awal sebuah halaman:
    page_config -> session_state -> CSS -> sidebar -> header.
    Dipanggil paling atas, tepat setelah import.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    init_session_state()
    inject_custom_css()
    render_sidebar()
    page_header(breadcrumb, title, caption)


# ======================================================================
# GUARD / VALIDASI ALUR HALAMAN
# ======================================================================

def require_dataset():
    """
    Pastikan dataset & pemetaan kolom (tanggal, harga) sudah tersedia.
    Menghentikan halaman dengan pesan yang konsisten jika belum siap.
    """
    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("Silakan unggah dan pilih dataset terlebih dahulu pada halaman **Input Dataset**.")
        st.stop()
    if st.session_state.get("date_column") is None:
        st.error("Kolom tanggal belum ditentukan pada halaman Input Dataset.")
        st.stop()
    if st.session_state.get("commodity_column") is None:
        st.error("Kolom harga belum ditentukan pada halaman Input Dataset.")
        st.stop()

    return (
        st.session_state.df.copy(),
        st.session_state.date_column,
        st.session_state.commodity_column,
    )


def require_trained_model():
    """Pastikan proses training (Input Parameter) sudah pernah dijalankan."""
    if st.session_state.get("model_result") is None:
        st.warning("Silakan jalankan proses training terlebih dahulu pada halaman **Input Parameter**.")
        st.stop()

    return (
        st.session_state.model_result,
        st.session_state.model_data,
        st.session_state.model_params,
    )


# ======================================================================
# PEMBERSIHAN DATA HARGA KOMODITAS
# ======================================================================

def clean_commodity_series(df: pd.DataFrame, commodity_column: str) -> pd.Series:
    """
    Bersihkan kolom harga komoditas dari format Rupiah (mis. "Rp12.345,67")
    menjadi nilai numerik (float), dan ubah placeholder ("-", "nan", dst) menjadi NaN.
    """
    cleaned = (
        df[commodity_column]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace(["-", "", "nan", "None"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ======================================================================
# FORMATTING ANGKA (GAYA INDONESIA)
# ======================================================================

def format_id(value, decimal: int = 0) -> str:
    """Format angka dengan pemisah ribuan '.' dan desimal ',' (gaya Indonesia)."""
    return f"{value:,.{decimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(value) -> str:
    """Format angka menjadi teks Rupiah, mis. 12345.6 -> 'Rp12.345,6'."""
    text = format_id(value, decimal=2).rstrip("0").rstrip(",")
    return f"Rp{text}"


# ======================================================================
# METRIK EVALUASI MODEL (dipakai di Input Parameter & Output)
# ======================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_prediction(y_true, y_pred, insample, model_name: str) -> dict:
    """Ringkasan metrik evaluasi (RMSE, MAE, MAPE, sMAPE, MASE, R2, Bias) untuk satu model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape_safe(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }