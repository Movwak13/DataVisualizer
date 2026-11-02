from genre_service import enrich_artists_with_spotify
import io, json, logging, os, re, sys, traceback, gc
from datetime import datetime, date
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import psutil

# ---------------------------------------------------------
# STATE & CACHE INITIALIZATION
# ---------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "nav_mode" not in st.session_state:
    st.session_state.nav_mode = "Visualize My Data"

def purge_and_reset_uploads():
    if not st.session_state.get("legal_acknowledgment", False):
        st.session_state.uploader_key += 1
        keys_to_clear = [k for k in list(st.session_state.keys()) if k not in ["uploader_key", "legal_acknowledgment", "nav_mode"]]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        gc.collect()

# ---------------------------------------------------------
# LOGGING & ENVIRONMENT INITIALIZATION
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "configuration.config")
LOG_DIR = os.path.join(BASE_DIR, "ErrorLogs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app_errors.log")

logger = logging.getLogger("DataVisualizerLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def log_error(context: str, exc: Exception, user_msg: str):
    logger.error(f"Error in {context}: {type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
    st.sidebar.error(f"⚠️ {user_msg}")

# ---------------------------------------------------------
# SENSITIVE PII & IP SANITIZATION ENGINE
# ---------------------------------------------------------
IPV4_REGEX = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
IPV6_REGEX = r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'

SENSITIVE_KEY_PATTERNS = [
    r'(^|_)ip(_addr|_address)?$', r'client_ip', r'remote_addr', r'ip_addr_decrypted',
    r'(^|_)user(_?id|_name|_account)?$', r'customer_id', r'listener_id', r'store_account_id',
    r'device(_?id|_identifier|_serial|_number)?$', r'hardware_id', r'build_version', r'user_agent',
    r'spotify_(track|episode|audiobook)_uri', r'audiobook_chapter_uri',
    r'session_id', r'token', r'location_zip', r'postal_code', r'latitude', r'longitude'
]

IP_SUSPECT_COLS = {'ip_addr', 'client_ip', 'ip', 'remote_addr', 'user_agent', 'user_agent_decrypted'}

def clean_raw_json_obj(item):
    if isinstance(item, list):
        return [clean_raw_json_obj(x) for x in item]
    elif isinstance(item, dict):
        cleaned = {}
        for k, v in item.items():
            k_lower = str(k).lower().strip()
            if any(re.search(pat, k_lower) for pat in SENSITIVE_KEY_PATTERNS):
                continue
            if isinstance(v, (dict, list)):
                cleaned[k] = clean_raw_json_obj(v)
            else:
                cleaned[k] = v
        return cleaned
    return item

def strip_ip_addresses(dataframe: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        c for c in dataframe.columns 
        if any(re.search(pat, str(c).lower().strip()) for pat in SENSITIVE_KEY_PATTERNS)
    ]
    if cols_to_drop:
        dataframe = dataframe.drop(columns=cols_to_drop, errors='ignore')

    target_text_cols = [c for c in dataframe.columns if str(c).lower().strip() in IP_SUSPECT_COLS]
    for col in target_text_cols:
        dataframe[col] = (
            dataframe[col]
            .astype(str)
            .str.replace(IPV4_REGEX, '[REDACTED_IP]', regex=True)
            .str.replace(IPV6_REGEX, '[REDACTED_IP]', regex=True)
        )
    return dataframe

def get_container_ram_stats():
    process = psutil.Process(os.getpid())
    process_mb = process.memory_info().rss / (1024 * 1024)
    cgroup_limit_file = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    cgroup_usage_file = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    
    container_mb = process_mb
    max_mb = 1024.0
    
    if os.path.exists(cgroup_usage_file) and os.path.exists(cgroup_limit_file):
        try:
            with open(cgroup_usage_file, "r") as f:
                container_mb = int(f.read().strip()) / (1024 * 1024)
            with open(cgroup_limit_file, "r") as f:
                raw_limit = int(f.read().strip())
                if raw_limit < (100 * 1024 * 1024 * 1024):
                    max_mb = raw_limit / (1024 * 1024)
        except Exception:
            container_mb = process_mb

    return {
        "process_mb": process_mb,
        "container_mb": container_mb,
        "max_mb": max_mb,
        "pct_used": min(100.0, (container_mb / max_mb) * 100.0)
    }

@st.cache_data(ttl=3600)
def load_config():
    defaults = {
        "profiles": {
            "Spotify": {"is_music": True},
            "Apple Music": {"is_music": True},
            "Pandora": {"is_music": True},
            "YouTube Music": {"is_music": True},
            "Amazon Music": {"is_music": True},
            "Other": {"is_music": False}
        },
        "styling": {
            "default_palette": "Viridis",
            "palettes": ["Viridis", "Plasma", "Turbo", "Sunsetdark", "Tealgrn", "Bluered", "Cividis"],
            "default_export_theme": "Dark Canvas"
        },
        "limits": {
            "max_music_catalog_items": 500, 
            "max_export_width": 1600, 
            "row_pixel_height": 26,
            "max_file_size_mb": 50,
            "max_total_rows": 500000
        },
        "ignored_columns": [
            "spotify_track_uri", "spotify_episode_uri", "audiobook_uri", "audiobook_chapter_uri",
            "ip_addr", "user_agent_decrypted", "username", "offline_timestamp", "incognito_mode",
            "offline", "skipped"
        ]
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse config: {str(e)}")
    return defaults

CONFIG = load_config()
DATASET_PROFILES = CONFIG.get("profiles", {})
STYLING_CFG = CONFIG.get("styling", {})
LIMITS_CFG = CONFIG.get("limits", {})
IGNORED_COLS = set(CONFIG.get("ignored_columns", []))
MAX_FILE_BYTES = LIMITS_CFG.get("max_file_size_mb", 50) * 1024 * 1024
MAX_TOTAL_ROWS = LIMITS_CFG.get("max_total_rows", 500000)

st.set_page_config(
    page_title="Universal Multi-Select Data Visualizer", 
    page_icon="📊", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ACCESSIBLE (WCAG 2.2 AA) CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .skip-to-main {
        position: absolute;
        top: -100px;
        left: 10px;
        background: #EF4444;
        color: #FFFFFF !important;
        padding: 8px 16px;
        z-index: 100000;
        font-weight: 700;
        border-radius: 4px;
        text-decoration: none;
        transition: top 0.2s ease;
    }
    .skip-to-main:focus { top: 10px; }
    .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    .app-title-wrapper {
        margin: 0.25rem 0 1.25rem 0;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .app-title-text {
        font-size: clamp(1.35rem, 2.2vw + 0.5rem, 2.1rem);
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0;
        padding: 0;
    }
    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 6px !important;
        padding: 8px 10px 0px 10px !important;
        border-radius: 8px 8px 0px 0px !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.12) !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 8px 8px 0px 0px !important;
        padding: clamp(6px, 1.2vw, 10px) clamp(12px, 1.8vw, 22px) !important;
        font-size: clamp(12px, 1vw + 2px, 14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom: none !important;
        background-color: transparent !important;
        min-height: 42px !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-bottom: 3px solid #EF4444 !important;
        font-weight: 600 !important;
    }
    .sidebar-nav-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .legal-card {
        background-color: #2D1515;
        border: 1px solid #EF4444;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 12px;
        font-size: 12px;
        line-height: 1.45;
        color: #FEE2E2;
    }
</style>
<a href="#main-dashboard-content" class="skip-to-main">Skip to main content</a>
<div class="app-title-wrapper" role="banner">
    <h1 class="app-title-text">📊 Universal Multi-Select Data Visualizer</h1>
</div>
<div id="main-dashboard-content" role="main"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-nav-label'>🧭 Navigation</div>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.sidebar.columns(2)
is_vis_active = (st.session_state.nav_mode == "Visualize My Data")

with col_nav1:
    if st.button("📊 Visualizer", use_container_width=True, type="primary" if is_vis_active else "secondary"):
        st.session_state.nav_mode = "Visualize My Data"
        st.rerun()

with col_nav2:
    if st.button("💡 Guide", use_container_width=True, type="primary" if not is_vis_active else "secondary"):
        st.session_state.nav_mode = "What Can I Do Here?"
        st.rerun()

nav_mode = st.session_state.nav_mode
st.sidebar.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar.expander("📁 Dashboard Controls", expanded=False):
    enable_date_filtering = st.toggle("Enable Date Range Filter", value=True, help="Constrain charts to specific dates.")
    show_bar_labels = st.toggle("Show Bar Value Labels", value=True, help="Display exact totals at the tip of each bar.")
    min_play_sec = st.slider("Min Play Duration (Seconds):", min_value=0, max_value=60, value=30, step=5, help="Filter out quick skips under this threshold.")
    exclude_unknowns = st.toggle("Exclude Unknown Metadata", value=True, help="Hide blank, unknown, or missing records.")
    show_gridlines = st.toggle("Show Chart Gridlines", value=True, help="Display background grid lines on charts.")
    ranking_direction = 'Highest First (Top N)'

with st.sidebar.expander("🎨 Stylings & Colors", expanded=False):
    available_palettes = STYLING_CFG.get("palettes", ["Viridis", "Plasma", "Turbo"])
    selected_palette = st.selectbox("Color Palette:", available_palettes, index=0)
    export_theme = st.selectbox("PNG Export Background:", ["Dark Canvas", "Light Canvas"], index=0)

chart_bg = "#111827" if export_theme == "Dark Canvas" else "#FFFFFF"
chart_text = "#E5E7EB" if export_theme == "Dark Canvas" else "#111827"
chart_grid = "#1F2937" if export_theme == "Dark Canvas" else "#E5E7EB"

# ---------------------------------------------------------
# EXPORT & UTILITY HELPERS
# ---------------------------------------------------------
def render_export_bar(fig, table_df, base_name, total_rows=None):
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            "📥 Download CSV", 
            table_df.to_csv(index=False).encode('utf-8'), 
            f"{base_name}.csv", 
            "text/csv", 
            key=f"csv_{base_name}"
        )
    with c2:
        try:
            st.download_button(
                "🖼️ Download PNG", 
                fig.to_image(format="png", width=1200, height=700, scale=2), 
                f"{base_name}.png", 
                "image/png", 
                key=f"png_{base_name}"
            )
        except Exception as e:
            logger.warning(f"PNG render unavailable: {str(e)}")

def apply_date_range_filter(data_df, key_prefix):
    dt_c = next((c for c in data_df.columns if c.endswith('_dt')), None)
    if not dt_c or not enable_date_filtering:
        return data_df
    
    min_d = data_df[dt_c].min().date()
    max_d = data_df[dt_c].max().date()
    if min_d == max_d:
        return data_df
    
    s_d, e_d = st.slider(
        "📅 Date Filter:", 
        min_value=min_d, 
        max_value=max_d, 
        value=(min_d, max_d), 
        format="YYYY-MM-DD", 
        key=f"{key_prefix}_d_slider"
    )
    return data_df[(data_df['Date_Only'] >= s_d) & (data_df['Date_Only'] <= e_d)]

# ---------------------------------------------------------
# DATA INGESTION ENGINE
# ---------------------------------------------------------
with st.sidebar.expander("💾 Data & Platform", expanded=True):
    st.markdown("""
    <div class="legal-card" role="region" aria-label="Privacy Notice">
        <strong>⚠️ NOTICE & PRIVACY DISCLAIMER:</strong><br>
        Do <strong>NOT</strong> upload sensitive, proprietary, confidential, or legally protected data.
        Processed in temporary container memory only.
    </div>
    """, unsafe_allow_html=True)

    terms_accepted = st.checkbox(
        "I certify that my files contain NO sensitive data, PHI, or confidential info.",
        value=False,
        key="legal_acknowledgment",
        on_change=purge_and_reset_uploads
    )

    if not terms_accepted:
        st.info("🔒 Ingestion disabled. Acknowledge the disclaimer above to upload files.")
        uploaded = None
        gsheet = ""
    else:
        uploaded = st.file_uploader(
            "Upload Files (CSV, Excel, JSON)", 
            type=["csv", "xlsx", "xls", "json"], 
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )
        gsheet = st.text_input("Public Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")

        if st.button("🗑️ Reset Session & Clear Data", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k not in ["uploader_key", "legal_acknowledgment", "nav_mode"]:
                    st.session_state.pop(k, None)
            st.session_state.pop("cached_df", None)
            st.session_state.pop("last_upload_signature", None)
            gc.collect()
            st.rerun()

    def validate_gsheet_url(url: str) -> bool:
        if not url: return False
        parsed = urlparse(url.strip())
        return parsed.scheme == "https" and parsed.netloc == "docs.google.com" and bool(re.match(r"^/spreadsheets/d/([a-zA-Z0-9-_]+)", parsed.path))

    def inspect_header_signature(uploaded_files, gsheet_url):
        cols_found = set()
        if uploaded_files:
            first_file = uploaded_files[0]
            try:
                first_file.seek(0)
                fn = first_file.name.lower()
                if fn.endswith('.csv'): cols_found.update(pd.read_csv(first_file, nrows=2).columns)
                elif fn.endswith(('.xlsx', '.xls')): cols_found.update(pd.read_excel(first_file, nrows=2).columns)
                elif fn.endswith('.json'):
                    content_json = json.load(first_file)
                    df_sample = pd.json_normalize(content_json[:2] if isinstance(content_json, list) else [content_json])
                    cols_found.update(df_sample.columns)
                first_file.seek(0)
            except Exception: pass

        cols_lower = {str(c).lower().strip() for c in cols_found}
        if any(k in cols_lower for k in ['master_metadata_track_name', 'msplayed', 'ms_played', 'conn_country']): return "Spotify"
        if any(k in cols_lower for k in ['event start timestamp', 'play duration milliseconds', 'content name']): return "Apple Music"
        if any(k in cols_lower for k in ['station_name', 'station_history', 'track_title']): return "Pandora"
        if any(k in cols_lower for k in ['titleurl', 'subtitles']): return "YouTube Music"
        if any(k in cols_lower for k in ['trackname', 'songtitle']) or any(k in cols_lower for k in ['playdate', 'eventtimestamp']): return "Amazon Music"
        return "Other"

    detected_platform = inspect_header_signature(uploaded, gsheet) if terms_accepted else "Other"
    profile_names = list(DATASET_PROFILES.keys())
    active_idx = profile_names.index(detected_platform) if detected_platform in profile_names else profile_names.index("Other")

    selected_profile = st.selectbox("Dataset Platform:", profile_names, index=active_idx)

# ---------------------------------------------------------
# SANITIZATION & SAFE EXPORT
# ---------------------------------------------------------
def sanitize_csv_data(df_in: pd.DataFrame) -> bytes:
    injection_pattern = r"^\s*[-=+@\t\r]"
    str_cols = df_in.select_dtypes(include=['object', 'string', 'category']).columns
    if len(str_cols) == 0:
        return df_in.to_csv(index=False).encode('utf-8')
    csv_buffer = io.StringIO()
    df_copy = df_in.copy(deep=False)
    for col in str_cols:
        s = df_copy[col].astype(str)
        bad_mask = s.str.contains(injection_pattern, regex=True, na=False)
        if bad_mask.any():
            df_copy[col] = np.where(bad_mask, "'" + s, s)
    df_copy.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

def parse_gsheet_url(url: str):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid_m = re.search(r"[#&]gid=([0-9]+)", url)
    gid = gid_m.group(1) if gid_m else "0"
    res = requests.get(f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}", timeout=10)
    res.raise_for_status()
    return pd.read_csv(io.BytesIO(res.content))

def parse_and_standardize_session(uploaded_files, gsheet_url=None, is_music=True):
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                f.seek(0)
                fn = f.name.lower()
                if fn.endswith('.csv'): frames.append(pd.read_csv(f))
                elif fn.endswith(('.xlsx', '.xls')): frames.append(pd.read_excel(f))
                elif fn.endswith('.json'):
                    c = json.load(f)
                    frames.append(pd.json_normalize(clean_raw_json_obj(c) if isinstance(c, list) else [clean_raw_json_obj(c)]))
            except Exception as e:
                log_error("File Parsing", e, "A file could not be parsed.")

    if gsheet_url and validate_gsheet_url(gsheet_url):
        try:
            frames.append(parse_gsheet_url(gsheet_url.strip()))
        except Exception as e:
            log_error("Google Sheets Ingestion", e, "Failed to load public Google Sheet.")

    if not frames: return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = strip_ip_addresses(df)

    if 'Track' in df.columns and df['Track'].astype(str).str.startswith("Watched ").any():
        df['Track'] = df['Track'].astype(str).str.replace(r"^Watched\s+", "", regex=True)

    if len(df) > MAX_TOTAL_ROWS:
        st.warning(f"Dataset truncated to the first {MAX_TOTAL_ROWS:,} rows.")
        df = df.iloc[:MAX_TOTAL_ROWS].copy()

    if is_music:
        aliases = {
            'ts': 'timestamp', 'endTime': 'timestamp', 'ms_played': 'ms_played', 'msPlayed': 'ms_played',
            'master_metadata_album_artist_name': 'Artist', 'artistName': 'Artist',
            'master_metadata_track_name': 'Track', 'trackName': 'Track',
            'master_metadata_album_album_name': 'Album', 'platform': 'Platform', 'conn_country': 'Country',
            'reason_end': 'reason_end', 'reasonEnd': 'reason_end',
            'Event Start Timestamp': 'timestamp', 'Content Name': 'Track', 'Song Name': 'Track',
            'Artist Name': 'Artist', 'Container Description': 'Album', 'Album Name': 'Album',
            'Play Duration Milliseconds': 'ms_played', 'End Reason Type': 'reason_end', 'Build Version': 'Platform',
            'track_title': 'Track', 'artist_name': 'Artist', 'album_title': 'Album',
            'datetime': 'timestamp', 'station_name': 'Platform',
            'SongTitle': 'Track', 'TrackName': 'Track', 'Song Title': 'Track', 'Track Name': 'Track',
            'ArtistName': 'Artist', 'AlbumTitle': 'Album', 'Album Title': 'Album',
            'PlayDate': 'timestamp', 'EventTimestamp': 'timestamp', 'PlaybackDate': 'timestamp',
            'Duration': 'ms_played', 'PlayDuration': 'ms_played',
            'title': 'Track', 'titleUrl': 'Track_URL', 'time': 'timestamp'
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

        if 'Track' in df.columns and 'Artist' in df.columns:
            df['Track (Artist)'] = df['Track'].fillna('Unknown Track').astype(str).str.strip() + " (" + df['Artist'].fillna('Unknown Artist').astype(str).str.strip() + ")"

        if 'Album' in df.columns and 'Artist' in df.columns:
            df['Album (Artist)'] = df['Album'].fillna('Unknown Album').astype(str).str.strip() + " (" + df['Artist'].fillna('Unknown Artist').astype(str).str.strip() + ")"

        if 'ms_played' in df.columns:
            df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce').fillna(0).astype(np.int64)
            if 'Hours Played' not in df.columns:
                df['Hours Played'] = (df['ms_played'] / 3600000.0).round(3)

        if 'Artist' in df.columns:
            try:
                genre_map = enrich_artists_with_spotify(df['Artist'], top_limit=250)
                df['Genre'] = df['Artist'].map(genre_map).fillna('Other / Uncategorized')
            except Exception:
                df['Genre'] = 'Other / Uncategorized'

    for col in df.columns:
        if any(tok in str(col).lower() for tok in ['timestamp', 'date', 'time', 'created_at', 'endtime']):
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val_sample = df[col].dropna()
                    unit = 'ms' if not val_sample.empty and val_sample.iloc[0] > 1e11 else 's'
                    p = pd.to_datetime(df[col], unit=unit, errors='coerce')
                else:
                    p = pd.to_datetime(df[col], errors='coerce')

                if p.notna().sum() > 0.5 * len(df):
                    df[f"{col}_dt"] = p
                    df['Date_Only'] = p.dt.date
                    df['Time_Only'] = p.dt.strftime('%H:%M')
                    df['Hour_Only'] = p.dt.hour.astype(np.int8)
                    df['Day_of_Week'] = p.dt.day_name()
                    df['Month_Year'] = p.dt.to_period('M').astype(str)
                    break
            except Exception:
                pass

    return df

# ---------------------------------------------------------
# GUIDE RENDERING ENGINE
# ---------------------------------------------------------
def render_full_guide():
    st.markdown("### 💡 What Can I Do Here?")
    t_overview, t_requests = st.tabs(["📋 Overview", "📥 How to Request Data"])
    
    with t_overview:
        st.markdown('''
        * **📊 Multi-Select Breakdowns:** Rank items by play count, total hours listened, or column averages.
        * **📅 Period Comparisons:** Compare listening habits across multiple years or specific months side-by-side.
        * **📈 Time Trends:** Monthly activity volume trend lines with multi-item selection.
        * **🔥 Temporal Heatmaps:** Identify hourly activity clusters across days of the week.
        ''')

    with t_requests:
        st.markdown('''
        * **Spotify:** Go to [Spotify Privacy Settings](https://www.spotify.com/account/privacy/) and request **Extended Streaming History**.
        * **Apple Music:** Visit [Apple Data & Privacy](https://privacy.apple.com) and export **Apple Music Play Activity.csv**.
        * **YouTube Music:** Download history via [Google Takeout](https://takeout.google.com/).
        ''')

if nav_mode == "What Can I Do Here?":
    render_full_guide()
    st.stop()

# ---------------------------------------------------------
# VIEW: VISUALIZE MY DATA
# ---------------------------------------------------------
has_input_data = bool(uploaded or (gsheet and validate_gsheet_url(gsheet)))
is_music_profile = DATASET_PROFILES.get(selected_profile, {}).get("is_music", False)

if not has_input_data:
    st.info("👋 Upload CSV, Excel, or JSON files, or paste a public Google Sheets link in the sidebar to begin analyzing your data.")
    st.stop()

upload_signature = (
    tuple((f.name, getattr(f, 'size', 0)) for f in uploaded) if uploaded else (),
    gsheet.strip() if gsheet else "",
    is_music_profile
)

if "cached_df" not in st.session_state or st.session_state.get("last_upload_signature") != upload_signature:
    with st.spinner("⚡ Ingesting and processing data..."):
        st.session_state.cached_df = parse_and_standardize_session(uploaded, gsheet, is_music=is_music_profile)
        st.session_state.last_upload_signature = upload_signature

df = st.session_state.cached_df
if df.empty:
    st.warning("No records ingested. Check format and contents.")
    st.stop()

dt_col = next((c for c in df.columns if c.endswith('_dt')), None)
has_datetime = (dt_col is not None)
if has_datetime:
    d_start_val = df[dt_col].min().date()
    d_end_val = df[dt_col].max().date()
else:
    d_start_val, d_end_val = None, None

# Column detection
primary_music_cols = ['Artist', 'Track (Artist)', 'Album (Artist)', 'Genre']
existing_primary = [c for c in primary_music_cols if c in df.columns]
other_cat_cols = [
    c for c in df.select_dtypes(include=['object', 'category', 'string']).columns 
    if not c.endswith('_dt') and c not in IGNORED_COLS and c not in ['Date_Only', 'Time_Only', 'Day_of_Week', 'Month_Year'] and c not in existing_primary
]
ordered_cat_cols = existing_primary + other_cat_cols
if not ordered_cat_cols:
    ordered_cat_cols = [c for c in df.columns if not c.endswith('_dt')]

num_cols = df.select_dtypes(include=['number']).columns.tolist()

# --- SUMMARY KPI BANNER ---
st.markdown("<section aria-label='Summary KPI Banner'>", unsafe_allow_html=True)
if is_music_profile:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Unique Artists", f"{df['Artist'].nunique():,}" if 'Artist' in df.columns else "N/A")
    k2.metric("Unique Tracks", f"{df['Track (Artist)'].nunique():,}" if 'Track (Artist)' in df.columns else "N/A")
    k3.metric("Total Albums", f"{df['Album (Artist)'].nunique():,}" if 'Album (Artist)' in df.columns else "N/A")
    
    tot_hours = df['Hours Played'].sum() if 'Hours Played' in df.columns else 0.0
    k4.metric("Hours Played", f"{tot_hours:,.1f} hrs")
    
    full_cnt = int(df['reason_end'].astype(str).str.lower().isin(['trackdone', 'natural_end', 'end']).sum()) if 'reason_end' in df.columns else 0
    skip_cnt = int(df['reason_end'].astype(str).str.lower().isin(['fwdbtn', 'manual_skip', 'skip']).sum()) if 'reason_end' in df.columns else 0
    k5.metric("Full Plays", f"{full_cnt:,}")
    
    tot_plays = len(df)
    skip_rate = (skip_cnt / tot_plays * 100) if tot_plays > 0 else 0.0
    k6.metric("Skip Rate", f"{skip_rate:.1f}%")
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Records", f"{len(df):,}")
    k2.metric("Dimensions", len(ordered_cat_cols))
    k3.metric("Numeric Measures", len(num_cols))
st.markdown("</section>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

with st.expander("📊 Interactive Visualizations & Analysis", expanded=True):
    t_breakdown, t_comparison, t_temporal, t_heatmap = st.tabs([
        "📊 Breakdowns & Catalog",
        "📅 Date Comparisons",
        "📈 Time Trends",
        "🔥 Temporal Heatmaps"
    ])

    # --- TAB 1: BREAKDOWNS & CATALOG ---
    with t_breakdown:
        st.subheader("Category Breakdown")

        c_group, c_calc, c_num, c_duration = st.columns([1.2, 1.2, 1.2, 1.4])
        group_col = c_group.selectbox(
            "📊 Select Data to Chart (Y-Axis):", 
            ordered_cat_cols, 
            index=0, 
            key="b_group"
        )
        calc = c_calc.selectbox(
            "Calculation:", 
            ["Frequency / Count", "Sum of Column", "Average of Column"], 
            key="b_calc"
        )
        val_c = c_num.selectbox(
            "Numeric Column:", 
            num_cols, 
            key="b_val"
        ) if calc != "Frequency / Count" and num_cols else None

        min_play_sec_b = c_duration.slider(
            "Min Play Duration (Sec):",
            min_value=0,
            max_value=60,
            value=30,
            step=5,
            key="b_min_duration"
        )

        group_lower = group_col.lower()
        if "artist" in group_lower and "album" not in group_lower and "track" not in group_lower:
            canned_options = ["Custom (Use Slider)", "Top 50 Artists", "Top 100 Artists", "Top 250 Artists", "Top 500 Artists"]
        elif "track" in group_lower or "song" in group_lower:
            canned_options = ["Custom (Use Slider)", "Top 50 Songs", "Top 100 Songs", "Top 250 Songs", "Top 500 Songs"]
        elif "album" in group_lower:
            canned_options = ["Custom (Use Slider)", "Top 10 Albums", "Top 25 Albums", "Top 50 Albums", "Top 100 Albums"]
        else:
            canned_options = ["Custom (Use Slider)"]

        col_quick, col_cutoff = st.columns([1, 2])
        quick_choice = col_quick.selectbox("⚡ Quick Visuals Preset:", canned_options, index=0, key="b_quick_canned")

        is_custom = (quick_choice == "Custom (Use Slider)")
        if is_custom:
            chosen_top_n = None
        else:
            extracted = re.search(r"\d+", quick_choice)
            chosen_top_n = int(extracted.group()) if extracted else 20

        use_date_in_breakdown = has_datetime and enable_date_filtering
        
        if use_date_in_breakdown and d_start_val and d_start_val != d_end_val:
            with col_cutoff:
                if is_custom:
                    c_top_sub, c_date_sub = st.columns([1, 1])
                    top_n = c_top_sub.slider("Display Limit (Max 500):", min_value=5, max_value=500, value=20, key="b_top_side")
                    start_d, end_d = c_date_sub.slider("📅 Date Range Selection:", min_value=d_start_val, max_value=d_end_val, value=(d_start_val, d_end_val), format="YYYY-MM-DD", key="b_date_side")
                else:
                    top_n = chosen_top_n
                    start_d, end_d = st.slider("📅 Date Range Selection:", min_value=d_start_val, max_value=d_end_val, value=(d_start_val, d_end_val), format="YYYY-MM-DD", key="b_date_side_canned")
        else:
            with col_cutoff:
                if is_custom:
                    top_n = st.slider("Display Limit (Max 500):", min_value=5, max_value=500, value=20, key="b_top_solo")
                else:
                    top_n = chosen_top_n
                    st.markdown(f"<div style='margin-top: 26px; font-size: 14px; color: #94A3B8;'><b>Preset Active:</b> Showing Top {top_n} {group_col}s</div>", unsafe_allow_html=True)

        available_slice_dims = [c for c in ordered_cat_cols if c != group_col]
        mask = np.ones(len(df), dtype=bool)

        if available_slice_dims:
            with st.expander("🔍 Filter / Slice by Specific Attribute (Optional)", expanded=False):
                col_s1, col_s2 = st.columns([1, 2])
                slice_dim = col_s1.selectbox("Filter Dimension:", available_slice_dims, index=0, key="b_slice_dim")
                slice_vals = sorted(df[slice_dim].dropna().unique().astype(str))
                chosen_slice = col_s2.multiselect(f"Filter specific {slice_dim}s (leave blank for all):", options=slice_vals, default=[], key="b_slice_vals")
                if chosen_slice:
                    mask &= df[slice_dim].astype(str).isin(chosen_slice).to_numpy()

        if use_date_in_breakdown and d_start_val and d_start_val != d_end_val:
            mask &= (df['Date_Only'] >= start_d).to_numpy() & (df['Date_Only'] <= end_d).to_numpy()

        if 'ms_played' in df.columns and min_play_sec_b > 0:
            mask &= (df['ms_played'] >= (min_play_sec_b * 1000)).to_numpy()

        if exclude_unknowns:
            mask &= ~df[group_col].astype(str).str.strip().str.lower().isin(
                ['unknown', 'unknown artist', 'unknown track', 'unknown album', 'none', 'nan', '']
            ).to_numpy()

        active_chart_df = df[mask]

        if calc == "Frequency / Count":
            grouped = active_chart_df.groupby(group_col, observed=True).size().reset_index(name="Count")
            lbl = "Count"
        elif calc == "Sum of Column" and val_c:
            grouped = active_chart_df.groupby(group_col, observed=True)[val_c].sum().round(2).reset_index(name=f"Sum_{val_c}")
            lbl = f"Sum_{val_c}"
        elif calc == "Average of Column" and val_c:
            grouped = active_chart_df.groupby(group_col, observed=True)[val_c].mean().round(2).reset_index(name=f"Avg_{val_c}")
            lbl = f"Avg_{val_c}"
        else: 
            grouped = pd.DataFrame(); lbl = ""

        if not grouped.empty:
            summary_top = grouped.sort_values(by=lbl, ascending=False).head(top_n).copy()
            summary_top['formatted_label'] = summary_top[lbl].apply(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) and float(x).is_integer() else f"{x:,.2f}")

            fig = px.bar(
                summary_top, x=lbl, y=group_col, orientation='h', color=lbl, 
                text='formatted_label' if show_bar_labels else None,
                color_continuous_scale=selected_palette, 
                title=f"Top {len(summary_top)} {group_col} by {lbl}"
            )
            if show_bar_labels:
                fig.update_traces(textposition='outside', cliponaxis=False)

            ui_height = max(500, min(len(summary_top) * 26, 3200))
            yaxis_kwargs = dict(
                categoryorder='total ascending', 
                automargin=True,
                showgrid=show_gridlines,
                gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)"
            )
            if len(summary_top) <= 250:
                yaxis_kwargs['dtick'] = 1

            fig.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                yaxis=yaxis_kwargs, 
                xaxis=dict(
                    automargin=True, 
                    gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", 
                    showgrid=show_gridlines, 
                    color=chart_text
                ),
                margin=dict(l=max(200, min(int(summary_top[group_col].astype(str).str.len().max() * 9.5), 450)), r=80, t=50, b=50),
                height=ui_height,
                autosize=True
            )
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
            render_export_bar(fig, grouped, f"breakdown_{group_col}_{lbl}", total_rows=len(summary_top))
            
            with st.expander("📋 Searchable Master Catalog & Table"):
                st.dataframe(grouped.sort_values(by=lbl, ascending=False), hide_index=True, use_container_width=True)

    # --- TAB 2: DATE COMPARISONS ---
    with t_comparison:
        st.subheader("Period-over-Period Date Comparisons")
        if not has_datetime:
            st.info("No timestamp field detected in this dataset to perform date comparisons.")
        else:
            valid_dates_df = df.dropna(subset=[dt_col]).copy()
            
            if valid_dates_df.empty:
                st.warning("No valid timestamps available.")
            else:
                valid_dates_df["_Year_Int"] = valid_dates_df[dt_col].dt.year
                valid_dates_df["_Month_Year_Str"] = valid_dates_df[dt_col].dt.strftime("%Y-%m")
                
                avail_years = sorted(valid_dates_df["_Year_Int"].unique(), reverse=True)
                avail_months = sorted(valid_dates_df["_Month_Year_Str"].unique(), reverse=True)
                
                c_mode, c_dim, c_metric = st.columns([1.2, 1.4, 1.4])
                compare_mode = c_mode.radio(
                    "Comparison Granularity:",
                    ["Year-over-Year (Full Years)", "Month & Year (Specific Months)"],
                    key="cmp_granularity"
                )
                
                cmp_dim = c_dim.selectbox("Dimension to Compare:", ordered_cat_cols, index=0, key="cmp_dim")
                
                metric_options = ["Play / Event Count"]
                if "Hours Played" in df.columns:
                    metric_options.append("Total Hours Played")
                elif num_cols:
                    metric_options.extend([f"Sum of {c}" for c in num_cols[:3]])
                
                cmp_metric = c_metric.selectbox("Metric to Compare:", metric_options, key="cmp_metric")
                
                if compare_mode == "Year-over-Year (Full Years)":
                    default_periods = [str(y) for y in avail_years[:min(3, len(avail_years))]]
                    chosen_periods = st.multiselect(
                        "Select Years to Compare (Choose 2 to 5):",
                        options=[str(y) for y in avail_years],
                        default=default_periods,
                        max_selections=5,
                        key="cmp_chosen_years"
                    )
                    period_col = "_Year_Int"
                    filter_vals = [int(p) for p in chosen_periods] if chosen_periods else []
                else:
                    default_periods = avail_months[:min(3, len(avail_months))]
                    chosen_periods = st.multiselect(
                        "Select Months to Compare (Choose 2 to 5):",
                        options=avail_months,
                        default=default_periods,
                        max_selections=5,
                        key="cmp_chosen_months"
                    )
                    period_col = "_Month_Year_Str"
                    filter_vals = chosen_periods

                if len(chosen_periods) < 2:
                    st.info("💡 Please select at least 2 periods above to display comparative analytics.")
                else:
                    cmp_df = valid_dates_df[valid_dates_df[period_col].isin(filter_vals)].copy()
                    if exclude_unknowns:
                        cmp_df = cmp_df[~cmp_df[cmp_dim].astype(str).str.strip().str.lower().isin(
                            ['unknown', 'unknown artist', 'unknown track', 'unknown album', 'none', 'nan', '']
                        )]
                    
                    if 'ms_played' in cmp_df.columns and min_play_sec > 0:
                        cmp_df = cmp_df[cmp_df['ms_played'] >= (min_play_sec * 1000)]
                        
                    cmp_df["_Period_Label"] = cmp_df[period_col].astype(str)
                    
                    if cmp_metric == "Play / Event Count":
                        agg_res = cmp_df.groupby(["_Period_Label", cmp_dim], observed=True).size().reset_index(name="Value")
                    elif cmp_metric == "Total Hours Played":
                        agg_res = cmp_df.groupby(["_Period_Label", cmp_dim], observed=True)["Hours Played"].sum().round(2).reset_index(name="Value")
                    else:
                        num_f = cmp_metric.replace("Sum of ", "")
                        agg_res = cmp_df.groupby(["_Period_Label", cmp_dim], observed=True)[num_f].sum().round(2).reset_index(name="Value")
                    
                    top_overall = (
                        agg_res.groupby(cmp_dim, observed=True)["Value"].sum()
                        .sort_values(ascending=False)
                        .head(15)
                        .index.tolist()
                    )
                    
                    if top_overall:
                        plot_cmp = agg_res[agg_res[cmp_dim].isin(top_overall)].copy()
                        plot_cmp[cmp_dim] = pd.Categorical(plot_cmp[cmp_dim], categories=top_overall[::-1], ordered=True)
                        plot_cmp = plot_cmp.sort_values([cmp_dim, "_Period_Label"])
                        
                        fig_cmp = px.bar(
                            plot_cmp,
                            x="Value",
                            y=cmp_dim,
                            color="_Period_Label",
                            barmode="group",
                            orientation="h",
                            title=f"Top 15 {cmp_dim}s: Period Comparison ({cmp_metric})",
                            labels={"_Period_Label": "Period", "Value": cmp_metric},
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        fig_cmp.update_layout(
                            paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                            yaxis=dict(automargin=True, showgrid=show_gridlines, gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)"),
                            xaxis=dict(automargin=True, showgrid=show_gridlines, gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", color=chart_text),
                            margin=dict(l=150, r=40, t=50, b=50),
                            height=max(500, len(top_overall) * 36),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True, config={"responsive": True})
                        
                        pivot_cmp = agg_res.pivot_table(index=cmp_dim, columns="_Period_Label", values="Value", fill_value=0)
                        pivot_cmp["Total"] = pivot_cmp.sum(axis=1)
                        pivot_cmp = pivot_cmp.sort_values(by="Total", ascending=False).drop(columns=["Total"])
                        
                        with st.expander("📋 View Side-by-Side Comparison Table", expanded=False):
                            st.dataframe(pivot_cmp.head(100), use_container_width=True)
                        
                        render_export_bar(fig_cmp, pivot_cmp.reset_index(), "date_comparison_export", total_rows=len(top_overall))

    # --- TAB 3: TIME TRENDS ---
    with t_temporal:
        st.subheader("Time Series Trends")
        if not has_datetime:
            st.info("No timestamp field detected in this dataset.")
        else:
            date_filtered_t = apply_date_range_filter(df, "trends")
            if 'ms_played' in date_filtered_t.columns and min_play_sec > 0:
                date_filtered_t = date_filtered_t[date_filtered_t['ms_played'] >= (min_play_sec * 1000)]
            
            col_tdim, col_tmode = st.columns([1, 1])
            target_tdim = col_tdim.selectbox("Filter Dimension:", ordered_cat_cols, index=0, key="tab_t_dim")
            t_mode = col_tmode.radio("Selection Mode:", ["All Items (Complete Dataset)", "Multi-Select Specific Items"], horizontal=True, key="tab_t_mode")
            
            filtered_t_df = date_filtered_t
            if t_mode == "Multi-Select Specific Items":
                all_tvals = sorted(date_filtered_t[target_tdim].dropna().unique().astype(str))
                selected_tvals = st.multiselect(f"Select one or more {target_tdim}s:", options=all_tvals, default=all_tvals[:3] if len(all_tvals)>=3 else all_tvals, key="tab_t_ms")
                if not selected_tvals:
                    st.warning(f"Please select at least one {target_tdim}.")
                    st.stop()
                filtered_t_df = date_filtered_t[date_filtered_t[target_tdim].astype(str).isin(selected_tvals)]

            if 1 < filtered_t_df[target_tdim].nunique() <= 10:
                monthly = filtered_t_df.groupby(['Month_Year', target_tdim], observed=True).size().reset_index(name="Volume")
                fig_m = px.line(monthly, x='Month_Year', y='Volume', color=target_tdim, markers=True, title=f"Monthly Trend by {target_tdim}")
            else:
                monthly = filtered_t_df.groupby('Month_Year', observed=True).size().reset_index(name="Volume")
                fig_m = px.line(monthly, x='Month_Year', y='Volume', markers=True, title="Monthly Activity Volume")
                
            fig_m.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                xaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text), 
                yaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text),
                margin=dict(l=60, r=40, t=50, b=50),
                autosize=True
            )
            st.plotly_chart(fig_m, use_container_width=True, config={'responsive': True})
            render_export_bar(fig_m, monthly, "monthly_activity", total_rows=20)

    # --- TAB 4: HEATMAPS ---
    with t_heatmap:
        st.subheader("Temporal Density Heatmap")
        if not has_datetime:
            st.info("No timestamp field detected in this dataset.")
        else:
            date_filtered_h = apply_date_range_filter(df, "heatmap")
            if 'ms_played' in date_filtered_h.columns and min_play_sec > 0:
                date_filtered_h = date_filtered_h[date_filtered_h['ms_played'] >= (min_play_sec * 1000)]
                
            axis_col1, axis_col2 = st.columns(2)
            y_axis_choice = axis_col1.selectbox("Y-Axis:", ["Day of Week", "Date Only", "Month-Year"], index=0, key="h_y")
            x_axis_choice = axis_col2.selectbox("X-Axis:", ["Hour of Day (0–23)", "Time (HH:MM Exact)", "Date Only"], index=0, key="h_x")

            available_slice_dims_h = [c for c in ordered_cat_cols]
            mask_h = np.ones(len(date_filtered_h), dtype=bool)

            if available_slice_dims_h:
                with st.expander("🔍 Filter / Slice by Specific Attribute (Optional)", expanded=False):
                    col_hs1, col_hs2 = st.columns([1, 2])
                    slice_dim_h = col_hs1.selectbox("Filter Dimension:", available_slice_dims_h, index=0, key="h_slice_dim")
                    slice_vals_h = sorted(date_filtered_h[slice_dim_h].dropna().unique().astype(str))
                    chosen_slice_h = col_hs2.multiselect(
                        f"Filter specific {slice_dim_h}s (leave blank for all):",
                        options=slice_vals_h,
                        default=[],
                        key="h_slice_vals"
                    )
                    if chosen_slice_h:
                        mask_h &= date_filtered_h[slice_dim_h].astype(str).isin(chosen_slice_h).to_numpy()

            filtered_h_df = date_filtered_h[mask_h]

            col_map = {"Day of Week": "Day_of_Week", "Date Only": "Date_Only", "Month-Year": "Month_Year", "Hour of Day (0–23)": "Hour_Only", "Time (HH:MM Exact)": "Time_Only"}
            y_field, x_field = col_map[y_axis_choice], col_map[x_axis_choice]

            if y_field == x_field:
                st.warning("Select distinct dimensions for X and Y axes.")
            elif filtered_h_df.empty:
                st.warning("No records match the current filter selection.")
            else:
                pivot = filtered_h_df.groupby([y_field, x_field], observed=True).size().unstack(fill_value=0).astype(np.int32)
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                if y_field == "Day_of_Week":
                    pivot = pivot.reindex([d for d in days_order if d in pivot.index])
                if x_field == "Day_of_Week":
                    pivot = pivot.reindex(columns=[d for d in days_order if d in pivot.columns])

                fig_heat = px.imshow(
                    pivot, labels=dict(x=x_axis_choice, y=y_axis_choice, color="Events"),
                    aspect="auto", color_continuous_scale=selected_palette, title=f"Activity Heatmap ({y_axis_choice} vs {x_axis_choice})",
                    text_auto=True if len(pivot) <= 12 and len(pivot.columns) <= 12 else False
                )
                fig_heat.update_layout(
                    paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                    yaxis=dict(automargin=True, color=chart_text), xaxis=dict(automargin=True, color=chart_text),
                    margin=dict(l=100, r=40, t=50, b=50), height=520,
                    autosize=True
                )
                st.plotly_chart(fig_heat, use_container_width=True, config={'responsive': True})
                render_export_bar(fig_heat, pivot.reset_index(), "heatmap_export", total_rows=len(pivot))

# ---------------------------------------------------------
# RAW DATA INSPECTION DRAWER
# ---------------------------------------------------------
with st.expander("📄 View Ingested Raw Data Table", expanded=False):
    st.caption(f"Displaying {len(df):,} cleaned rows.")
    st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Ingested Dataset (CSV)",
        sanitize_csv_data(df),
        "cleaned_streaming_history.csv",
        "text/csv",
        key="dl_clean_history"
    )
