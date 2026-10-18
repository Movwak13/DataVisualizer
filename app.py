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
            del st.session_state[k]
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
# COMPREHENSIVE SENSITIVE PII & IP SANITIZATION ENGINE
# ---------------------------------------------------------
IPV4_REGEX = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
IPV6_REGEX = r'(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}'

SENSITIVE_KEY_PATTERNS = [
    # IP Addresses
    r'(^|_)ip(_addr|_address)?$', r'client_ip', r'remote_addr', r'ip_addr_decrypted',
    # User Identifiers
    r'(^|_)user(_?id|_name|_account)?$', r'customer_id', r'listener_id', r'store_account_id',
    # Hardware & Devices
    r'device(_?id|_identifier|_serial|_number)?$', r'hardware_id', r'build_version', r'user_agent',
    # Streaming & Account URIs
    r'spotify_(track|episode|audiobook)_uri', r'audiobook_chapter_uri',
    # Geo / Location / Sessions
    r'session_id', r'token', r'location_zip', r'postal_code', r'latitude', r'longitude'
]

def clean_raw_json_obj(item):
    """Recursively removes sensitive keys and scrubs IPs from nested dicts/lists in JSON."""
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
            elif isinstance(v, str):
                v_clean = re.sub(IPV4_REGEX, '[REDACTED_IP]', v)
                v_clean = re.sub(IPV6_REGEX, '[REDACTED_IP]', v_clean)
                cleaned[k] = v_clean
            else:
                cleaned[k] = v
        return cleaned
    return item

def strip_ip_addresses(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drops PII/IP columns and scrubs embedded IPs across all string fields."""
    cols_to_drop = [
        c for c in dataframe.columns 
        if any(re.search(pat, str(c).lower().strip()) for pat in SENSITIVE_KEY_PATTERNS)
    ]
    if cols_to_drop:
        dataframe = dataframe.drop(columns=cols_to_drop, errors='ignore')

    text_cols = dataframe.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        if col not in ['Date_Only', 'Time_Only', 'Day_of_Week', 'Month_Year', 'Hour_Only']:
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
# RESPONSIVE, CROSS-BROWSER CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    html {
        -webkit-tap-highlight-color: transparent;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
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
        overflow-x: auto !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
    }
    div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 8px 8px 0px 0px !important;
        padding: clamp(6px, 1.2vw, 10px) clamp(12px, 1.8vw, 22px) !important;
        font-size: clamp(12px, 1vw + 2px, 14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom: none !important;
        background-color: transparent !important;
        flex-shrink: 0 !important;
        min-height: 42px !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-bottom: 3px solid #EF4444 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .sidebar-nav-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] div.row-widget.stButton button {
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 8px 6px !important;
        min-height: 40px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    .legal-card {
        background-color: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 12px;
        font-size: 11.5px;
        line-height: 1.45;
        color: #FCA5A5;
    }
    .legal-card strong {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-title-wrapper">
    <h1 class="app-title-text">📊 Universal Multi-Select Data Visualizer</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# RESTORED SIDEBAR NAVIGATION (TILE BUTTONS)
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
    enable_date_filtering = st.toggle(
        "Enable Date Range Filter", 
        value=True, 
        help="Turn on double-ended chronological sliders to constrain charts to specific date ranges."
    )
    show_bar_labels = st.toggle(
        "Show Bar Value Labels", 
        value=True, 
        help="Display exact counts or numeric totals at the outer tip of each bar for easy reading."
    )
    exclude_unknowns = st.toggle(
        "Exclude Unknown / Null Metadata",
        value=True,
        help="Hide 'Unknown Artist', 'Unknown Track', and empty records so real data occupies the top rankings."
    )
    ranking_direction = st.radio(
        "Ranking Direction:",
        options=["Highest First (Top N)", "Lowest First (Tail N)"],
        index=0,
        help="Switch between viewing your top performers and discovering forgotten tracks or hidden gems."
    )
    use_log_scale = st.toggle(
        "Logarithmic Value Scale",
        value=False,
        help="Compress extreme high-volume outliers so you can clearly compare medium and lower-volume items."
    )
    show_gridlines = st.toggle(
        "Show Canvas Gridlines",
        value=True,
        help="Display background calibration grid lines on charts and exported PNG graphics."
    )

with st.sidebar.expander("🎨 Stylings & Colors", expanded=False):
    available_palettes = STYLING_CFG.get("palettes", ["Viridis", "Plasma", "Turbo"])
    selected_palette = st.selectbox(
        "Color Palette:", 
        available_palettes, 
        index=0, 
        help="Select gradient color ramps applied to bar charts, heatmaps, and trend lines."
    )
    export_theme = st.selectbox(
        "PNG Export Background:", 
        ["Dark Canvas", "Light Canvas"], 
        index=0, 
        help="Determines whether image downloads render with a dark canvas (#111827) or light canvas (#FFFFFF)."
    )

chart_bg = "#111827" if export_theme == "Dark Canvas" else "#FFFFFF"
chart_text = "#E5E7EB" if export_theme == "Dark Canvas" else "#111827"
chart_grid = "#1F2937" if export_theme == "Dark Canvas" else "#E5E7EB"

# ---------------------------------------------------------
# DATA & PLATFORM INGESTION (LEGALLY SECURED)
# ---------------------------------------------------------
with st.sidebar.expander("💾 Data & Platform", expanded=True):
    st.markdown("""
    <div class="legal-card">
        <strong>⚠️ NOTICE & PRIVACY DISCLAIMER:</strong><br>
        Do <strong>NOT</strong> upload sensitive, proprietary, confidential, or legally protected data. 
        This includes <strong>Protected Health Information (PHI)</strong> under HIPAA, financial/banking details, 
        social security numbers, passwords, or classified material.<br><br>
        This utility is provided <em>"AS IS"</em> without warranty of any kind. This application is <strong>not a HIPAA Covered Entity or Business Associate</strong>. 
        By uploading data, you confirm that your files contain exclusively public or non-sensitive personal records and agree to hold harmless and indemnify the project owner from any liability, damages, or regulatory violations.
    </div>
    """, unsafe_allow_html=True)

    terms_accepted = st.checkbox(
        "I certify that my files contain NO sensitive data, PHI (HIPAA), or confidential info, and accept these terms.",
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
            key=f"uploader_{st.session_state.uploader_key}",
            help="Select streaming history files or spreadsheets. Processed in volatile memory only."
        )
        gsheet = st.text_input(
            "Public Google Sheet URL", 
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="Enter a public Google Sheets sharing link. First sheet will be fetched via secure HTTPS."
        )

        if st.button("🗑️ Reset Session & Clear Data", use_container_width=True, help="Immediately releases all memory structures and resets the view."):
            for k in list(st.session_state.keys()):
                if k not in ["uploader_key", "legal_acknowledgment", "nav_mode"]:
                    del st.session_state[k]
            gc.collect()
            st.rerun()

    def validate_gsheet_url(url: str) -> bool:
        if not url:
            return False
        clean_url = url.strip()
        parsed = urlparse(clean_url)
        if parsed.scheme != "https" or parsed.netloc != "docs.google.com":
            return False
        return bool(re.match(r"^/spreadsheets/d/([a-zA-Z0-9-_]+)", parsed.path))    def inspect_header_signature(uploaded_files, gsheet_url):
        cols_found = set()
        if uploaded_files:
            first_file = uploaded_files[0]
            if hasattr(first_file, 'size') and first_file.size > MAX_FILE_BYTES:
                return None
            try:
                first_file.seek(0)
                fn = first_file.name.lower()
                if fn.endswith('.csv'):
                    cols_found.update(pd.read_csv(first_file, nrows=2).columns)
                elif fn.endswith(('.xlsx', '.xls')):
                    cols_found.update(pd.read_excel(first_file, nrows=2).columns)
                elif fn.endswith('.json'):
                    content_json = json.load(first_file)
                    df_sample = pd.json_normalize(content_json[:2] if isinstance(content_json, list) else [content_json])
                    cols_found.update(df_sample.columns)
                first_file.seek(0)
            except Exception as e:
                logger.warning(f"Signature detection read issue: {type(e).__name__}")

        if not cols_found and validate_gsheet_url(gsheet_url):
            try:
                m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", gsheet_url.strip())
                if m:
                    res = requests.get(
                        f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid=0", 
                        timeout=5,
                        allow_redirects=False,
                        headers={"User-Agent": "DataVisualizer-CloudSecure/1.0"}
                    )
                    if res.status_code == 200:
                        cols_found.update(pd.read_csv(io.StringIO(res.text), nrows=2).columns)
            except Exception as e:
                logger.warning(f"GSheet signature detection issue: {type(e).__name__}")

        cols_lower = {str(c).lower().strip() for c in cols_found}
        if any(k in cols_lower for k in ['master_metadata_track_name', 'msplayed', 'ms_played', 'conn_country']):
            return "Spotify"
        if any(k in cols_lower for k in ['event start timestamp', 'play duration milliseconds', 'content name']):
            return "Apple Music"
        if any(k in cols_lower for k in ['station_name', 'station_history', 'track_title']) or ('artist_name' in cols_lower and 'album_title' in cols_lower):
            return "Pandora"
        if any(k in cols_lower for k in ['titleurl', 'subtitles']) or ('title' in cols_lower and 'header' in cols_lower):
            return "YouTube Music"
        if any(k in cols_lower for k in ['trackname', 'songtitle']) or ('artistname' in cols_lower and 'albumtitle' in cols_lower) or any(k in cols_lower for k in ['playdate', 'eventtimestamp']):
            return "Amazon Music"
        return None

    detected_platform = inspect_header_signature(uploaded, gsheet) if terms_accepted else None
    profile_names = list(DATASET_PROFILES.keys())
    active_idx = profile_names.index(detected_platform) if detected_platform in profile_names else profile_names.index("Other")

    selected_profile = st.selectbox(
        "Dataset Platform:", 
        profile_names, 
        index=active_idx, 
        disabled=False,
        help="Specifies the data schema for music metrics. Auto-detected from file headers, but you can override it manually."
    )
    if detected_platform:
        st.caption(f"Auto-detected platform: {detected_platform}")

# ---------------------------------------------------------
# OPTIMIZED VECTORIZED SANITIZATION & SAFE EXPORT
# ---------------------------------------------------------
def sanitize_csv_data(df_in: pd.DataFrame) -> bytes:
    df_copy = df_in.copy()
    injection_pattern = r"^[-=+@\t\r]"
    for col in df_copy.select_dtypes(include=['object', 'string', 'category']).columns:
        df_copy[col] = df_copy[col].astype(str)
        s = df_copy[col]
        bad_mask = s.str.contains(injection_pattern, regex=True, na=False)
        if bad_mask.any():
            df_copy.loc[bad_mask, col] = "'" + s[bad_mask]
    csv_bytes = df_copy.to_csv(index=False).encode('utf-8')
    del df_copy
    return csv_bytes

def parse_gsheet_url(url: str):
    if not validate_gsheet_url(url):
        raise ValueError("Invalid Google Sheets URL schema.")
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid_m = re.search(r"[#&]gid=([0-9]+)", url)
    gid = gid_m.group(1) if gid_m else "0"
    res = requests.get(
        f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}", 
        timeout=10,
        allow_redirects=False,
        headers={"User-Agent": "DataVisualizer-CloudSecure/1.0"}
    )
    res.raise_for_status()
    return pd.read_csv(io.StringIO(res.text))

def parse_and_standardize_session(uploaded_files, gsheet_url=None, is_music=True):
    frames = []
        if uploaded_files:
            for f in uploaded_files:
                if hasattr(f, 'size') and f.size > MAX_FILE_BYTES:
                    st.error(f"File exceeds maximum allowed size ({MAX_FILE_BYTES // (1024*1024)} MB).")
                    continue
                try:
                    f.seek(0)
                    fn = f.name.lower()
                    if fn.endswith('.csv'):
                        frames.append(pd.read_csv(f))
                    elif fn.endswith(('.xlsx', '.xls')):
                        frames.append(pd.read_excel(f))
                    elif fn.endswith('.json'):
                        c = json.load(f)
                        c = clean_raw_json_obj(c)
                        frames.append(pd.json_normalize(c if isinstance(c, list) else [c]))
                except Exception as e:
                    log_error("File Parsing", e, "A file could not be parsed.")
    if gsheet_url and validate_gsheet_url(gsheet_url):
        try:
            frames.append(parse_gsheet_url(gsheet_url.strip()))
        except Exception as e:
            log_error("Google Sheets Ingestion", e, "Failed to load public Google Sheet.")
    
    if not frames: 
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    del frames

    df = strip_ip_addresses(df)

    if 'Track' in df.columns and df['Track'].astype(str).str.startswith("Watched ").any():
        df['Track'] = df['Track'].astype(str).str.replace(r"^Watched\s+", "", regex=True)
    if 'subtitles' in df.columns and 'Artist' not in df.columns:
        def extract_yt_artist(val):
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return val[0].get('name', 'Unknown Artist')
            elif isinstance(val, str) and 'name' in val:
                try:
                    parsed = json.loads(val.replace("'", "\""))
                    return parsed[0].get('name', 'Unknown Artist') if isinstance(parsed, list) else 'Unknown Artist'
                except Exception:
                    return val
            return val
        df['Artist'] = df['subtitles'].apply(extract_yt_artist)

    if len(df) > MAX_TOTAL_ROWS:
        st.warning(f"Dataset truncated to the first {MAX_TOTAL_ROWS:,} rows to maintain container stability.")
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

    dt_protected = {'Date_Only', 'Time_Only', 'Hour_Only', 'Day_of_Week', 'Month_Year', 'timestamp'}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in dt_protected and not col.endswith('_dt'):
            if df[col].nunique() < (len(df) * 0.5):
                df[col] = df[col].astype('category')

    return df

def compute_streaks_and_sessions(data_df, session_idle_minutes=30, start_date=None, end_date=None):
    dt_candidates = [c for c in data_df.columns if c.endswith('_dt')]
    if not dt_candidates:
        return {}
    
    dt_col = dt_candidates[0]
    ts_all = data_df[dt_col].dropna().sort_values()
    if ts_all.empty:
        return {}

    def calculate_max_streak(timestamps):
        if timestamps.empty:
            return 0
        uniq_dates = pd.Series(timestamps.dt.date.unique()).sort_values()
        if len(uniq_dates) <= 1:
            return len(uniq_dates)
        diffs = (pd.to_datetime(uniq_dates) - pd.to_datetime(uniq_dates.shift(1))).dt.days
        streak_lens = uniq_dates.groupby((diffs != 1).cumsum()).size()
        return int(streak_lens.max())

    all_time_max_streak = calculate_max_streak(ts_all)

    if start_date is not None and end_date is not None:
        ts_frame = ts_all[(ts_all.dt.date >= start_date) & (ts_all.dt.date <= end_date)]
    else:
        ts_frame = ts_all

    selected_frame_streak = calculate_max_streak(ts_frame)

    active_ts = ts_frame if not ts_frame.empty else ts_all
    if len(active_ts) == 0:
        return {
            "all_time_max_streak": all_time_max_streak,
            "selected_frame_streak": 0,
            "total_sessions": 0,
            "avg_tracks_per_session": 0.0
        }

    epoch_seconds = active_ts.astype('int64').to_numpy() // 10**9
    time_gaps_min = np.diff(epoch_seconds) / 60.0
    session_cuts = np.where(time_gaps_min > session_idle_minutes)[0]
    
    total_sessions = len(session_cuts) + 1
    session_lengths = np.diff(np.concatenate(([-1], session_cuts, [len(active_ts) - 1])))
    avg_tracks_per_session = round(float(np.mean(session_lengths)), 1)

    return {
        "all_time_max_streak": all_time_max_streak,
        "selected_frame_streak": selected_frame_streak,
        "total_sessions": total_sessions,
        "avg_tracks_per_session": avg_tracks_per_session
    }

# ---------------------------------------------------------
# GUIDE RENDERING ENGINE (OVERVIEW & REQUEST TABS)
# ---------------------------------------------------------
def render_full_guide():
    st.markdown("### 💡 What Can I Do Here?")
    st.markdown("Explore your streaming activity, configure multi-select views, or follow instructions to export personal data from your music service.")
    
    t_overview, t_requests = st.tabs(["📋 Overview", "📥 How to Request Data"])
    
    with t_overview:
        col_left, col_right = st.columns([1, 1], gap="large")
        with col_left:
            st.markdown("#### 🧭 Types of Visualizations Available")
            st.markdown('''
            <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background-color: rgba(255,255,255,0.03);">
                <b>1. 📊 Multi-Select Breakdowns (Horizontal Bar Charts)</b><br>
                Rank entities such as Artists, Albums, Songs, or categories by raw play count, total hours listened, or column averages.
            </div>
            <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background-color: rgba(255,255,255,0.03);">
                <b>2. 📈 Multi-Select Time Trends (Time Series Line & Bar Charts)</b><br>
                Track activity month-over-month or by day of the week, comparing multiple entities over time.
            </div>
            <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background-color: rgba(255,255,255,0.03);">
                <b>3. 🔥 Temporal Density Heatmaps (Interactive Matrix)</b><br>
                Spot peak activity windows (Day of Week vs. Hour of Day). Brighter tiles highlight when you are most active.
            </div>
            <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); background-color: rgba(255,255,255,0.03);">
                <b>4. 📋 Catalog & Global Leaderboard</b><br>
                Master searchable index of all items with total hours, counts, and instant CSV export.
            </div>
            ''', unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 🎨 Interactive Sample Previews (Mock Data)")
            sample_b_data = pd.DataFrame({
                "Artist": ["Daft Punk", "The Weeknd", "Radiohead", "Kendrick Lamar", "Fleetwood Mac"],
                "Hours": [82.4, 65.1, 54.3, 49.0, 38.6]
            })
            fig_sample_b = px.bar(
                sample_b_data, x="Hours", y="Artist", orientation="h",
                color="Hours", color_continuous_scale=selected_palette,
                title="Sample Preview: Top Artists by Hours Played", text="Hours"
            )
            fig_sample_b.update_traces(texttemplate="%{text:.1f} hrs", textposition="outside", cliponaxis=False)
            fig_sample_b.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                yaxis=dict(categoryorder="total ascending", automargin=True),
                xaxis=dict(automargin=True, gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text),
                margin=dict(l=100, r=40, t=40, b=30), height=240
            )
            st.plotly_chart(fig_sample_b, use_container_width=True, config={'responsive': True})

    with t_requests:
        st.markdown("#### 📥 Where & How to Download Your Data")
        st.caption("Select your streaming provider below for step-by-step instructions and direct portal links.")

        with st.expander("🟢 Spotify (Extended Streaming History)", expanded=True):
            st.markdown('''
            * **Direct Privacy Portal:** [Spotify Privacy Settings](https://www.spotify.com/account/privacy/)
            * **Steps to Request:**
                1. Sign in to your Spotify Account and scroll down to **Download your data**.
                2. Request **Step 2: Extended streaming history** (the lifetime audio history) rather than the standard basic package.
                3. Confirm the verification email sent by Spotify.
            * **Delivery Time:** Typically 5 to 14 days.
            * **What to Upload:** Spotify delivers .zip batches containing files named Streaming_History_Audio_*.json or endsong_*.json. Drag and drop these .json files directly into this visualizer.
            ''')

        with st.expander("🍎 Apple Music", expanded=False):
            st.markdown('''
            * **Direct Privacy Portal:** [Apple Data & Privacy](https://privacy.apple.com)
            * **Steps to Request:**
                1. Sign in with your Apple ID.
                2. Under *Manage your data*, click **Request a copy of your data**.
                3. Check the box for **Apple Media Services information** and submit.
            * **Delivery Time:** Typically 2 to 5 days.
            * **What to Upload:** In the downloaded archive, open Apple_Media_Services/Apple Music Activity/ and upload Apple Music Play Activity.csv.
            ''')

        with st.expander("▶️ YouTube Music", expanded=False):
            st.markdown('''
            * **Direct Privacy Portal:** [Google Takeout](https://takeout.google.com/)
            * **Steps to Request:**
                1. Click **Deselect all**, then scroll down to **YouTube and YouTube Music**.
                2. Under the format settings, click **All YouTube data included** and select only **history**.
                3. Choose **JSON** as the export format if prompted, then click **Create export**.
            * **Delivery Time:** A few minutes to several hours.
            * **What to Upload:** Extract the downloaded archive and upload YouTube and YouTube Music/history/watch-history.json.
            ''')

        with st.expander("📦 Amazon Music", expanded=False):
            st.markdown('''
            * **Direct Privacy Portal:** [Amazon Privacy Central (Request My Data)](https://www.amazon.com/hz/privacy-central/data-requests/preview.html)
            * **Steps to Request:**
                1. Sign in to your Amazon account and navigate to *Request Your Information*.
                2. In the category dropdown, select **Amazon Music**.
                3. Submit the request and confirm via the verification link sent to your email.
            * **Delivery Time:** 24 to 72 hours.
            * **What to Upload:** Upload the received .csv or .json file (typically PlayHistory.csv or Amazon_Music_Listening_History.csv).
            ''')

        with st.expander("📻 Pandora", expanded=False):
            st.markdown('''
            * **Direct Privacy Portal:** [Pandora Help Center & Privacy](https://help.pandora.com/)
            * **Steps to Request:**
                1. Submit a customer privacy request under **"Request My Personal Information"** (GDPR / CCPA / US Privacy rights).
                2. Pandora support will verify your identity and generate your account listening archive.
            * **What to Upload:** Upload the delivered station_history.csv or play_history.csv.
            ''')

# If user selected Guide in sidebar, render guide and stop
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
    st.caption("🔒 **Online Privacy Protected:** Uploaded records exist solely in temporary container RAM during your active session. No user files are written to permanent server storage.")
    st.stop()

df = parse_and_standardize_session(uploaded, gsheet, is_music=is_music_profile)
if df.empty:
    st.warning("No records ingested. Check format and contents.")
    st.stop()

post_ingest_ram = get_container_ram_stats()
logger.info(f"Ingestion complete: {len(df):,} rows. Process RAM: {post_ingest_ram['process_mb']:.1f} MB | Container RAM: {post_ingest_ram['container_mb']:.1f} MB")

dt_col = next((c for c in df.columns if c.endswith('_dt')), None)
if dt_col:
    d_start_val = df[dt_col].min().date()
    d_end_val = df[dt_col].max().date()
else:
    d_start_val, d_end_val = None, None

# Top KPI & Metric Tabs
top_tab_qa, top_tab_info = st.tabs(["⚡ Quick Analysis", "ℹ️ Data Info"])
with top_tab_qa:
    if is_music_profile:
        st.markdown("<section aria-label='Music Summary Metrics'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Artists", f"{df['Artist'].nunique():,}" if 'Artist' in df.columns else "N/A")
        c2.metric("Unique Tracks", f"{df['Track (Artist)'].nunique():,}" if 'Track (Artist)' in df.columns else "N/A")
        c3.metric("Total Albums", f"{df['Album (Artist)'].nunique():,}" if 'Album (Artist)' in df.columns else "N/A")

        total_hours = df['Hours Played'].sum() if 'Hours Played' in df.columns else 0.0
        total_skipped = int(df['skipped'].fillna(False).sum()) if 'skipped' in df.columns else (
            int(df['reason_end'].astype(str).str.lower().isin(['fwdbtn', 'manual_skip', 'skip']).sum()) if 'reason_end' in df.columns else 0
        )
        full_plays = int(df['reason_end'].astype(str).str.lower().isin(['trackdone', 'natural_end', 'end']).sum()) if 'reason_end' in df.columns else 0

        p1, p2, p3 = st.columns(3)
        p1.metric("Total Hours Played", f"{total_hours:,.2f} hrs")
        p2.metric("Total Songs Skipped", f"{total_skipped:,}")
        p3.metric("Full Plays", f"{full_plays:,}")
        
        streaks = compute_streaks_and_sessions(df, start_date=d_start_val, end_date=d_end_val)
        if streaks:
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Longest Streak (All Time)", f"{streaks['all_time_max_streak']} days")
            s2.metric("Longest Streak (Selected Frame)", f"{streaks['selected_frame_streak']} days")
            s3.metric("Listening Sessions", f"{streaks['total_sessions']:,}")
            s4.metric("Avg Tracks / Session", f"{streaks['avg_tracks_per_session']}")
        st.markdown("</section>", unsafe_allow_html=True)
    else:
        st.markdown("<section aria-label='General Dataset Summary Metrics'>", unsafe_allow_html=True)
        num_cols_qa = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records Ingested", f"{len(df):,}")
        c2.metric("Total Attributes", len(df.columns))
        c3.metric("Numeric Metrics Detected", len(num_cols_qa))

        if num_cols_qa:
            lead_col = num_cols_qa[0]
            p1, p2, p3 = st.columns(3)
            p1.metric(f"Total Sum ({lead_col})", f"{df[lead_col].sum():,.2f}")
            p1.metric(f"Average ({lead_col})", f"{df[lead_col].mean():,.2f}")
            p3.metric(f"Max ({lead_col})", f"{df[lead_col].max():,.2f}")
        st.markdown("</section>", unsafe_allow_html=True)

with top_tab_info:
    st.markdown("<section aria-label='Data Info Metrics'>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    i1.metric("Total Records", f"{len(df):,}")
    i2.metric("Dimension Fields", len([c for c in df.columns if df[c].dtype.name in ['object', 'category', 'string']]))
    i3.metric("Date Tracking", "Active" if 'Date_Only' in df.columns else "Disabled")
    st.markdown("</section>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# OPTIMIZED COLUMN METADATA EXTRACTION
# ---------------------------------------------------------
has_datetime = 'Date_Only' in df.columns
dt_derived = {'Date_Only', 'Time_Only', 'Hour_Only', 'Day_of_Week', 'Month_Year'}
all_dt_cols = {c for c in df.columns if c.endswith('_dt') or c in dt_derived or c in ['timestamp', 'ts', 'endTime']}
num_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

cat_cols = [
    c for c in df.columns 
    if c not in all_dt_cols and c not in IGNORED_COLS and c not in ['ms_played', 'Hours Played']
    and (df[c].dtype.name in ['object', 'category', 'string', 'bool'])
]

display_cat_cols = [c for c in cat_cols if c not in ['Track', 'Album']]
priority_order = ['Artist', 'Track (Artist)', 'Album (Artist)']
ordered_cat_cols = [c for c in priority_order if c in display_cat_cols] + [c for c in display_cat_cols if c not in priority_order]

def render_export_bar(fig, table_df, base_name, total_rows=20):
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            "📥 Download CSV", 
            sanitize_csv_data(table_df), 
            f"{base_name}.csv", 
            "text/csv", 
            key=f"csv_{base_name}",
            help="Download the aggregated data table in sanitized CSV format."
        )
    with c2:
        try:
            export_fig = go.Figure(fig)
            export_fig.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
                font=dict(color=chart_text),
                xaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text),
                yaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text)
            )
            row_h = LIMITS_CFG.get("row_pixel_height", 26)
            calculated_height = max(800, int(total_rows * row_h) + 120)
            max_w = LIMITS_CFG.get("max_export_width", 1600)
            img_bytes = export_fig.to_image(format="png", width=max_w, height=calculated_height, scale=2)
            st.download_button(
                "🖼️ Download PNG", 
                img_bytes, f"{base_name}.png", 
                "image/png", key=f"png_{base_name}",
                help="Export a high-resolution standalone PNG graphic for reports or presentations."
            )
        except Exception as e:
            log_error("Image Export", e, "Export engine unavailable (requires python kaleido package).")

def apply_multiselect_filter(data_frame, key_prefix):
    col_dim, col_mode = st.columns([1, 1])
    target_dim = col_dim.selectbox(
        "Filter / Slice By Dimension:", 
        ordered_cat_cols, 
        index=0, 
        key=f"{key_prefix}_dim",
        help="Select the categorical field used to slice or narrow down items for analysis."
    )
    mode = col_mode.radio(
        "Selection Mode:", 
        ["All Items (Complete Dataset)", "Multi-Select Specific Items"], 
        horizontal=True, 
        key=f"{key_prefix}_mode",
        help="Choose 'All Items' to evaluate the full scope, or 'Multi-Select' to isolate specific artists, songs, or categories."
    )
    
    active_df = data_frame
    if mode == "Multi-Select Specific Items":
        all_vals = sorted(data_frame[target_dim].dropna().unique().astype(str))
        selected_vals = st.multiselect(
            f"Select one or more {target_dim}s:", 
            options=all_vals, 
            default=all_vals[:3] if len(all_vals)>=3 else all_vals, 
            key=f"{key_prefix}_ms",
            help=f"Select individual {target_dim} entries from the list. Records matching any selection will be included."
        )
        if not selected_vals:
            st.warning(f"Please select at least one {target_dim}.")
            st.stop()
        active_df = data_frame[data_frame[target_dim].astype(str).isin(selected_vals)]
    
    st.caption(f"Analyzing {len(active_df):,} of {len(data_frame):,} records.")
    return active_df, target_dim

def apply_date_range_filter(data_frame, key_prefix):
    if not has_datetime or not enable_date_filtering:
        return data_frame
    
    dt_col_local = next((c for c in data_frame.columns if c.endswith('_dt')), None)
    if not dt_col_local:
        return data_frame

    min_date_val = data_frame[dt_col_local].min().date()
    max_date_val = data_frame[dt_col_local].max().date()
    if min_date_val == max_date_val:
        return data_frame

    selected_range = st.slider(
        "📅 Date Range Filter:", 
        min_value=min_date_val, 
        max_value=max_date_val, 
        value=(min_date_val, max_date_val), 
        format="YYYY-MM-DD", 
        key=f"{key_prefix}_date",
        help="Drag start and end handles to constrain analysis to a specific chronological timeframe."
    )
    start_d, end_d = selected_range
    return data_frame[(data_frame['Date_Only'] >= start_d) & (data_frame['Date_Only'] <= end_d)]

# ---------------------------------------------------------
# LOWER DASHBOARD VISUALIZERS
# ---------------------------------------------------------
t_breakdown, t_temporal, t_heatmap, t_catalog = st.tabs([
    "📊 Multi-Select Breakdowns",
    "📈 Multi-Select Time Trends",
    "🔥 Multi-Select Heatmaps",
    "📋 Catalog & Rankings"
])

# --- TAB 1: BREAKDOWNS ---
with t_breakdown:
    st.subheader("Category Breakdown with Multi-Selection")

    c_group, c_calc, c_num, c_duration = st.columns([1.2, 1.2, 1.2, 1.4])
    group_col = c_group.selectbox(
        "📊 Select Data to Chart (Y-Axis):", 
        ordered_cat_cols, 
        index=0, 
        key="b_group",
        help="Primary category to plot on the vertical Y-axis (e.g., Artist, Track, Album)."
    )
    calc = c_calc.selectbox(
        "Calculation:", 
        ["Frequency / Count", "Sum of Column", "Average of Column"], 
        key="b_calc",
        help="Aggregation formula: Count totals plays, Sum totals numeric columns, Average calculates the mean."
    )
    val_c = c_num.selectbox(
        "Numeric Column:", 
        num_cols, 
        key="b_val",
        help="Select numeric column to sum or average when not using frequency count."
    ) if calc != "Frequency / Count" else None

    min_play_sec = c_duration.slider(
        "Min Play Duration (Sec):",
        min_value=0,
        max_value=60,
        value=30,
        step=5,
        key="b_min_duration",
        help="Filter out accidental skips and scrubbing by ignoring track plays shorter than this threshold."
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
    quick_choice = col_quick.selectbox(
        "⚡ Quick Visuals Preset:",
        canned_options,
        index=0,
        key="b_quick_canned",
        help="Instant leaderboard presets for this category, or choose Custom to manually control the count."
    )

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
                top_n = c_top_sub.slider(
                    "Display Limit (Max 500):", 
                    min_value=5, 
                    max_value=500, 
                    value=20, 
                    key="b_top_side",
                    help="Adjust how many ranked entries appear on the chart."
                )
                start_d, end_d = c_date_sub.slider(
                    "📅 Date Range Selection:", 
                    min_value=d_start_val, 
                    max_value=d_end_val, 
                    value=(d_start_val, d_end_val), 
                    format="YYYY-MM-DD", 
                    key="b_date_side",
                    help="Restrict the bar chart analysis to plays within this calendar window."
                )
            else:
                top_n = chosen_top_n
                start_d, end_d = st.slider(
                    "📅 Date Range Selection:", 
                    min_value=d_start_val, 
                    max_value=d_end_val, 
                    value=(d_start_val, d_end_val), 
                    format="YYYY-MM-DD", 
                    key="b_date_side_canned",
                    help="Restrict the bar chart analysis to plays within this calendar window."
                )
    else:
        with col_cutoff:
            if is_custom:
                top_n = st.slider(
                    "Display Limit (Max 500):", 
                    min_value=5, 
                    max_value=500, 
                    value=20, 
                    key="b_top_solo",
                    help="Adjust how many ranked entries appear on the chart."
                )
            else:
                top_n = chosen_top_n
                direction_label = "Bottom" if "Lowest" in ranking_direction else "Top"
                st.markdown(f"<div style='margin-top: 26px; font-size: 14px; color: #94A3B8;'><b>Preset Active:</b> Showing {direction_label.lower()} {top_n} {group_col}s</div>", unsafe_allow_html=True)

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

    if 'ms_played' in df.columns and min_play_sec > 0:
        mask &= (df['ms_played'] >= (min_play_sec * 1000)).to_numpy()

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
        is_ascending_sort = ("Lowest" in ranking_direction)
        summary_top = grouped.sort_values(by=lbl, ascending=is_ascending_sort).head(top_n).copy()
        summary_top['formatted_label'] = summary_top[lbl].apply(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) and float(x).is_integer() else f"{x:,.2f}")

        rank_title_prefix = "Bottom" if is_ascending_sort else "Top"
        fig = px.bar(
            summary_top, x=lbl, y=group_col, orientation='h', color=lbl, 
            text='formatted_label' if show_bar_labels else None,
            color_continuous_scale=selected_palette, 
            title=f"{rank_title_prefix} {len(summary_top)} {group_col} by {lbl}"
        )
        if show_bar_labels:
            fig.update_traces(textposition='outside', cliponaxis=False)

        ui_height = max(500, min(len(summary_top) * 26, 4000))
        yaxis_kwargs = dict(
            categoryorder='total ascending' if not is_ascending_sort else 'total descending', 
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
                color=chart_text,
                type="log" if use_log_scale else "linear"
            ),
            margin=dict(l=max(200, min(int(summary_top[group_col].astype(str).str.len().max() * 9.5), 450)), r=80, t=50, b=50),
            height=ui_height,
            autosize=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
        render_export_bar(fig, grouped, f"breakdown_{group_col}_{lbl}", total_rows=len(summary_top))
        with st.expander("View Data Table"):
            st.dataframe(summary_top.drop(columns=['formatted_label'], errors='ignore'), hide_index=True, use_container_width=True)

# --- TAB 2: TEMPORAL TRENDS ---
with t_temporal:
    st.subheader("Time Series Trends with Multi-Selection")
    if not has_datetime:
        st.info("No timestamp field detected in this dataset.")
    else:
        date_filtered_t = apply_date_range_filter(df, "trends")
        if 'ms_played' in date_filtered_t.columns and min_play_sec > 0:
            date_filtered_t = date_filtered_t[date_filtered_t['ms_played'] >= (min_play_sec * 1000)]
        
        filtered_t_df, t_dim = apply_multiselect_filter(date_filtered_t, "tab_t")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        c_tr, c_dw = st.columns(2)
        
        with c_tr:
            if 1 < filtered_t_df[t_dim].nunique() <= 10:
                monthly = filtered_t_df.groupby(['Month_Year', t_dim], observed=True).size().reset_index(name="Volume")
                fig_m = px.line(monthly, x='Month_Year', y='Volume', color=t_dim, markers=True, title=f"Monthly Trend by {t_dim}")
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

        with c_dw:
            if 1 < filtered_t_df[t_dim].nunique() <= 8:
                dw_agg = filtered_t_df.groupby(['Day_of_Week', t_dim], observed=True).size().reset_index(name="Volume")
                fig_dw = px.bar(dw_agg, x='Day_of_Week', y='Volume', color=t_dim, barmode='group', text='Volume' if show_bar_labels else None, title=f"Day of Week by {t_dim}")
            else:
                dw_agg = filtered_t_df.groupby('Day_of_Week', observed=True).size().reindex(days_order).fillna(0).reset_index(name="Volume")
                fig_dw = px.bar(dw_agg, x='Day_of_Week', y='Volume', title="Day of Week Distribution", color='Volume', text='Volume' if show_bar_labels else None, color_continuous_scale=selected_palette)
            
            if show_bar_labels:
                fig_dw.update_traces(textposition='outside', cliponaxis=False)
            fig_dw.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                xaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text), 
                yaxis=dict(gridcolor=chart_grid if show_gridlines else "rgba(0,0,0,0)", showgrid=show_gridlines, color=chart_text),
                margin=dict(l=60, r=40, t=50, b=50),
                autosize=True
            )
            st.plotly_chart(fig_dw, use_container_width=True, config={'responsive': True})
            render_export_bar(fig_dw, dw_agg, "dow_activity", total_rows=10)

# --- TAB 3: HEATMAPS ---
with t_heatmap:
    st.subheader("Temporal Density Heatmap")
    if not has_datetime:
        st.info("No timestamp field detected in this dataset.")
    else:
        date_filtered_h = apply_date_range_filter(df, "heatmap")
        if 'ms_played' in date_filtered_h.columns and min_play_sec > 0:
            date_filtered_h = date_filtered_h[date_filtered_h['ms_played'] >= (min_play_sec * 1000)]
            
        filtered_h_df, _ = apply_multiselect_filter(date_filtered_h, "tab_h")
        
        axis_col1, axis_col2 = st.columns(2)
        y_axis_choice = axis_col1.selectbox("Y-Axis:", ["Day of Week", "Date Only", "Month-Year"], index=0, key="h_y")
        x_axis_choice = axis_col2.selectbox("X-Axis:", ["Hour of Day (0–23)", "Time (HH:MM Exact)", "Date Only"], index=0, key="h_x")

        col_map = {"Day of Week": "Day_of_Week", "Date Only": "Date_Only", "Month-Year": "Month_Year", "Hour of Day (0–23)": "Hour_Only", "Time (HH:MM Exact)": "Time_Only"}
        y_field, x_field = col_map[y_axis_choice], col_map[x_axis_choice]

        if y_field == x_field:
            st.warning("Select distinct dimensions for X and Y axes.")
        else:
            pivot = filtered_h_df.groupby([y_field, x_field], observed=True).size().unstack(fill_value=0).astype(np.int32)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if y_field == "Day_of_Week":
                pivot = pivot.reindex([d for d in days_order if d in pivot.index])
            if x_field == "Day_of_Week":
                pivot = pivot.reindex(columns=[d for d in days_order if d in pivot.columns])

            fig_heat = px.imshow(
                pivot, labels=dict(x=x_axis_choice, y=y_axis_choice, color="Events"),
                aspect="auto", color_continuous_scale=selected_palette, title=f"Activity Heatmap ({y_axis_choice} vs {x_axis_choice})"
            )
            fig_heat.update_layout(
                paper_bgcolor=chart_bg, plot_bgcolor=chart_bg, font=dict(color=chart_text),
                yaxis=dict(automargin=True, color=chart_text), xaxis=dict(automargin=True, color=chart_text),
                margin=dict(l=100, r=40, t=50, b=50), height=520,
                autosize=True
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={'responsive': True})
            render_export_bar(fig_heat, pivot.reset_index(), "heatmap_export", total_rows=len(pivot))

# --- TAB 4: CATALOG & RANKINGS ---
with t_catalog:
    st.subheader("Global Leaderboard & Catalog")
    target_catalog = st.selectbox(
        "Entity to Inspect:", 
        ordered_cat_cols, 
        index=0, 
        key="cat_picker",
        help="Select entity type (Artist, Track, Album, Country) to populate the master sorted ranking catalog."
    )
    
    catalog_mask = np.ones(len(df), dtype=bool)
    if 'ms_played' in df.columns and min_play_sec > 0:
        catalog_mask &= (df['ms_played'] >= (min_play_sec * 1000)).to_numpy()
    
    if exclude_unknowns:
        catalog_mask &= ~df[target_catalog].astype(str).str.strip().str.lower().isin(
            ['unknown', 'unknown artist', 'unknown track', 'unknown album', 'none', 'nan', '']
        ).to_numpy()

    catalog_source_df = df[catalog_mask]

    agg_dict = {'Count': (target_catalog, 'count')}
    if 'Hours Played' in catalog_source_df.columns:
        agg_dict['Total Hours'] = ('Hours Played', 'sum')

    catalog_df = catalog_source_df.groupby(target_catalog, observed=True).agg(**agg_dict).reset_index()
    is_ascending_sort = ("Lowest" in ranking_direction)

    if 'Total Hours' in catalog_df.columns:
        catalog_df['Total Hours'] = catalog_df['Total Hours'].round(2)
        catalog_df = catalog_df.sort_values(by='Total Hours', ascending=is_ascending_sort)
    else:
        catalog_df = catalog_df.sort_values(by='Count', ascending=is_ascending_sort)

    c_search, c_page_size = st.columns([3, 1])
    search_query = c_search.text_input(
        f"Search {target_catalog}s (Press Enter):",
        placeholder="Type to filter rows instantly...",
        key=f"search_{target_catalog}"
    )
    page_size = c_page_size.selectbox("Rows per page:", [50, 100, 250], index=0)

    filtered_catalog = catalog_df
    if search_query.strip():
        safe_query = re.escape(search_query.strip())
        filtered_catalog = catalog_df[
            catalog_df[target_catalog].astype(str).str.contains(safe_query, case=False, na=False)
        ]

    total_matches = len(filtered_catalog)
    total_pages = max(1, (total_matches - 1) // page_size + 1)

    c_page_num, c_status = st.columns([1, 3])
    page_num = c_page_num.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_matches)

    c_status.markdown(
        f"<div style='margin-top: 32px; font-size: 13px; color: #94A3B8;'>"
        f"Showing rows <b>{start_idx + 1 if total_matches > 0 else 0}–{end_idx:,}</b> of <b>{total_matches:,}</b> items"
        f"</div>",
        unsafe_allow_html=True
    )

    st.dataframe(filtered_catalog.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)
    
    clean_export_name = target_catalog.lower().replace(' ', '_').replace('(', '').replace(')', '')
    st.download_button(
        f"📥 Download Full {target_catalog} Catalog (CSV)",
        sanitize_csv_data(filtered_catalog), 
        f"{clean_export_name}_catalog.csv", 
        "text/csv",
        key=f"dl_{clean_export_name}",
        help="Download the filtered catalog table as a sanitized CSV file."
    )

