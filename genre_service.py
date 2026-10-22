import os, sqlite3, time, base64, requests, logging
import streamlit as st

logger = logging.getLogger("DataVisualizerLogger")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artist_genre_cache.sqlite")

def init_genre_cache():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS artist_genres (
                artist_name TEXT PRIMARY KEY,
                spotify_id TEXT,
                primary_genre TEXT,
                raw_genres TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite genre cache: {e}")

init_genre_cache()

def get_cached_genres(artist_names):
    if not artist_names:
        return {}
    results = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        placeholders = ','.join(['?'] * len(artist_names))
        c.execute(f"SELECT artist_name, primary_genre FROM artist_genres WHERE artist_name IN ({placeholders})", list(artist_names))
        for row in c.fetchall():
            if row[1] not in ("Indie / Unclassified", "Other / Uncategorized", "Other"):
                results[row[0]] = row[1]
        conn.close()
    except Exception as e:
        logger.error(f"Cache lookup error: {e}")
    return results

def save_cached_genres(artist_genre_dict):
    if not artist_genre_dict:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        data = [(k, v, v) for k, v in artist_genre_dict.items()]
        c.executemany("INSERT OR REPLACE INTO artist_genres (artist_name, primary_genre, raw_genres) VALUES (?, ?, ?)", data)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Cache save error: {e}")

def get_spotify_token():
    client_id = st.secrets.get("SPOTIFY_CLIENT_ID", os.environ.get("SPOTIFY_CLIENT_ID"))
    client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", os.environ.get("SPOTIFY_CLIENT_SECRET"))
    if not client_id or not client_secret:
        return None

    now = time.time()
    if "spotify_api_token" in st.session_state and st.session_state.get("spotify_token_expires", 0) > now:
        return st.session_state.spotify_api_token

    auth_str = f"{client_id}:{client_secret}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    try:
        res = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {b64_auth}"},
            timeout=8
        )
        if res.status_code == 200:
            data = res.json()
            token = data["access_token"]
            st.session_state.spotify_api_token = token
            st.session_state.spotify_token_expires = now + data.get("expires_in", 3600) - 60
            return token
    except Exception as e:
        logger.error(f"Spotify token request error: {e}")
    return None

GENRE_BUCKETS = {
    "Metal": ["metal", "death metal", "black metal", "thrash", "metalcore", "sludge", "doom metal", "deathcore", "djent", "power metal"],
    "Punk / Hardcore": ["punk", "hardcore", "post-hardcore", "pop punk", "emo", "screamo", "skate punk", "crust", "grindcore"],
    "Rock": ["rock", "grunge", "alternative rock", "classic rock", "hard rock", "psychedelic", "garage rock", "stoner rock", "blues rock"],
    "Indie / Alt": ["indie", "shoegaze", "dream pop", "post-punk", "art pop", "indie pop", "lo-fi", "slowcore", "noise pop", "new wave", "britpop"],
    "Electronic / Dance": ["electronic", "edm", "house", "techno", "synthwave", "ambient", "trance", "idm", "drum and bass", "dubstep", "electro", "garage", "downtempo"],
    "Hip-Hop / Rap": ["rap", "hip hop", "trap", "boom bap", "drill", "underground rap", "cloud rap", "conscious hip hop"],
    "R&B / Soul": ["r&b", "soul", "neo soul", "funk", "motown", "contemporary r&b", "disco"],
    "Pop": ["pop", "dance pop", "electropop", "synthpop", "k-pop", "j-pop", "hyperpop", "teen pop"],
    "Folk / Acoustic": ["folk", "singer-songwriter", "acoustic", "bluegrass", "americana", "traditional folk"],
    "Country": ["country", "alt-country", "outlaw country", "americana", "bluegrass", "nashville"],
    "Jazz / Blues": ["jazz", "blues", "bossa nova", "fusion", "bebop", "cool jazz", "soul jazz"],
    "Classical / Score": ["classical", "soundtrack", "score", "orchestral", "baroque", "romantic era", "chamber music"]
}

def map_tags_to_bucket(raw_tags):
    if not raw_tags:
        return None
    tags_lower = [str(t).lower() for t in raw_tags]
    for bucket, keywords in GENRE_BUCKETS.items():
        if any(kw in tag for tag in tags_lower for kw in keywords):
            return bucket
    for tag in tags_lower:
        if len(tag) > 2 and not any(ch in tag for ch in ["music", "seen live", "favorite"]):
            return tag.title()
    return None

def fetch_musicbrainz_genre(artist_name):
    try:
        url = "https://musicbrainz.org/ws/2/artist"
        params = {"query": f'artist:"{artist_name}"', "fmt": "json", "limit": 1}
        headers = {"User-Agent": "PersonalDataVisualizer/2.0 ( contact@movwak.dev )"}
        res = requests.get(url, params=params, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            artists = data.get("artists", [])
            if artists:
                tags = [t.get("name") for t in artists[0].get("tags", [])]
                genres = [g.get("name") for g in artists[0].get("genres", [])]
                all_tags = genres + tags
                if all_tags:
                    return map_tags_to_bucket(all_tags)
    except Exception:
        pass
    return None

def enrich_artists_with_spotify(artist_series, top_limit=250):
    init_genre_cache()
    counts = artist_series.value_counts()
    top_artists = counts.index[:top_limit].tolist()
    
    resolved = get_cached_genres(top_artists)
    missing = [a for a in top_artists if a not in resolved]

    if missing:
        token = get_spotify_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        newly_resolved = {}
        
        # Batch resolve up to 60 missing artists per run
        for a_name in missing[:60]:
            clean_name = str(a_name).strip()
            genre_found = None
            
            # 1. Try Spotify Direct Artist Search & Profile
            if token:
                try:
                    s_res = requests.get(
                        "https://api.spotify.com/v1/search",
                        params={"q": f'artist:"{clean_name}"', "type": "artist", "limit": 1},
                        headers=headers,
                        timeout=4
                    )
                    if s_res.status_code == 200:
                        items = s_res.json().get("artists", {}).get("items", [])
                        if items:
                            art_id = items[0].get("id")
                            raw_genres = items[0].get("genres", [])
                            
                            # If search genres empty, query direct artist endpoint
                            if not raw_genres and art_id:
                                a_res = requests.get(f"https://api.spotify.com/v1/artists/{art_id}", headers=headers, timeout=4)
                                if a_res.status_code == 200:
                                    raw_genres = a_res.json().get("genres", [])
                                    
                            genre_found = map_tags_to_bucket(raw_genres)
                except Exception:
                    pass

            # 2. Fallback to MusicBrainz if Spotify returns no genre tags
            if not genre_found:
                genre_found = fetch_musicbrainz_genre(clean_name)

            newly_resolved[clean_name] = genre_found if genre_found else "Other / Indie"

        save_cached_genres(newly_resolved)
        resolved.update(newly_resolved)

    return resolved
