import os, io, re, json, hashlib, sqlite3
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from dotenv import load_dotenv
st.set_page_config(page_title="Assistant Analytique Multi-utilisateurs", layout="wide")

ENV_PATH = Path.cwd() / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)
# =============================================================================
# OPTIMISATIONS PERFORMANCE POUR GROS VOLUMES
# =============================================================================
from functools import lru_cache
# =============================================================================
# 1. CONFIGURATION DUCKDB OPTIMISÉE
# =============================================================================

def init_optimized_duckdb():
    """Initialise DuckDB avec config optimisée pour gros volumes"""
    conn = duckdb.connect(database=":memory:")
    
    # Optimisations critiques
    conn.execute("SET memory_limit='8GB'")  # Limite mémoire
    conn.execute("SET threads=4")  # Parallélisation
    conn.execute("SET enable_progress_bar=false")
    conn.execute("SET preserve_insertion_order=false")  # Plus rapide
    
    # Cache résultats
    conn.execute("SET temp_directory='/tmp/duckdb_cache'")
    
    return conn

# Utilisation dans session_state
if "conn" not in st.session_state:
    st.session_state.conn = init_optimized_duckdb()

# =============================================================================
# 2. CHARGEMENT PROGRESSIF (STREAMING)
# =============================================================================

def load_large_csv_streaming(file_path, table_name, conn, chunk_size=100000):
    """Charge un CSV en chunks pour éviter saturation mémoire"""
    
    # Créer la table vide d'abord
    first_chunk = pd.read_csv(file_path, nrows=1000)
    conn.register("temp_schema", first_chunk)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_schema LIMIT 0")
    conn.unregister("temp_schema")
    
    # Charger par chunks
    total_rows = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        conn.register("temp_chunk", chunk)
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_chunk")
        conn.unregister("temp_chunk")
        
        total_rows += len(chunk)
        progress_bar.progress(min(i * chunk_size / 6000000, 1.0))
        status_text.text(f"Chargement : {total_rows:,} lignes...")
    
    progress_bar.empty()
    status_text.empty()
    return total_rows

# =============================================================================
# 3. ÉCHANTILLONNAGE INTELLIGENT
# =============================================================================

def smart_sampling(table_name, conn, max_rows=500000):
    """
    Pour très gros volumes, travailler sur un échantillon représentatif
    au lieu de tout charger
    """
    
    # Compter les lignes
    total = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    if total > max_rows:
        st.warning(f"⚠️ Table volumineuse ({total:,} lignes). Utilisation d'un échantillon de {max_rows:,} lignes.")
        
        # Créer vue échantillonnée
        sample_rate = max_rows / total
        conn.execute(f"""
            CREATE OR REPLACE VIEW {table_name}_sample AS 
            SELECT * FROM {table_name} 
            USING SAMPLE {sample_rate * 100}%
        """)
        
        return f"{table_name}_sample", True
    
    return table_name, False

# =============================================================================
# 4. CACHE REQUÊTES FRÉQUENTES
# =============================================================================

@lru_cache(maxsize=100)
def cached_query(sql_hash, sql, conn_id):
    """Cache les résultats des requêtes identiques"""
    # Note: conn doit être passé séparément car non hashable
    return st.session_state.conn.execute(sql).fetchdf()

def execute_with_cache(sql):
    """Exécute SQL avec cache"""
    sql_hash = hash(sql)
    conn_id = id(st.session_state.conn)
    return cached_query(sql_hash, sql, conn_id)

# =============================================================================
# 5. PAGINATION RÉSULTATS
# =============================================================================

def paginate_results(df, page_size=1000):
    """Affiche résultats par pages pour éviter crash navigateur"""
    
    if len(df) > page_size:
        st.info(f"📊 {len(df):,} résultats. Affichage paginé.")
        
        # Sélecteur de page
        num_pages = (len(df) - 1) // page_size + 1
        page = st.selectbox("Page", range(1, num_pages + 1), key="page_selector")
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.caption(f"Lignes {start_idx + 1} à {min(end_idx, len(df))} sur {len(df):,}")
        return df.iloc[start_idx:end_idx]
    
    return df

# =============================================================================
# 6. CONNEXION DIRECTE POSTGRESQL (SANS DUCKDB)
# =============================================================================

def execute_on_postgres_directly(sql, conn_params):
    """
    Pour TRÈS gros volumes, exécuter directement sur PostgreSQL
    sans tout charger dans DuckDB
    """
    import psycopg2
    
    conn = psycopg2.connect(**conn_params)
    
    # Limiter automatiquement
    if "limit" not in sql.lower():
        sql += " LIMIT 10000"
    
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    return df

# =============================================================================
# 7. MONITORING MÉMOIRE
# =============================================================================

import psutil

def check_memory_usage():
    """Vérifie l'usage mémoire et alerte si critique"""
    
    memory = psutil.virtual_memory()
    usage_percent = memory.percent
    
    if usage_percent > 85:
        st.error(f"⚠️ Mémoire critique : {usage_percent}% utilisée !")
        st.warning("Conseil : Réduire la taille des données ou utiliser l'échantillonnage")
        return False
    elif usage_percent > 70:
        st.warning(f"⚠️ Mémoire élevée : {usage_percent}% utilisée")
    
    return True

# Afficher dans sidebar
if st.sidebar.checkbox("Monitoring mémoire", value=False):
    memory = psutil.virtual_memory()
    st.sidebar.metric("Mémoire RAM", f"{memory.percent}%", 
                     delta=f"{memory.used / 1024**3:.1f} / {memory.total / 1024**3:.1f} GB")

# =============================================================================
# 8. USAGE DANS L'APP
# =============================================================================

# Exemple d'intégration dans ton code existant :

def improved_csv_upload():
    """Version améliorée du upload CSV"""
    
    file = st.file_uploader("CSV", type=["csv"])
    
    if file:
        # Sauver temporairement
        temp_path = f"/tmp/{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Vérifier taille
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        st.info(f"📦 Fichier : {file_size_mb:.1f} MB")
        
        if file_size_mb > 100:  # Plus de 100 MB
            st.warning("⚠️ Fichier volumineux. Chargement optimisé en cours...")
            
            # Chargement par streaming
            table_name = "large_data"
            total_rows = load_large_csv_streaming(
                temp_path, 
                table_name, 
                st.session_state.conn
            )
            
            st.success(f"✅ {total_rows:,} lignes chargées")
            
            # Proposer échantillonnage
            table_name, is_sampled = smart_sampling(table_name, st.session_state.conn)
            
            if is_sampled:
                st.info("💡 Les analyses se feront sur l'échantillon. Pour analyse complète, utiliser connexion directe PostgreSQL.")
        
        else:
            # Méthode standard pour petits fichiers
            df = pd.read_csv(temp_path)
            # ... ton code existant
        
        # Nettoyer
        os.remove(temp_path)

# =============================================================================
# 9. CONFIG RECOMMANDÉE PAR VOLUME DE DONNÉES
# =============================================================================


#DIMENSIONNEMENT SERVEUR SELON DONNÉES :

#┌──────────────────┬──────────┬─────────┬──────────┬─────────────┐
#│ Volume Données   │ RAM Min  │ CPU Min │ Stockage │ Users Max   │
#├──────────────────┼──────────┼─────────┼──────────┼─────────────┤
#│ < 1 GB           │ 8 GB     │ 4 cores │ 100 GB   │ 20          │
#│ 1-5 GB           │ 16 GB    │ 8 cores │ 250 GB   │ 50          │
#│ 5-20 GB          │ 32 GB    │ 16 cores│ 500 GB   │ 100         │
#│ > 20 GB          │ 64+ GB   │ 32 cores│ 1 TB SSD │ 200+        │
#└──────────────────┴──────────┴─────────┴──────────┴─────────────┘

#Pour GS2E avec potentiellement plusieurs GB :
#→ Recommandation : 32 GB RAM minimum
#→ Architecture hybrid : DuckDB pour petites requêtes + Connexion directe PostgreSQL pour gros volumes







# =============================================================================
# GESTION DES UTILISATEURS
# =============================================================================

DB_USERS = "users.db"

def init_users_db():
    """Initialise la base de données des utilisateurs"""
    conn = sqlite3.connect(DB_USERS)
    cursor = conn.cursor()
    
    # Table utilisateurs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table historique des analyses par utilisateur
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            sql_query TEXT,
            result_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Table configurations de connexion DB par utilisateur
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_db_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            config_name TEXT NOT NULL,
            db_type TEXT NOT NULL,
            connection_params TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email=None):
    """Crée un nouveau utilisateur"""
    try:
        conn = sqlite3.connect(DB_USERS)
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, password_hash, email)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, "Compte créé avec succès !"
    except sqlite3.IntegrityError:
        return False, "Ce nom d'utilisateur existe déjà."
    except Exception as e:
        return False, f"Erreur : {e}"

def authenticate_user(username, password):
    """Authentifie un utilisateur"""
    conn = sqlite3.connect(DB_USERS)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {"id": user[0], "username": user[1], "email": user[2]}
    return False, None

def save_user_analysis(user_id, question, sql_query, result_count):
    """Sauvegarde une analyse dans l'historique utilisateur"""
    try:
        conn = sqlite3.connect(DB_USERS)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_analyses (user_id, question, sql_query, result_count) VALUES (?, ?, ?, ?)",
            (user_id, question, sql_query, result_count)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde analyse : {e}")
        return False

def get_user_analyses(user_id, limit=10):
    """Récupère l'historique des analyses d'un utilisateur"""
    conn = sqlite3.connect(DB_USERS)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT question, sql_query, result_count, created_at 
           FROM user_analyses 
           WHERE user_id = ? 
           ORDER BY created_at DESC 
           LIMIT ?""",
        (user_id, limit)
    )
    analyses = cursor.fetchall()
    conn.close()
    return analyses

def save_db_config(user_id, config_name, db_type, connection_params):
    """Sauvegarde une configuration de connexion DB"""
    try:
        conn = sqlite3.connect(DB_USERS)
        cursor = conn.cursor()
        params_json = json.dumps(connection_params)
        cursor.execute(
            "INSERT INTO user_db_configs (user_id, config_name, db_type, connection_params) VALUES (?, ?, ?, ?)",
            (user_id, config_name, db_type, params_json)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde config : {e}")
        return False

def get_user_db_configs(user_id):
    """Récupère les configurations DB d'un utilisateur"""
    conn = sqlite3.connect(DB_USERS)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, config_name, db_type, connection_params FROM user_db_configs WHERE user_id = ?",
        (user_id,)
    )
    configs = cursor.fetchall()
    conn.close()
    return [{"id": c[0], "name": c[1], "type": c[2], "params": json.loads(c[3])} for c in configs]

# =============================================================================
# PAGE DE LOGIN
# =============================================================================

def show_login_page():
    """Affiche la page de connexion/inscription"""
    st.title("🔐 Connexion - Assistant Analytique")
    
    tab1, tab2 = st.tabs(["Se connecter", "Créer un compte"])
    
    with tab1:
        st.subheader("Connexion")
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")
            
            if submit:
                if not username or not password:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    success, user_data = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.success(f"Bienvenue {username} !")
                        st.experimental_rerun()
                    else:
                        st.error("Identifiants incorrects.")
    
    with tab2:
        st.subheader("Créer un compte")
        with st.form("signup_form"):
            new_username = st.text_input("Nom d'utilisateur", key="signup_username")
            new_email = st.text_input("Email (optionnel)", key="signup_email")
            new_password = st.text_input("Mot de passe", type="password", key="signup_password")
            new_password_confirm = st.text_input("Confirmer le mot de passe", type="password", key="signup_password_confirm")
            submit_signup = st.form_submit_button("Créer mon compte")
            
            if submit_signup:
                if not new_username or not new_password:
                    st.error("Veuillez remplir tous les champs obligatoires.")
                elif new_password != new_password_confirm:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(new_password) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères.")
                else:
                    success, message = create_user(new_username, new_password, new_email)
                    if success:
                        st.success(message + " Vous pouvez maintenant vous connecter.")
                    else:
                        st.error(message)

# =============================================================================
# FONCTIONS UTILITAIRES (reprises de app.py)
# =============================================================================

def show_dataframe(df: pd.DataFrame):
    try:
        st.dataframe(df, use_container_width=True)
    except TypeError:
        st.dataframe(df)

def show_chart(fig):
    if fig is None:
        return
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

def init_chat_fn():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return (None, None, None, "OPENAI_API_KEY manquante dans .env")
    use_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if use_azure else os.getenv("OPENAI_MODEL","gpt-4o-mini")
    try:
        if use_azure:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-10-21"),
            )
            def chat(messages):
                r = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
                return r.choices[0].message.content
            return ("azure-v1", chat, model, None)
        from openai import OpenAI
        client = OpenAI(api_key=key)
        def chat(messages):
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
            return r.choices[0].message.content
        return ("v1", chat, model, None)
    except Exception as e:
        return (None, None, None, str(e))

SYSTEM_PROMPT = """
Tu es un expert SQL (dialecte DuckDB).
Règles:
- Utilise UNIQUEMENT la table {table_name} et les colonnes listées.
- Génère une requête SQL de type SELECT, sans UPDATE/DELETE/INSERT/DDL.
- Ajoute des agrégations si nécessaire (sum, count, avg, date_trunc).
- Si aucune limite n'est fournie, considérer LIMIT 5000.
- Propose un graphique pertinent: "bar", "line" ou "scatter".
Répond STRICTEMENT au format JSON:
{{
  "sql": "<REQUETE SELECT>",
  "rationale": "<explication courte>",
  "chart": {{"type": "bar|line|scatter|null", "x": "<col|null>", "y": "<col|[cols]|null>"}}
}}
"""

def schema_note(schema):
    lines = [f"- {c['name']} ({c['dtype']})" for c in schema]
    return "Colonnes disponibles:\n" + "\n".join(lines)

def nl_to_sql(question, table_name, schema, chat_fn):
    sys_prompt = SYSTEM_PROMPT.format(table_name=table_name)
    content = f"{schema_note(schema)}\n\nQuestion utilisateur:\n{question}"
    txt = chat_fn([
        {"role":"system","content":sys_prompt},
        {"role":"user","content":content}
    ])
    m = re.search(r"\{.*\}", (txt or "").strip(), re.S)
    if not m:
        return {"sql":"", "rationale":"Réponse non JSON.", "chart":{"type":None,"x":None,"y":None}}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"sql":"", "rationale":"JSON invalide.", "chart":{"type":None,"x":None,"y":None}}

def exec_sql_safe(sql, conn) -> pd.DataFrame:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")
    for kw in [" update "," delete "," insert "," drop "," alter "," create "]:
        if kw in f" {s} ":
            raise ValueError("Mots-clés non autorisés.")
    if " limit " not in s:
        sql += " LIMIT 5000"
    return conn.execute(sql).fetchdf()

def plot_df(df: pd.DataFrame, cfg: dict):
    if not cfg or not cfg.get("type"):
        return None
    ctype = str(cfg.get("type") or "").lower()
    x = cfg.get("x"); y = cfg.get("y")
    if x and x not in df.columns: x = None
    if isinstance(y, list):
        y = [c for c in y if c in df.columns] or None
    elif isinstance(y, str) and y not in df.columns:
        y = None
    if x is None: x = df.columns[0] if len(df.columns) else None
    if y is None:
        num = df.select_dtypes(include="number").columns.tolist()
        y = num[:1] if num else None
    if x is None or y is None:
        return None
    try:
        if ctype == "bar": return px.bar(df, x=x, y=y)
        if ctype == "line": return px.line(df, x=x, y=y)
        if ctype == "scatter": return px.scatter(df, x=x, y=y)
    except Exception:
        return None

def connect_to_database(db_type, connection_params):
    """Établit une connexion à la base de données externe"""
    try:
        if db_type == "postgresql":
            import psycopg2
            conn = psycopg2.connect(**connection_params)
            return conn
        elif db_type == "mysql":
            import mysql.connector
            conn = mysql.connector.connect(**connection_params)
            return conn
        elif db_type == "sqlite":
            import sqlite3
            conn = sqlite3.connect(connection_params['database'])
            return conn
        else:
            raise ValueError(f"Type de base non supporté: {db_type}")
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        return None

def get_database_tables(conn, db_type):
    """Récupère la liste des tables"""
    try:
        if db_type == "postgresql":
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        elif db_type == "mysql":
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        elif db_type == "sqlite":
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
    except Exception as e:
        st.error(f"Erreur récupération tables: {e}")
        return []

def get_table_schema(conn, db_type, table_name):
    """Récupère le schéma d'une table"""
    try:
        if db_type == "postgresql":
            cursor = conn.cursor()
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
            schema = [{"name": row[0], "dtype": row[1]} for row in cursor.fetchall()]
            cursor.close()
            return schema
        elif db_type == "mysql":
            cursor = conn.cursor()
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = DATABASE()")
            schema = [{"name": row[0], "dtype": row[1]} for row in cursor.fetchall()]
            cursor.close()
            return schema
        elif db_type == "sqlite":
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema = [{"name": row[1], "dtype": row[2]} for row in cursor.fetchall()]
            cursor.close()
            return schema
    except Exception as e:
        st.error(f"Erreur schéma: {e}")
        return []

def get_table_data(conn, db_type, table_name, limit=100):
    """Récupère les données d'une table"""
    try:
        if db_type in ["postgresql", "mysql"]:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            cursor.close()
            return df
        elif db_type == "sqlite":
            return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
    except Exception as e:
        st.error(f"Erreur données: {e}")
        return pd.DataFrame()

# =============================================================================
# INITIALISATION SESSION STATE
# =============================================================================

# Initialisation DB utilisateurs
init_users_db()

# Session state pour auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# Session state pour l'app (par utilisateur)
if "conn" not in st.session_state:
    st.session_state.conn = None
if "table_name" not in st.session_state:
    st.session_state.table_name = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "turns" not in st.session_state:
    st.session_state.turns = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "last_table" not in st.session_state:
    st.session_state.last_table = None
if "external_conn" not in st.session_state:
    st.session_state.external_conn = None
if "db_tables" not in st.session_state:
    st.session_state.db_tables = []

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

# Si pas connecté, afficher la page de login
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# =============================================================================
# INTERFACE UTILISATEUR CONNECTÉ
# =============================================================================

# Init LLM
IMPL, CHAT, MODEL, CLIENT_ERR = init_chat_fn()
READY = (CHAT is not None) and (MODEL is not None)

# Init connexion DuckDB pour cet utilisateur
if st.session_state.conn is None:
    st.session_state.conn = duckdb.connect(database=":memory:")

# Header avec info utilisateur
col1, col2 = st.columns([4, 1])
with col1:
    st.title(f"🧠 Assistant Analytique - Bonjour {st.session_state.user['username']} !")
with col2:
    if st.button("🚪 Déconnexion"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.conn = None
        st.session_state.turns = []
        st.experimental_rerun()

# Sidebar
st.sidebar.header(f"👤 {st.session_state.user['username']}")

# Historique des analyses
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Dernières analyses")
user_history = get_user_analyses(st.session_state.user['id'], limit=5)
if user_history:
    for h in user_history:
        with st.sidebar.expander(f"🔍 {h[0][:50]}..."):
            st.caption(f"📅 {h[3]}")
            st.code(h[1], language="sql")
            st.caption(f"📊 {h[2]} résultats")
else:
    st.sidebar.info("Aucune analyse pour le moment")

# Configuration DB
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Connexion Base de Données")

# Charger les configs sauvegardées
saved_configs = get_user_db_configs(st.session_state.user['id'])
config_names = ["Nouvelle connexion"] + ([c['name'] for c in saved_configs] if saved_configs else [])
selected_config = st.sidebar.selectbox("Configuration", config_names)

if selected_config != "Nouvelle connexion" and saved_configs:
    # Utiliser une config sauvegardée
    config = next(c for c in saved_configs if c['name'] == selected_config)
    db_type = config['type']
    connection_params = config['params']
    
    st.sidebar.info(f"Type: {db_type}")
    if st.sidebar.button("🔌 Se connecter", key="connect_saved"):
        conn = connect_to_database(db_type, connection_params)
        if conn:
            st.session_state.external_conn = conn
            st.session_state.db_type = db_type
            tables = get_database_tables(conn, db_type)
            st.session_state.db_tables = tables
            st.sidebar.success("✅ Connecté !")
            st.experimental_rerun()
else:
    # Nouvelle connexion
    db_type = st.sidebar.selectbox("Type de DB", ["postgresql", "mysql", "sqlite"])
    connection_params = {}
    
    if db_type == "postgresql":
        connection_params['host'] = st.sidebar.text_input("Host", "localhost", key="pg_host")
        connection_params['port'] = st.sidebar.number_input("Port", value=5432, step=1, key="pg_port")
        connection_params['database'] = st.sidebar.text_input("Database", key="pg_db")
        connection_params['user'] = st.sidebar.text_input("User", key="pg_user")
        connection_params['password'] = st.sidebar.text_input("Password", type="password", key="pg_pass")
    elif db_type == "mysql":
        connection_params['host'] = st.sidebar.text_input("Host", "localhost", key="my_host")
        connection_params['port'] = st.sidebar.number_input("Port", value=3306, step=1, key="my_port")
        connection_params['database'] = st.sidebar.text_input("Database", key="my_db")
        connection_params['user'] = st.sidebar.text_input("User", key="my_user")
        connection_params['password'] = st.sidebar.text_input("Password", type="password", key="my_pass")
    elif db_type == "sqlite":
        connection_params['database'] = st.sidebar.text_input("Chemin fichier", "database.db", key="sq_path")
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("🔌 Connecter", key="connect_new"):
            # Vérifier que les params nécessaires sont remplis
            if db_type in ["postgresql", "mysql"]:
                if not all([connection_params.get('host'), connection_params.get('database'), 
                           connection_params.get('user'), connection_params.get('password')]):
                    st.sidebar.error("Remplis tous les champs !")
                else:
                    conn = connect_to_database(db_type, connection_params)
                    if conn:
                        st.session_state.external_conn = conn
                        st.session_state.db_type = db_type
                        tables = get_database_tables(conn, db_type)
                        st.session_state.db_tables = tables
                        st.sidebar.success("✅ Connecté !")
                        st.experimental_rerun()
            else:  # sqlite
                conn = connect_to_database(db_type, connection_params)
                if conn:
                    st.session_state.external_conn = conn
                    st.session_state.db_type = db_type
                    tables = get_database_tables(conn, db_type)
                    st.session_state.db_tables = tables
                    st.sidebar.success("✅ Connecté !")
                    st.experimental_rerun()
    
    with col_b:
        config_name = st.text_input("Nom config", key="config_name_input", placeholder="Ma config")
        if st.button("💾", key="save_config", help="Sauvegarder cette config"):
            if config_name and connection_params:
                save_db_config(st.session_state.user['id'], config_name, db_type, connection_params)
                st.sidebar.success("Config sauvegardée !")
                st.experimental_rerun()

# Sélection de table si connecté
if st.session_state.external_conn and st.session_state.db_tables:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Tables disponibles")
    selected_table = st.sidebar.selectbox("Sélectionne une table", [""] + st.session_state.db_tables, key="table_selector")
    
    if selected_table:
        schema = get_table_schema(st.session_state.external_conn, st.session_state.db_type, selected_table)
        df_preview = get_table_data(st.session_state.external_conn, st.session_state.db_type, selected_table, 50)
        
        if not df_preview.empty:
            st.session_state.table_name = selected_table
            st.session_state.schema = schema
            
            # Charger la table dans DuckDB pour l'analyse
            st.session_state.conn.register("tmp_external", df_preview)
            st.session_state.conn.execute(f"CREATE OR REPLACE TABLE {selected_table} AS SELECT * FROM tmp_external")
            st.session_state.conn.unregister("tmp_external")
            
            # Générer suggestions uniquement si nouvelle table et LLM ready
            if st.session_state.last_table != selected_table and READY:
                try:
                    schema_str = schema_note(schema)
                    sample = df_preview.head(5).to_dict(orient="records")
                    prompt = f"{schema_str}\n\nVoici un extrait des données:\n{sample}\n\nPropose 3 questions pertinentes pour analyser ce jeu de données. Donne juste la question, sans numéro ni tiret."
                    resp = CHAT([
                        {"role": "system", "content": "Tu es un expert en data analytics."},
                        {"role": "user", "content": prompt}
                    ])
                    if isinstance(resp, str):
                        suggestions = [s.strip("-• 123456789.") for s in resp.split("\n") if s.strip() and len(s.strip()) > 10]
                    else:
                        suggestions = []
                    st.session_state.suggestions = suggestions[:3] if suggestions else []
                    st.session_state.last_table = selected_table
                    st.session_state.question_input = ""
                except Exception as e:
                    st.sidebar.warning(f"Erreur génération suggestions: {e}")
                    st.session_state.suggestions = []
            
            st.sidebar.success(f"✅ {selected_table} chargée")
            st.sidebar.caption(f"{len(df_preview)} lignes, {len(schema)} colonnes")
        else:
            st.sidebar.warning(f"Table {selected_table} vide ou inaccessible")

# Upload CSV
st.markdown("### 📤 Ou charger un CSV")
file = st.file_uploader("Charge un fichier CSV", type=["csv"])

if file:
    try:
        # Détecter le séparateur
        head = file.read(2048).decode("utf-8", errors="ignore")
        file.seek(0)
        sep = ";" if head.count(";") > head.count(",") else ","
        
        df = pd.read_csv(file, sep=sep)
        
        # Normalisation des colonnes
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["date", "time", "timestamp"]):
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, dayfirst=True)
            if any(k in lc for k in ["amount", "montant", "price", "total"]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        df = df.dropna(how="all")
        tn = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(file.name)[0].lower()) or "data"
        
        # Charger dans DuckDB
        st.session_state.conn.register("tmp_df", df)
        st.session_state.conn.execute(f"CREATE OR REPLACE TABLE {tn} AS SELECT * FROM tmp_df")
        st.session_state.conn.unregister("tmp_df")
        st.session_state.table_name = tn
        
        # Récupérer le schéma
        schema_df = st.session_state.conn.execute(f"PRAGMA table_info('{tn}')").fetchdf()
        st.session_state.schema = [
            {"name": r["name"], "dtype": str(df[r["name"]].dtype) if r["name"] in df.columns else "unknown"}
            for _, r in schema_df.iterrows()
        ]
        
        # Générer suggestions pour CSV
        if st.session_state.last_table != tn and READY:
            try:
                schema_str = schema_note(st.session_state.schema)
                sample = df.head(5).to_dict(orient="records")
                prompt = f"{schema_str}\n\nVoici un extrait:\n{sample}\n\nPropose 3 questions pertinentes. Juste les questions, sans numéro."
                resp = CHAT([
                    {"role": "system", "content": "Tu es un expert en data analytics."},
                    {"role": "user", "content": prompt}
                ])
                suggestions = [s.strip("-• 123456789.") for s in resp.split("\n") if s.strip() and len(s.strip()) > 10]
                st.session_state.suggestions = suggestions[:3] if suggestions else []
                st.session_state.last_table = tn
                st.session_state.question_input = ""
            except Exception as e:
                st.warning(f"Erreur génération suggestions: {e}")
                st.session_state.suggestions = []
        
        st.success(f"✅ {tn} chargée ({len(df):,} lignes, {len(df.columns)} colonnes)")
        
        with st.expander("👀 Aperçu des données"):
            show_dataframe(df.head(50))
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")

# =============================================================================
# ZONE D'ANALYSE
# =============================================================================

if not st.session_state.table_name:
    st.info("🔹 Connecte-toi à une DB ou charge un CSV pour commencer")
else:
    st.markdown("---")
    st.subheader("💬 Analyse de données")
    
    # Afficher info sur la table active
    st.caption(f"📊 Table active : **{st.session_state.table_name}**")
    
    # Historique des résultats en premier
    if st.session_state.turns:
        st.markdown("### 📊 Résultats")
        for i, t in enumerate(st.session_state.turns, 1):
            with st.expander(f"🔎 {i}. {t['question']}", expanded=(i == len(st.session_state.turns))):
                if t.get("sql"):
                    st.code(t["sql"], language="sql")
                if t.get("rationale"):
                    st.caption(t["rationale"])
                
                df_show = t.get("df")
                if df_show is not None and len(df_show) > 0:
                    show_dataframe(df_show)
                    
                    # Graphique
                    fig = plot_df(df_show, t.get("chart_cfg", {}))
                    if fig:
                        show_chart(fig)
                    
                    # Export
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "⬇️ CSV",
                            df_show.to_csv(index=False).encode("utf-8"),
                            file_name=f"analyse_{i}.csv",
                            mime="text/csv",
                            key=f"csv_{i}"
                        )
                    with col2:
                        buf = io.BytesIO()
                        # Copier le dataframe et enlever les timezones pour Excel
                        df_export = df_show.copy()
                        for col in df_export.select_dtypes(include=['datetime', 'datetimetz']).columns:
                            if hasattr(df_export[col].dtype, 'tz') and df_export[col].dtype.tz is not None:
                                df_export[col] = df_export[col].dt.tz_localize(None)
                        
                        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                            df_export.to_excel(xw, index=False, sheet_name="resultats")
                        st.download_button(
                            "⬇️ Excel",
                            buf.getvalue(),
                            file_name=f"analyse_{i}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"xlsx_{i}"
                        )
                else:
                    st.info("Aucun résultat")
        
        st.markdown("---")
    
    # Ensuite le formulaire de question EN BAS
    st.markdown("### ❓ Pose ta question")
    
    # Suggestions
    if st.session_state.suggestions and len(st.session_state.suggestions) > 0:
        used = [t['question'] for t in st.session_state.turns]
        available = [s for s in st.session_state.suggestions if s not in used]
        
        if available:
            st.markdown("**💡 Questions suggérées :**")
            cols = st.columns(len(available))
            for idx, (col, sugg) in enumerate(zip(cols, available)):
                with col:
                    if st.button(sugg, key=f"sugg_{idx}"):
                        st.session_state.question_input = sugg
                        st.experimental_rerun()
    else:
        if not READY:
            st.info("⚠️ LLM non configuré. Configure OPENAI_API_KEY dans .env pour avoir des suggestions")
        elif st.session_state.table_name:
            st.info("💡 Les suggestions apparaîtront après avoir sélectionné une table")
    
    # Formulaire question
    with st.form("ask_form"):
        question = st.text_area("Ta question", value=st.session_state.question_input, height=80)
        submit = st.form_submit_button("🔍 Analyser")
        
        if submit and question.strip():
            st.session_state.question_input = ""
            
            if not READY:
                st.error("LLM non prêt. Vérifie .env")
            else:
                try:
                    with st.spinner("Génération SQL..."):
                        plan = nl_to_sql(question.strip(), st.session_state.table_name, st.session_state.schema, CHAT)
                        sql = plan.get("sql", "").strip()
                        
                        if not sql:
                            st.error("Impossible de générer SQL")
                        else:
                            df_res = exec_sql_safe(sql, st.session_state.conn)
                            
                            # Sauvegarder l'analyse
                            save_user_analysis(
                                st.session_state.user['id'],
                                question.strip(),
                                sql,
                                len(df_res) if df_res is not None else 0
                            )
                            
                            st.session_state.turns.append({
                                "question": question.strip(),
                                "sql": sql,
                                "rationale": plan.get("rationale", ""),
                                "df": df_res.head(200) if df_res is not None else pd.DataFrame(),
                                "chart_cfg": plan.get("chart", {}),
                            })
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
    
    # Bouton effacer historique
    if st.session_state.turns:
        if st.button("🧹 Effacer l'historique de session"):
            st.session_state.turns = []
            st.success("Historique effacé")
            st.experimental_rerun()