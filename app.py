# app.py
# Assistant Analytique (type Genie) – MVP local
# Compatible Python 3.9.7 + Streamlit 1.12.0
# - Upload CSV -> DuckDB
# - Question NL -> SQL (OpenAI v1 ou v0.x auto-détecté)
# - Tableau + Graph + Exports
# - Garde-fous SQL (lecture seule)
# - Neutralisation des proxies d'environnement pour éviter l'erreur "proxies"

import os, io, re, json
from pathlib import Path

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from dotenv import load_dotenv

# ----------------- Config page -----------------
st.set_page_config(page_title="Assistant Analytique (type Genie)", layout="wide")

# ----------------- Charger .env explicitement -----------------
ENV_PATH = Path.cwd() / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Neutraliser d'éventuelles variables proxies qui cassent le SDK
for v in ["OPENAI_PROXY","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    os.environ.pop(v, None)

# ----------------- Helpers UI (compat Streamlit<=1.12) -----------------
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

# ----------------- Etat de session -----------------
if "conn" not in st.session_state:
    st.session_state.conn = duckdb.connect(database=":memory:")
if "table_name" not in st.session_state:
    st.session_state.table_name = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- LLM: init compatible v1 / v0.x -----------------
def init_chat_fn():
    """
    Retourne (impl, chat_fn, model, client_error)
    - impl: "v1", "v0", "azure-v1" ou None
    - chat_fn(messages:list[dict]) -> str (content)
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return (None, None, None, "OPENAI_API_KEY manquante dans .env")

    # Si Azure explicitement configuré (facultatif pour MVP)
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

        # Tentative SDK v1+
        from openai import OpenAI
        client = OpenAI(api_key=key)   # Aucun 'proxies' transmis
        def chat(messages):
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
            return r.choices[0].message.content
        return ("v1", chat, model, None)

    except TypeError as e:
        # Fallback SDK legacy (v0.x)
        try:
            import openai as openai_legacy
            openai_legacy.api_key = key
            def chat(messages):
                r = openai_legacy.ChatCompletion.create(model=model, messages=messages, temperature=0.1)
                return r["choices"][0]["message"]["content"]
            return ("v0", chat, model, None)
        except Exception as e2:
            return (None, None, None, f"{e} / {e2}")

    except Exception as e:
        return (None, None, None, str(e))

IMPL, CHAT, MODEL, CLIENT_ERR = init_chat_fn()
READY = (CHAT is not None) and (MODEL is not None)

# ----------------- NL -> SQL -----------------
SYSTEM_PROMPT = """Tu es un expert SQL (dialecte DuckDB).
Règles:
- Utilise UNIQUEMENT la table {table_name} et les colonnes listées.
- Génère une requête SQL de type SELECT, sans UPDATE/DELETE/INSERT/DDL.
- Ajoute des agrégations si nécessaire (sum, count, avg, date_trunc).
- Si aucune limite n'est fournie, considérer LIMIT 5000.
- Propose un graphique pertinent: "bar", "line" ou "scatter".
- Fuseau: Africa/Abidjan (UTC+0).
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

def nl_to_sql(question, table_name, schema):
    if not READY:
        raise RuntimeError(f"Client LLM non initialisé. Détails: {CLIENT_ERR or 'vérifie OPENAI_API_KEY / modèle'}")
    sys_prompt = SYSTEM_PROMPT.format(table_name=table_name)
    content = f"{schema_note(schema)}\n\nQuestion utilisateur:\n{question}"
    txt = CHAT([
        {"role":"system","content":sys_prompt},
        {"role":"user","content":content}
    ])
    m = re.search(r"\{.*\}", (txt or "").strip(), re.S)
    if not m:
        return {"sql":"","rationale":"Réponse non JSON.","chart":{"type":None,"x":None,"y":None}}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"sql":"","rationale":"JSON invalide.","chart":{"type":None,"x":None,"y":None}}

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
    if x is None: x = df.columns[0]
    if y is None:
        num = df.select_dtypes(include="number").columns.tolist()
        y = num[:1] if num else None
    try:
        if ctype == "bar":    return px.bar(df, x=x, y=y)
        if ctype == "line":   return px.line(df, x=x, y=y)
        if ctype == "scatter":return px.scatter(df, x=x, y=y)
    except Exception:
        return None

# ----------------- Sidebar -----------------
st.sidebar.header("⚙️ Paramètres")
if CLIENT_ERR:
    st.sidebar.error(f"LLM init error : {CLIENT_ERR}")
if READY:
    st.sidebar.success("LLM prêt : " + (IMPL or "OpenAI"))
else:
    st.sidebar.warning("LLM non prêt. Vérifie `.env` (OPENAI_API_KEY / OPENAI_MODEL).")

st.sidebar.markdown("### 💬 Exemples")
st.sidebar.write("""
- CA et #transactions par **jour** sur **7 jours**, par **canal** (graph lignes).
- Top 10 **marchands** par **taux d'échec** (7 derniers jours).
- Heures de **pic d'activité** hier (bar chart).
- Montants **> P99** (30 jours) avec **client, marchand, statut**.
""")

# ----------------- Titre -----------------
st.title("🧠 Assistant Analytique (type Genie) – MVP local")

# ----------------- Upload CSV -----------------
file = st.file_uploader("📤 Charger un fichier CSV (UTF-8)", type=["csv"])
if file:
    head = file.read(2048).decode("utf-8", errors="ignore"); file.seek(0)
    sep = ";" if head.count(";") > head.count(",") else ","
    try:
        df = pd.read_csv(file, sep=sep)
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}"); st.stop()

    # Normalisation simple
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["date","time","timestamp"]):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if any(k in lc for k in ["amount","montant","price","total"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")

    tn = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(file.name)[0].lower()) or "transactions"
    st.session_state.conn.register("tmp_df", df)
    st.session_state.conn.execute(f"CREATE OR REPLACE TABLE {tn} AS SELECT * FROM tmp_df")
    st.session_state.conn.unregister("tmp_df")
    st.session_state.table_name = tn

    schema_df = st.session_state.conn.execute(f"PRAGMA table_info('{tn}')").fetchdf()
    st.session_state.schema = [
        {"name": r["name"], "dtype": str(df[r["name"]].dtype) if r["name"] in df.columns else "unknown"}
        for _, r in schema_df.iterrows()
    ]

    st.success(f"✅ Fichier chargé en table **{tn}** ({len(df):,} lignes, {len(df.columns)} colonnes)")
    with st.expander("👀 Aperçu (50 premières lignes)"):
        show_dataframe(df.head(50))

st.markdown("---")

# ----------------- Question -> Réponse -----------------
if st.session_state.table_name and st.session_state.schema:
    if st.session_state.messages:
        st.subheader("Historique")
        for m in st.session_state.messages[-10:]:
            prefix = "👤" if m["role"] == "user" else "🤖"
            st.markdown(f"{prefix} {m['content']}")
        st.markdown("---")

    st.subheader("Pose ta question")
    with st.form(key="ask_form", clear_on_submit=False):
        question = st.text_area("Ex.: « CA et #transactions par jour sur 7 jours par canal »", height=80)
        btn = st.form_submit_button("Analyser")

    if btn and question.strip():
        st.session_state.messages.append({"role":"user","content":question.strip()})
        if not READY:
            st.error("LLM non prêt. Ajoute OPENAI_API_KEY dans .env ou corrige la config.")
        else:
            try:
                with st.spinner("Je génère la requête SQL..."):
                    plan = nl_to_sql(question.strip(), st.session_state.table_name, st.session_state.schema)
                sql = plan.get("sql","").strip()
                rationale = plan.get("rationale","")
                chart_cfg = plan.get("chart",{})

                if not sql:
                    st.error("Impossible de générer une requête SQL.")
                    st.session_state.messages.append({"role":"assistant","content":"Désolé, je n'ai pas pu générer de SQL."})
                else:
                    df_res = exec_sql_safe(sql, st.session_state.conn)
                    st.markdown("**SQL généré :**"); st.code(sql, language="sql")
                    if rationale: st.caption("Raisonnement : " + rationale)

                    if df_res is not None and len(df_res) > 0:
                        st.subheader("Résultats")
                        show_dataframe(df_res.head(200))
                        fig = plot_df(df_res, chart_cfg)
                        if fig: show_chart(fig)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button("⬇️ Export CSV",
                                df_res.to_csv(index=False).encode("utf-8"),
                                "resultats.csv","text/csv")
                        with c2:
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                                df_res.to_excel(xw, index=False, sheet_name="résultats")
                            st.download_button("⬇️ Export Excel", buf.getvalue(),
                                "resultats.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        st.session_state.messages.append({"role":"assistant","content":f"SQL exécuté ({len(df_res)} lignes)."})
                    else:
                        st.info("La requête n'a retourné aucune ligne.")
                        st.session_state.messages.append({"role":"assistant","content":"Aucune ligne retournée."})

            except Exception as e:
                st.error(f"Erreur : {e}")
                st.session_state.messages.append({"role":"assistant","content":f"Erreur : {e}"})

    if st.button("🧹 Effacer l'historique"):
        st.session_state.messages = []
        st.success("Historique effacé.")
else:
    st.info("🔹 Charge d’abord un CSV (ex. `data/transactions_sample.csv`).")
