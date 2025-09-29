import os, io, re, json
from pathlib import Path
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from dotenv import load_dotenv



st.set_page_config(page_title="Assistant Analytique (type Genie)", layout="wide")

ENV_PATH = Path.cwd() / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)
for v in ["OPENAI_PROXY","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
    os.environ.pop(v, None)

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

if "conn" not in st.session_state:
    st.session_state.conn = duckdb.connect(database=":memory:")
if "table_name" not in st.session_state:
    st.session_state.table_name = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turns" not in st.session_state:
    st.session_state.turns = []

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
    except TypeError as e:
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

SYSTEM_PROMPT = """
Tu es un expert SQL (dialecte DuckDB).
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
  "chart": {{"type": "bar\nline\nscatter\nnull", "x": "<col\nnull>", "y": "<col\n[cols]\nnull>"}}
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

st.title("🧠 Assistant Analytique (type Genie) – MVP local")

# ----------- Upload CSV ou Coller Transactions -----------
file = st.file_uploader("📤 Charger un fichier CSV (UTF-8)", type=["csv"])
st.markdown("### 📋 Ou collez vos données ici (copier-coller brut)")
texte = st.text_area("Collez ici les lignes CSV (avec entête)", height=10)

df = None
if file:
    head = file.read(2048).decode("utf-8", errors="ignore"); file.seek(0)
    sep = ";" if head.count(";") > head.count(",") else ","
    try:
        df = pd.read_csv(file, sep=sep)
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}"); st.stop()
elif texte:
    try:
        head = texte[:2048]
        sep = ";" if head.count(";") > head.count(",") else ","
        df = pd.read_csv(io.StringIO(texte), sep=sep)
    except Exception as e:
        st.error(f"Erreur lecture des transactions collées : {e}")

if df is not None:
    # Normalisation simple
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["date","time","timestamp"]):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if any(k in lc for k in ["amount","montant","price","total"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    tn = "transactions_collees" if texte and not file else re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(file.name)[0].lower()) or "transactions"
    st.session_state.conn.register("tmp_df", df)
    st.session_state.conn.execute(f"CREATE OR REPLACE TABLE {tn} AS SELECT * FROM tmp_df")
    st.session_state.conn.unregister("tmp_df")
    st.session_state.table_name = tn
    st.markdown("### 🚨 Détection de fraude (règles simples)")

    if st.session_state.table_name:
        # Check if "Original Amount" exists
        if "Original Amount" in df.columns:
            query_fraude = f"""
            SELECT *
            FROM {st.session_state.table_name}
            WHERE "Original Amount" > (
                SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY "Original Amount")
                FROM {st.session_state.table_name}
            )
            OR EXTRACT(HOUR FROM "Transaction Initiated Time") BETWEEN 0 AND 5
            OR "Status" = 'Failed'
            """
            try:
                df_fraude = exec_sql_safe(query_fraude, st.session_state.conn)
                st.success(f"{len(df_fraude)} transactions potentiellement frauduleuses détectées.")
                show_dataframe(df_fraude)
                fig_fraude = plot_df(df_fraude, {"type": "scatter", "x": "Transaction Initiated Time", "y": ["Original Amount"]})
                show_chart(fig_fraude)
            except Exception as e:
                st.error(f"Erreur lors de la détection de fraude : {e}")
        else:
            st.warning('Colonne "Original Amount" absente : détection de fraude non appliquée.')

    schema_df = st.session_state.conn.execute(f"PRAGMA table_info('{tn}')").fetchdf()
    st.session_state.schema = [
        {"name": r["name"], "dtype": str(df[r["name"]].dtype) if r["name"] in df.columns else "unknown"}
        for _, r in schema_df.iterrows()
    ]
    st.success(f"✅ Données chargées en table **{tn}** ({len(df):,} lignes, {len(df.columns)} colonnes)")
    with st.expander("👀 Aperçu (50 premières lignes)"):
        show_dataframe(df.head(50))

st.markdown("---")

# ----------- Suggestions de questions -----------
suggestions = [
    "Quel est le montant total transféré par jour ?",
    "Quel est le nombre de transactions par heure ?",
    "Quel est le volume de transactions par canal ?",
    "Quel est le montant total par type de service ?",
]

st.markdown("### 💡 Suggestions de questions")
selected = st.selectbox("Clique sur une question pour la pré-remplir :", [""] + suggestions)

# ----------- Question -> Réponse (multi-cells) -----------
if st.session_state.table_name and st.session_state.schema:
    st.subheader("Historique des analyses")
    for i, t in enumerate(st.session_state.turns, start=1):
        with st.container():
            st.markdown(f"### 🔎 Question {i}")
            st.markdown(f"👤 {t['question']}")
            if t.get("sql"):
                st.markdown("**SQL généré :**")
                st.code(t["sql"], language="sql")
            if t.get("rationale"):
                st.caption("Raisonnement : " + t["rationale"])
            df_show = t.get("df")
            if df_show is not None and len(df_show) > 0:
                st.subheader("Résultats")
                show_dataframe(df_show)
                fig_i = plot_df(df_show, t.get("chart_cfg", {}))
                if fig_i:
                    show_chart(fig_i)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "⬇️ Export CSV",
                        df_show.to_csv(index=False).encode("utf-8"),
                        file_name=f"resultats_{i}.csv",
                        mime="text/csv",
                        key=f"csv_{i}",
                    )
                with c2:
                    buf_i = io.BytesIO()
                    # On retire le fuseau horaire de toutes les colonnes datetime
                    df_export = df_show.copy()
                    for col in df_export.select_dtypes(include=["datetimetz"]).columns:
                        df_export[col] = df_export[col].dt.tz_localize(None)
                    with pd.ExcelWriter(buf_i, engine="openpyxl") as xw:
                        df_export.to_excel(xw, index=False, sheet_name="résultats")

                    st.download_button(
                        "⬇️ Export Excel",
                        buf_i.getvalue(),
                        file_name=f"resultats_{i}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"xlsx_{i}",
                    )
            else:
                st.info("La requête n'a retourné aucune ligne.")
            st.markdown("---")
    st.subheader("Pose ta question")
    form_key = f"ask_form_{len(st.session_state.turns)}"
    # Ajout d'une variable pour stocker la question sélectionnée via bouton
    if "selected_suggestion" not in st.session_state:
        st.session_state.selected_suggestion = ""
    # Affichage des boutons horizontaux pour les suggestions
    cols = st.columns(len(suggestions))
    for idx, (col, sugg) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(sugg, key=f"sugg_btn_{len(st.session_state.turns)}_{idx}"):
                st.session_state.selected_suggestion = sugg
                # Effet popup : toast si dispo, sinon info
                if hasattr(st, "toast"):
                    st.toast(f"Suggestion sélectionnée : {sugg}")
                else:
                    st.info(f"Suggestion sélectionnée : {sugg}")
    default_question = st.session_state.selected_suggestion
    with st.form(key=form_key, clear_on_submit=True):
        question = st.text_area(
            "Ex.: « CA et #transactions par jour sur 7 jours par canal »",
            value=default_question,
            height=80,
            key=f"q_{len(st.session_state.turns)}",
        )
        btn = st.form_submit_button("Analyser")
        if btn and question.strip():
            st.session_state.selected_suggestion = ""  # Reset après soumission
            st.session_state.messages.append({"role": "user", "content": question.strip()})
            if not READY:
                st.error("LLM non prêt. Ajoute OPENAI_API_KEY dans .env ou corrige la config.")
            else:
                try:
                    with st.spinner("Je génère la requête SQL..."):
                        plan = nl_to_sql(question.strip(), st.session_state.table_name, st.session_state.schema)
                        sql = (plan.get("sql", "") or "").strip()
                        rationale = plan.get("rationale", "")
                        chart_cfg = plan.get("chart", {}) or {}
                        if not sql:
                            st.session_state.turns.append({
                                "question": question.strip(),
                                "sql": "",
                                "rationale": "Impossible de générer une requête SQL.",
                                "df": pd.DataFrame(),
                                "chart_cfg": chart_cfg,
                            })
                            st.error("Impossible de générer une requête SQL.")
                            st.experimental_rerun()
                        df_res = exec_sql_safe(sql, st.session_state.conn)
                        st.session_state.turns.append({
                            "question": question.strip(),
                            "sql": sql,
                            "rationale": rationale,
                            "df": df_res.head(200) if df_res is not None else pd.DataFrame(),
                            "chart_cfg": chart_cfg,
                        })
                        if df_res is not None and len(df_res) > 0:
                            st.session_state.messages.append({"role": "assistant", "content": f"SQL exécuté ({len(df_res)} lignes)."})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": "Aucune ligne retournée."})
                        st.experimental_rerun()
                except Exception as e:
                    st.session_state.turns.append({
                        "question": question.strip(),
                        "sql": "",
                        "rationale": f"Erreur : {e}",
                        "df": pd.DataFrame(),
                        "chart_cfg": {},
                    })
                    st.error(f"Erreur : {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Erreur : {e}"})
                    st.experimental_rerun()
    if st.button("🧹 Effacer l'historique"):
        st.session_state.turns = []
        st.session_state.messages = []
        st.success("Historique effacé.")
else:
    st.info("🔹 Charge d’abord un CSV ou colle tes transactions (ex. `data/transactions_sample.csv`).")
