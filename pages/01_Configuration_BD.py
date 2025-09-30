import streamlit as st
import psycopg2

st.title("🔗 Configuration Base de Données")

db_type = st.selectbox(
    "Type de base de données",
    ["postgresql"],
    key="db_type_select"
)

connection_params = {}
connection_params['host'] = st.text_input("Host", "localhost")
connection_params['port'] = st.number_input("Port", value=5432, step=1)
connection_params['database'] = st.text_input("Database")
connection_params['user'] = st.text_input("Username")
connection_params['password'] = st.text_input("Password", type="password")

if st.button("Se connecter et afficher les tables"):
    try:
        conn = psycopg2.connect(
            host=connection_params['host'],
            port=connection_params['port'],
            dbname=connection_params['database'],
            user=connection_params['user'],
            password=connection_params['password']
        )
        st.session_state.conn = conn
        st.success("Connexion réussie à PostgreSQL !")

        # Récupérer et afficher les tables
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
        if tables:
            st.write("Tables disponibles :")
            st.write(tables)
        else:
            st.info("Aucune table trouvée dans le schéma public.")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")