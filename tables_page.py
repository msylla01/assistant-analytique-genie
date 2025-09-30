import streamlit as st

st.title("Tables disponibles dans la base de données")

tables = st.session_state.get("tables", [])
if tables:
    st.write("Tables :")
    st.write(tables)
else:
    st.info("Aucune table trouvée ou pas de connexion active.")