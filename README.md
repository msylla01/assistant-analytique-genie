# Assistant Analytique (type Genie) – MVP local

## Prérequis
# 1️⃣ Aller dans le dossier du projet
cd ~/Desktop/Assistant-Analytique-type-Genie-MVP-local

# 2️⃣ Créer un environnement virtuel (si pas déjà fait)
python3 -m venv .venv

# 3️⃣ Activer l'environnement virtuel
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
# Windows CMD
# .venv\Scripts\activate.bat

# 4️⃣ Mettre à jour pip
pip install --upgrade pip

# 5️⃣ Installer toutes les dépendances
pip install -r requirements.txt

# 6️⃣ Copier le fichier d'exemple pour créer ton .env
cp .env.example .env   # Linux/macOS
# copy .env.example .env  # Windows

# 7️⃣ Éditer .env pour ajouter ta clé OpenAI
# (ouvre le fichier et remplace OPENAI_API_KEY par ta vraie clé)
# nano .env  # ou utilisez votre éditeur préféré

# 8️⃣ Lancer l'application Streamlit
streamlit run app.py

