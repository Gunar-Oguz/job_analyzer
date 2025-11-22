curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

mkdir backend

mkdir frontend

mkdir docker

touch README.MD

touch .gitignore

# files for .gitignore

cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
.DS_Store
*.log
.vscode/
EOF

cat .gitignore

git init

git add .

git commit -m "Initial Project Struture"

uvicorn main:app --reload

uv pip install --system --break-system-packages requests python-dotenv

touch fetch_jobs.py

uvicorn main:app --reload