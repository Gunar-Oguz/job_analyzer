"""Entry point for Streamlit Community Cloud: `streamlit run streamlit_app.py`.

Runs the app in standalone mode (no backend, no API keys). Set BACKEND_URL
to use the FastAPI backend instead.
"""

from frontend.app import main

main()
