"""Page script of the hosted envoi web app. The container runs ``streamlit run app.py``."""

import resource

# No core files: the Streamlit server process holds the service-account key
# that each session uploaded, and a core file would write it to disk. The hard
# limit is 0 too, so the process and the worker processes that it starts
# cannot raise the limit again. This script runs only in the Linux image.
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# The import comes after the limit on purpose.
from envoi_webapp.app import render_app

render_app()
