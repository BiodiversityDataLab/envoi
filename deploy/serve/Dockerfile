# Container image of the envoi web app in hosted mode, for SciLifeLab Serve.
#
# Build from the repository root, which must be the build context:
#
#     docker build --file deploy/serve/Dockerfile --tag envoi-webapp .
#     docker run --rm --publish 8501:8501 envoi-webapp
#
# The image copies only the files that it needs. It contains no credentials,
# tests, test data, or outputs: each user uploads their own Earth Engine key in
# the app. The .dockerignore file in the repository root also keeps such files
# out of the build context.

# python:3.12-slim (Debian, Python 3.12.15), pinned by the digest of its
# multi-platform index (Docker Hub, pushed 2026-10-06). The rasterio and pyproj
# wheels contain their own GDAL and PROJ. Their only system library that the
# slim image lacks is libexpat (installed below). Update the digest together
# with deploy/serve/requirements.txt.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

LABEL org.opencontainers.image.source="https://github.com/BiodiversityDataLab/envoi" \
      org.opencontainers.image.description="envoi web app in hosted mode" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The GDAL library inside the rasterio wheel loads libexpat.so.1, and the slim
# image does not have it. Without it, "import rasterio" fails, and the app
# fails at the first page view (the smoke check found this on 2026-10-09).
# The other system libraries of the pinned wheels (libc, libgcc, libstdc++,
# zlib) are in the base image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# The pinned dependencies first, so that a change in the source does not
# repeat this slow step. --no-deps installs exactly the pinned list. The
# "pip check" below fails the build when the list misses a dependency.
COPY deploy/serve/requirements.txt /tmp/envoi-build/requirements.txt
RUN pip install --no-cache-dir --no-deps --requirement /tmp/envoi-build/requirements.txt

# envoi from the source, into the system site-packages. pyproject.toml lists
# the package data (the dataset catalog and the web-app logos), and
# src/envoi/_version.py gives the version. The build folder is removed after
# the install.
COPY pyproject.toml README.md LICENSE /tmp/envoi-build/
COPY src/ /tmp/envoi-build/src/
RUN pip install --no-cache-dir --no-deps /tmp/envoi-build \
    && pip check \
    && rm -rf /tmp/envoi-build

# The user that runs the app. Serve runs the container as UID 1000.
RUN groupadd --gid 1000 envoi \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin envoi

# /app holds only the page script, the Streamlit settings, and the smoke check.
# root owns these files, so the app cannot change them.
WORKDIR /app
COPY deploy/serve/app.py deploy/serve/smoke_check.py ./
COPY deploy/serve/.streamlit/config.toml .streamlit/config.toml

ENV ENVOI_WEBAPP_MODE=hosted
USER 1000:1000
EXPOSE 8501

# The slim image has no curl, so the health check uses Python.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status != 200)"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
