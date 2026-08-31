# ISDMAAS service image.
#
# Two stages: dependencies are installed into a virtual environment in the
# builder and copied into a clean runtime, so the compiler toolchain never ships
# in the final image.

FROM python:3.11-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

# The service never needs to write outside its data directory, so it runs as an
# unprivileged user. A container that runs as root turns any file-write bug into
# a host-level concern.
RUN useradd --create-home --uid 10001 isdmaas

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ISDMAAS_ENV=production \
    ISDMAAS_DATA_DIR=/data \
    ISDMAAS_PHASE55_DIR=/opt/isdmaas/phase55

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/isdmaas/app
COPY --chown=isdmaas:isdmaas . /opt/isdmaas/app

# Writable state (catalogs, operator registry, operational database) lives on a
# volume; everything else is read-only application code.
RUN mkdir -p /data && chown isdmaas:isdmaas /data
VOLUME ["/data"]

USER isdmaas
EXPOSE 8000

# ISDMAAS_SECRET_KEY and ISDMAAS_CORS_ORIGINS are required at runtime — the
# service refuses to start in production without them. See DEPLOY.md.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "phase11_api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
