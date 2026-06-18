FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" numpy requests sgp4
COPY phase7_collision.py phase8_maneuver.py phase10_safety.py phase11_api.py ./
# mount or copy phase55/reports/state_*.json at runtime; set PHASE55_DIR
ENV PHASE55_DIR=/data/phase55
EXPOSE 8000
CMD ["uvicorn", "phase11_api:app", "--host", "0.0.0.0", "--port", "8000"]
