FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
# Fail the build here, not at the first request, if the headless OpenCV wheel is missing a system library.
RUN pip install --no-cache-dir -r requirements.txt && python -c "import cv2; print(cv2.__version__)"

COPY app/ app/
COPY data/ data/

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
