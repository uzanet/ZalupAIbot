FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачивание русской модели Vosk (маленькая версия ~45MB)
RUN wget -q https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip \
    && unzip vosk-model-small-ru-0.22.zip \
    && mv vosk-model-small-ru-0.22 /app/model \
    && rm vosk-model-small-ru-0.22.zip

# Копирование кода бота
COPY bot.py .

# Переменные окружения
ENV VOSK_MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
