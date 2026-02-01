FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачивание русской модели Vosk (маленькая версия ~45MB)
RUN wget -q https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip \
    && unzip vosk-model-small-ru-0.22.zip \
    && mv vosk-model-small-ru-0.22 /app/model \
    && rm vosk-model-small-ru-0.22.zip

# Предзагрузка модели Silero TTS при сборке (кэшируется в образе)
RUN python -c "import torch; torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v4_ru')"

# Копирование кода бота
COPY bot.py .

# Переменные окружения
ENV VOSK_MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1
ENV TTS_SPEAKER=xenia

CMD ["python", "bot.py"]
