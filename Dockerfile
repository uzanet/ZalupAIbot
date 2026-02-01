FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python зависимостей (с увеличенным таймаутом)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Предзагрузка модели Whisper (small) при сборке
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Предзагрузка модели Silero TTS при сборке
RUN python -c "import torch; torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v4_ru')"

# Копирование кода бота
COPY bot.py .

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV TTS_SPEAKER=xenia

CMD ["python", "bot.py"]
