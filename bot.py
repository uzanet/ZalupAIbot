#!/usr/bin/env python3
"""
ZalupAIBot - Telegram бот для расшифровки голосовых сообщений.
Использует Vosk для бесплатного офлайн распознавания русской речи.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from vosk import Model, KaldiRecognizer
import subprocess

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка токена бота
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set!")

# Путь к модели Vosk
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/app/model")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Глобальная переменная для модели Vosk
vosk_model = None


def load_vosk_model():
    """Загрузка модели Vosk для распознавания речи."""
    global vosk_model
    if vosk_model is None:
        logger.info(f"Загрузка модели Vosk из {MODEL_PATH}...")
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена по пути {MODEL_PATH}. "
                "Скачайте русскую модель с https://alphacephei.com/vosk/models"
            )
        vosk_model = Model(MODEL_PATH)
        logger.info("Модель Vosk успешно загружена!")
    return vosk_model


def convert_ogg_to_wav(ogg_path: str, wav_path: str) -> bool:
    """Конвертация OGG в WAV с помощью ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", ogg_path,
                "-ar", "16000",  # Частота дискретизации 16kHz для Vosk
                "-ac", "1",      # Моно
                "-f", "wav",
                "-y",            # Перезаписать если существует
                wav_path
            ],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка конвертации: {e.stderr.decode()}")
        return False


def transcribe_audio(wav_path: str) -> str:
    """Распознавание речи из WAV файла с помощью Vosk."""
    model = load_vosk_model()
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    result_text = []

    with open(wav_path, "rb") as audio_file:
        # Пропускаем WAV заголовок
        audio_file.read(44)

        while True:
            data = audio_file.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if result.get("text"):
                    result_text.append(result["text"])

        # Получаем финальный результат
        final_result = json.loads(recognizer.FinalResult())
        if final_result.get("text"):
            result_text.append(final_result["text"])

    return " ".join(result_text).strip()


def format_user_name(user: types.User) -> str:
    """Форматирование имени пользователя."""
    parts = []
    if user.first_name:
        parts.append(user.first_name)
    if user.last_name:
        parts.append(user.last_name)
    if not parts and user.username:
        parts.append(f"@{user.username}")
    return " ".join(parts) if parts else "Неизвестный"


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Обработка команды /start."""
    await message.answer(
        "👋 Привет! Я ZalupAI Bot.\n\n"
        "🎤 Я автоматически расшифровываю голосовые сообщения в текст.\n\n"
        "Добавь меня в групповой чат и дай права на чтение сообщений - "
        "я буду присылать текстовую расшифровку после каждого голосового сообщения."
    )


@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """Обработка голосовых сообщений."""
    user = message.from_user
    user_name = format_user_name(user)
    timestamp = datetime.now().strftime("%H:%M:%S")

    logger.info(f"Получено голосовое сообщение от {user_name}")

    # Отправляем уведомление о начале обработки
    processing_msg = await message.reply("🔄 Расшифровываю...")

    try:
        # Скачиваем голосовое сообщение
        voice = message.voice
        file = await bot.get_file(voice.file_id)

        # Создаем временные файлы
        with NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
            ogg_path = ogg_file.name

        with NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        try:
            # Скачиваем файл
            await bot.download_file(file.file_path, ogg_path)

            # Конвертируем в WAV
            if not convert_ogg_to_wav(ogg_path, wav_path):
                await processing_msg.edit_text("❌ Ошибка конвертации аудио")
                return

            # Распознаем речь
            text = transcribe_audio(wav_path)

            if not text:
                text = "[не удалось распознать речь]"

            # Формируем ответ
            response = f"🎙 **Голосовой эфир от пользователя ({user_name}) {timestamp}**\n\n{text}"

            # Редактируем сообщение с результатом
            await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

            logger.info(f"Расшифровка завершена: {text[:50]}...")

        finally:
            # Удаляем временные файлы
            for path in [ogg_path, wav_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")


@dp.message(F.video_note)
async def handle_video_note(message: types.Message):
    """Обработка видеосообщений (кружочков)."""
    user = message.from_user
    user_name = format_user_name(user)
    timestamp = datetime.now().strftime("%H:%M:%S")

    logger.info(f"Получено видеосообщение от {user_name}")

    processing_msg = await message.reply("🔄 Расшифровываю видеосообщение...")

    try:
        video_note = message.video_note
        file = await bot.get_file(video_note.file_id)

        with NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_path = video_file.name

        with NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        try:
            await bot.download_file(file.file_path, video_path)

            # Извлекаем аудио из видео
            if not convert_ogg_to_wav(video_path, wav_path):
                await processing_msg.edit_text("❌ Ошибка извлечения аудио")
                return

            text = transcribe_audio(wav_path)

            if not text:
                text = "[не удалось распознать речь]"

            response = f"🎥 **Голосовой эфир от пользователя ({user_name}) {timestamp}**\n\n{text}"

            await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

        finally:
            for path in [video_path, wav_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Ошибка обработки видеосообщения: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")


async def main():
    """Главная функция запуска бота."""
    logger.info("Запуск ZalupAI Bot...")

    # Предзагрузка модели при старте
    load_vosk_model()

    logger.info("Бот запущен и готов к работе!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
