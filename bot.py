#!/usr/bin/env python3
"""
ZalupAIBot - Telegram бот для расшифровки голосовых сообщений и озвучки текста.
Использует Whisper для распознавания речи и Silero TTS для синтеза.
"""

import os
import asyncio
import logging
import re
from datetime import datetime
from tempfile import NamedTemporaryFile

import torch
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters.callback_data import CallbackData
from faster_whisper import WhisperModel
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

# ID администратора (только он может менять настройки)
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# Настройки Whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Доступные голоса Silero TTS v4
AVAILABLE_VOICES = {
    "xenia": "Ксения (женский)",
    "aidar": "Айдар (мужской)",
    "baya": "Бая (женский)",
    "kseniya": "Ксения 2 (женский)",
    "eugene": "Евгений (мужской)",
}

# Глобальные настройки TTS (можно менять через команды)
tts_settings = {
    "speaker": os.getenv("TTS_SPEAKER", "xenia"),
    "speed": float(os.getenv("TTS_SPEED", "1.3")),  # 0.5 - 2.0
    "sample_rate": 48000,
}

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Глобальные переменные для моделей
whisper_model = None
tts_model = None
bot_info = None


# Callback data для inline кнопок
class VoiceCallback(CallbackData, prefix="voice"):
    speaker: str


class SpeedCallback(CallbackData, prefix="speed"):
    value: str


def is_admin(user_id: int) -> bool:
    """Проверка, является ли пользователь администратором."""
    return user_id == ADMIN_ID


def load_whisper_model():
    """Загрузка модели Whisper для распознавания речи."""
    global whisper_model
    if whisper_model is None:
        logger.info(f"Загрузка модели Whisper ({WHISPER_MODEL})...")
        whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
        logger.info("Модель Whisper успешно загружена!")
    return whisper_model


def load_tts_model():
    """Загрузка модели Silero TTS для синтеза речи."""
    global tts_model
    if tts_model is None:
        logger.info("Загрузка модели Silero TTS...")
        device = torch.device('cpu')
        tts_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru'
        )
        tts_model.to(device)
        logger.info("Модель Silero TTS успешно загружена!")
    return tts_model


def transcribe_audio(audio_path: str) -> str:
    """Распознавание речи из аудио файла с помощью Whisper."""
    model = load_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    return " ".join(text_parts).strip()


def text_to_speech(text: str, output_path: str) -> bool:
    """Синтез речи из текста с помощью Silero TTS."""
    try:
        model = load_tts_model()

        audio = model.apply_tts(
            text=text,
            speaker=tts_settings["speaker"],
            sample_rate=tts_settings["sample_rate"]
        )

        import torchaudio
        torchaudio.save(output_path, audio.unsqueeze(0), tts_settings["sample_rate"])

        return True
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}")
        return False


def convert_wav_to_ogg(wav_path: str, ogg_path: str, speed: float = 1.0) -> bool:
    """Конвертация WAV в OGG с изменением скорости."""
    try:
        # atempo поддерживает значения от 0.5 до 2.0
        # Для значений вне этого диапазона нужно применять фильтр несколько раз
        speed = max(0.5, min(2.0, speed))

        cmd = [
            "ffmpeg", "-i", wav_path,
            "-filter:a", f"atempo={speed}",
            "-acodec", "libopus",
            "-b:a", "64k",
            "-y",
            ogg_path
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка конвертации в OGG: {e.stderr.decode()}")
        return False


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


def is_bot_mentioned(message: types.Message, bot_username: str) -> bool:
    """Проверка, упомянут ли бот в сообщении."""
    if not message.text:
        return False

    text_lower = message.text.lower()

    if f"@{bot_username.lower()}" in text_lower:
        return True

    if message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mention = message.text[entity.offset:entity.offset + entity.length]
                if mention.lower() == f"@{bot_username.lower()}":
                    return True

    return False


def extract_text_for_tts(message: types.Message, bot_username: str) -> str:
    """Извлечение текста для озвучки (без упоминания бота)."""
    text = message.text or ""
    text = re.sub(rf'@{re.escape(bot_username)}\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура настроек."""
    buttons = [
        [InlineKeyboardButton(text="🎙 Сменить голос", callback_data="menu_voice")],
        [InlineKeyboardButton(text="⚡ Сменить скорость", callback_data="menu_speed")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_voice_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора голоса."""
    buttons = []
    for voice_id, voice_name in AVAILABLE_VOICES.items():
        marker = "✅ " if voice_id == tts_settings["speaker"] else ""
        buttons.append([
            InlineKeyboardButton(
                text=f"{marker}{voice_name}",
                callback_data=VoiceCallback(speaker=voice_id).pack()
            )
        ])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_speed_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора скорости."""
    speeds = ["0.8", "1.0", "1.2", "1.3", "1.5", "1.7", "2.0"]
    buttons = []
    row = []
    for speed in speeds:
        marker = "✅" if float(speed) == tts_settings["speed"] else ""
        row.append(InlineKeyboardButton(
            text=f"{marker}{speed}x",
            callback_data=SpeedCallback(value=speed).pack()
        ))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_current_settings_text() -> str:
    """Текст с текущими настройками."""
    voice_name = AVAILABLE_VOICES.get(tts_settings["speaker"], tts_settings["speaker"])
    return (
        f"⚙️ **Настройки TTS**\n\n"
        f"🎙 Голос: **{voice_name}**\n"
        f"⚡ Скорость: **{tts_settings['speed']}x**"
    )


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Обработка команды /start."""
    await message.answer(
        "👋 Привет! Я ZalupAI Bot.\n\n"
        "🎤 **Расшифровка голосовых:**\n"
        "Я автоматически расшифровываю голосовые сообщения в текст с пунктуацией.\n\n"
        "🔊 **Озвучка текста:**\n"
        "Перешли мне текстовое сообщение и упомяни меня (@) — я озвучу его голосом диктора.\n\n"
        "Добавь меня в групповой чат (не забудь отключить Privacy Mode через @BotFather).",
        parse_mode=ParseMode.MARKDOWN
    )


@dp.message(Command("settings"))
async def cmd_settings(message: types.Message):
    """Команда настроек (только для админа)."""
    if not is_admin(message.from_user.id):
        await message.reply("⛔ Только администратор может менять настройки.")
        return

    await message.answer(
        get_current_settings_text(),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_settings_keyboard()
    )


@dp.callback_query(F.data == "menu_voice")
async def callback_menu_voice(callback: types.CallbackQuery):
    """Меню выбора голоса."""
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админа", show_alert=True)
        return

    await callback.message.edit_text(
        "🎙 **Выберите голос:**",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_voice_keyboard()
    )
    await callback.answer()


@dp.callback_query(F.data == "menu_speed")
async def callback_menu_speed(callback: types.CallbackQuery):
    """Меню выбора скорости."""
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админа", show_alert=True)
        return

    await callback.message.edit_text(
        f"⚡ **Выберите скорость:**\n\nТекущая: {tts_settings['speed']}x",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_speed_keyboard()
    )
    await callback.answer()


@dp.callback_query(F.data == "menu_back")
async def callback_menu_back(callback: types.CallbackQuery):
    """Возврат в главное меню настроек."""
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админа", show_alert=True)
        return

    await callback.message.edit_text(
        get_current_settings_text(),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_settings_keyboard()
    )
    await callback.answer()


@dp.callback_query(VoiceCallback.filter())
async def callback_set_voice(callback: types.CallbackQuery, callback_data: VoiceCallback):
    """Установка голоса."""
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админа", show_alert=True)
        return

    tts_settings["speaker"] = callback_data.speaker
    voice_name = AVAILABLE_VOICES.get(callback_data.speaker, callback_data.speaker)

    await callback.message.edit_text(
        f"✅ Голос изменён на **{voice_name}**",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_settings_keyboard()
    )
    await callback.answer(f"Голос: {voice_name}")
    logger.info(f"Админ сменил голос на: {callback_data.speaker}")


@dp.callback_query(SpeedCallback.filter())
async def callback_set_speed(callback: types.CallbackQuery, callback_data: SpeedCallback):
    """Установка скорости."""
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админа", show_alert=True)
        return

    tts_settings["speed"] = float(callback_data.value)

    await callback.message.edit_text(
        f"✅ Скорость изменена на **{callback_data.value}x**",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_settings_keyboard()
    )
    await callback.answer(f"Скорость: {callback_data.value}x")
    logger.info(f"Админ сменил скорость на: {callback_data.value}")


@dp.message(Command("voice"))
async def cmd_voice(message: types.Message):
    """Команда для озвучки текста после команды."""
    text = message.text.replace("/voice", "").strip()

    if not text:
        await message.reply("Напиши текст после команды: `/voice Привет, мир!`", parse_mode=ParseMode.MARKDOWN)
        return

    await synthesize_and_send(message, text)


@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """Обработка голосовых сообщений."""
    user = message.from_user
    user_name = format_user_name(user)
    timestamp = datetime.now().strftime("%H:%M:%S")

    logger.info(f"Получено голосовое сообщение от {user_name}")

    processing_msg = await message.reply("🔄 Расшифровываю...")

    try:
        voice = message.voice
        file = await bot.get_file(voice.file_id)

        with NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
            ogg_path = ogg_file.name

        try:
            await bot.download_file(file.file_path, ogg_path)
            text = transcribe_audio(ogg_path)

            if not text:
                text = "[не удалось распознать речь]"

            response = f"🎙 **Голосовой эфир от пользователя ({user_name}) {timestamp}**\n\n{text}"
            await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)
            logger.info(f"Расшифровка завершена: {text[:50]}...")

        finally:
            try:
                os.unlink(ogg_path)
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

        try:
            await bot.download_file(file.file_path, video_path)
            text = transcribe_audio(video_path)

            if not text:
                text = "[не удалось распознать речь]"

            response = f"🎥 **Голосовой эфир от пользователя ({user_name}) {timestamp}**\n\n{text}"
            await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass

    except Exception as e:
        logger.error(f"Ошибка обработки видеосообщения: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")


async def synthesize_and_send(message: types.Message, text: str):
    """Синтез и отправка голосового сообщения."""
    if len(text) > 1000:
        await message.reply("❌ Слишком длинный текст (максимум 1000 символов)")
        return

    if len(text) < 2:
        await message.reply("❌ Слишком короткий текст")
        return

    processing_msg = await message.reply("🔊 Озвучиваю...")

    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        with NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
            ogg_path = ogg_file.name

        try:
            if not text_to_speech(text, wav_path):
                await processing_msg.edit_text("❌ Ошибка синтеза речи")
                return

            if not convert_wav_to_ogg(wav_path, ogg_path, tts_settings["speed"]):
                await processing_msg.edit_text("❌ Ошибка конвертации аудио")
                return

            await processing_msg.delete()

            voice_file = FSInputFile(ogg_path)
            await message.reply_voice(voice_file, caption=f"🔊 {text[:100]}{'...' if len(text) > 100 else ''}")

            logger.info(f"Озвучка завершена: {text[:50]}...")

        finally:
            for path in [wav_path, ogg_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Ошибка озвучки: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")


@dp.message(F.forward_date & F.text)
async def handle_forwarded_text(message: types.Message):
    """Обработка пересланных текстовых сообщений с упоминанием бота."""
    global bot_info

    if not bot_info:
        return

    if not is_bot_mentioned(message, bot_info.username):
        return

    text = extract_text_for_tts(message, bot_info.username)

    if not text:
        await message.reply("❌ Не нашёл текст для озвучки")
        return

    logger.info(f"Озвучка пересланного сообщения: {text[:50]}...")
    await synthesize_and_send(message, text)


@dp.message(F.text)
async def handle_text_with_mention(message: types.Message):
    """Обработка текстовых сообщений с упоминанием бота (reply на текст)."""
    global bot_info

    if not bot_info:
        return

    if not is_bot_mentioned(message, bot_info.username):
        return

    if message.reply_to_message and message.reply_to_message.text:
        text = message.reply_to_message.text
        logger.info(f"Озвучка сообщения по reply: {text[:50]}...")
        await synthesize_and_send(message, text)
        return

    text = extract_text_for_tts(message, bot_info.username)

    if text:
        await synthesize_and_send(message, text)


async def main():
    """Главная функция запуска бота."""
    global bot_info

    logger.info("Запуск ZalupAI Bot...")

    bot_info = await bot.get_me()
    logger.info(f"Бот: @{bot_info.username}")

    if ADMIN_ID:
        logger.info(f"Админ ID: {ADMIN_ID}")
    else:
        logger.warning("ADMIN_ID не установлен! Команда /settings будет недоступна.")

    load_whisper_model()
    load_tts_model()

    logger.info(f"TTS настройки: голос={tts_settings['speaker']}, скорость={tts_settings['speed']}x")
    logger.info("Бот запущен и готов к работе!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
