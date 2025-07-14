import asyncio
import logging
import tempfile
import os
from typing import Optional
import langdetect
import torch
import torch.serialization

# Coqui globals fix for torch >= 2.6
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig
    ])

    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("⚠️ Coqui TTS not available, using fallback TTS")

# Optional fallback TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("⚠️ pyttsx3 not available")

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, use_coqui: bool = True):
        self.model_name = "xtts_v2"
        self.tts_engine = None
        self.use_coqui = use_coqui and COQUI_AVAILABLE
        self.coqui_tts = None

        if self.use_coqui:
            try:
                self.coqui_tts = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                    gpu=False
                )
                logger.info("✅ Coqui TTS loaded successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to load Coqui TTS models: {e}")
                self.use_coqui = False

        self._initialize_tts()

    def _initialize_tts(self):
        if not self.use_coqui:
            try:
                self._init_fallback_tts()
            except Exception as e:
                logger.error(f"❌ Failed to load fallback TTS: {e}")

    def _init_fallback_tts(self):
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not available")

        logger.info("🔄 Loading pyttsx3 TTS...")
        self.tts_engine = pyttsx3.init()

        voices = self.tts_engine.getProperty('voices')
        german_voice = next((v for v in voices if 'german' in v.name.lower()), None)
        if german_voice:
            self.tts_engine.setProperty('voice', german_voice.id)

        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)
        self.model_name = "pyttsx3-fallback"
        logger.info("✅ pyttsx3 TTS loaded")

    async def synthesize(self, text: str, language: Optional[str] = None) -> bytes:
        if not language:
            language = self._detect_language(text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text, language)

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        if self.coqui_tts:
            return self._synthesize_coqui(text, language)
        elif self.tts_engine:
            return self._synthesize_pyttsx3(text)
        else:
            raise RuntimeError("No TTS engine available")

    def _synthesize_coqui(self, text: str, language: str) -> bytes:
        speaker_wav_map = {
            "de": "de_sample.wav",
            "en": "en_sample.wav"
        }
        speaker_wav_path = speaker_wav_map.get(language)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.coqui_tts.tts_to_file(
                text=text,
                language=language,
                speaker_wav=speaker_wav_path,
                file_path=temp_path
            )
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _synthesize_pyttsx3(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _detect_language(self, text: str) -> str:
        try:
            lang = langdetect.detect(text)
            return {
                'de': 'de', 'en': 'en', 'fr': 'fr', 'es': 'es',
                'it': 'it', 'nl': 'nl', 'pl': 'pl', 'pt': 'pt', 'ru': 'ru'
            }.get(lang, 'en')
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {e}, defaulting to 'de'")
            return 'de'

    def cleanup(self):
        try:
            if self.tts_engine:
                self.tts_engine.stop()
                self.tts_engine = None
            if self.coqui_tts:
                self.coqui_tts = None
            logger.info("🧹 TTS service cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ TTS cleanup warning: {e}")

    def __del__(self):
        self.cleanup()
