from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import io
import asyncio
import logging
import uuid

# === New: Import LLM engine ===
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceBot Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class MessageRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

# Initialize services
whisper_model = None
tts_engine = None
llm = None

@app.on_event("startup")
async def startup_event():
    global whisper_model, tts_engine, llm

    logger.info("🚀 Starting VoiceBot Backend...")

    try:
        import whisper
        logger.info("📝 Loading Whisper...")
        whisper_model = whisper.load_model("tiny")
        logger.info("✅ Whisper loaded!")
    except Exception as e:
        logger.error(f"❌ Whisper failed: {e}")

    try:
        import pyttsx3
        logger.info("🔊 Loading TTS...")
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 180)
        tts_engine.setProperty('volume', 0.9)
        logger.info("✅ TTS loaded!")
    except Exception as e:
        logger.error(f"❌ TTS failed: {e}")

    try:
        logger.info("🧠 Loading Mistral LLM model...")
        llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048, n_threads=4)
        logger.info("✅ Mistral loaded!")
    except Exception as e:
        logger.error(f"❌ Mistral failed to load: {e}")

from fastapi import FastAPI, UploadFile, File

import os
import ffmpeg
import tempfile
import whisper
import logging
from fastapi import UploadFile, File
import wave

# Initialize Whisper model
model = whisper.load_model("base")  # Or "small", "medium", "large" based on your needs

# Initialize logger
logger = logging.getLogger(__name__)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"🎧 Received file: {file.filename}")
        audio_data = await file.read()

        # Step 1: Save the audio as a temporary .webm file
        temp_dir = "C:/Users/jskel/Downloads/juraj-main/juraj-main/Backend/temp_files/"
        os.makedirs(temp_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=temp_dir) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        logger.info(f"📁 Temp file saved at: {temp_path}")

        # Step 2: Convert the .webm file to .wav using FFmpeg with resampling to 16kHz
        wav_path = temp_path.replace(".webm", ".wav")
        logger.info(f"📁 Converting to .wav at: {wav_path}")

        # Add resampling to 16kHz
        ffmpeg.input(temp_path).output(wav_path, ar='16000').run()

        # Step 3: Check if .wav file exists and log its details
        if not os.path.exists(wav_path):
            logger.error(f"❌ .wav file does not exist at: {wav_path}")
            return {"text": "Fehler bei der Transkription: .wav file not found"}

        logger.info(f"📁 .wav file size: {os.path.getsize(wav_path)} bytes")
        logger.info(f"📁 Checking if .wav file exists: {os.path.exists(wav_path)}")

        # Step 4: Ensure that the .wav file has the correct format (RIFF header)
        with wave.open(wav_path, "rb") as wf:
            if wf.getframerate() != 16000:  # Whisper expects 16kHz sample rate
                raise ValueError("Incorrect sample rate: expected 16kHz")

        # Step 5: Run Whisper transcription on the .wav file
        logger.info(f"🧠 Running Whisper transcription on: {wav_path}")
        result = model.transcribe(wav_path)
        transcription = result["text"].strip()

        logger.info(f"✅ Transcription result: {transcription}")
        os.unlink(temp_path)  # Clean up .webm file
        os.unlink(wav_path)   # Clean up .wav file
        return {"text": transcription}
           

        

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return {"text": f"Fehler bei der Transkription: {str(e)}"}


@app.post("/respond")
async def generate_response(request: MessageRequest):
    global llm
    try:
        prompt = f"You are a helpful assistant. Answer the following user query as clearly and concisely as possible.\nUser: {request.message}\nAssistant:"
        logger.info(f"🧠 Sending prompt to LLM: {request.message[:50]}...")

        output = llm(prompt, max_tokens=200, stop=["User:", "Assistant:"], echo=False)
        response_text = output["choices"][0]["text"].strip()

        logger.info("✅ Response generated.")
        return {"response": response_text}

    except Exception as e:
        logger.error(f"❌ LLM response error: {e}")
        return {"response": "Entschuldigung, ich konnte gerade keine Antwort generieren."}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": {"status": "running"},
            "tts": {"status": "running" if tts_engine else "stopped"},
            "nlp": {"status": "running"}
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
