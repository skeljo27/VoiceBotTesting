from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
import io
import asyncio
import logging
import uuid
import numpy as np
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
    
# === New: Import LLM engine ===
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === TTS ===
from tts_service import TTSService

# === STT ===
from voice_transcriber import transcribe_from_microphone

# Global variables for services
whisper_model = None
tts_service = TTSService(use_coqui=True)
llm = None

# === RAG System ===
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import pickle
import faiss

MAX_TOKENS = 200
SIMILARITY_THRESHOLD = 0.6
MAX_CONTEXT_CHARS = 1000

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/docs.pkl", "rb") as f:
    docs_raw = pickle.load(f)

docs = [d["page_content"] for d in docs_raw if "page_content" in d]
all_intents = list(set(
    d["metadata"]["intent"]
    for d in docs_raw
    if "metadata" in d and "intent" in d["metadata"]
))

index = faiss.read_index("data/index.faiss")

    

def embed(text: str) -> np.ndarray:
    return embedding_model.encode(text)

def detect_intent(text: str, intents: list[str]) -> str | None:
    vec = embed(text)
    best_intent = None
    best_score = -1
    for intent in intents:
        intent_vec = embed(intent)
        sim = np.dot(vec, intent_vec) / (np.linalg.norm(vec) * np.linalg.norm(intent_vec))
        if sim > best_score and sim > SIMILARITY_THRESHOLD:
            best_score = sim
            best_intent = intent
    return best_intent

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """Very basic DE/EN language detection"""
    german_keywords = ["der", "die", "das", "ein", "und", "nicht", "Sie", "ich", "wie", "für", "Antrag", "Reisepass"]
    count = sum(1 for word in german_keywords if word.lower() in text.lower())
    return "de" if count >= 2 else "en"

def rag_generate_response(user_input: str) -> str:
    language = detect_language(user_input)
    logger.info(f"🌍 Detected language: {language}")

    vector = embed(user_input)
    intent = detect_intent(user_input, all_intents)

    matched_chunks = []
    if intent:
        filtered_docs = [d for d in docs_raw if d.get("metadata", {}).get("intent") == intent]
        for d in filtered_docs:
            vec = embed(d["page_content"])
            sim = np.dot(vector, vec) / (np.linalg.norm(vector) * np.linalg.norm(vec))
            if sim > SIMILARITY_THRESHOLD:
                matched_chunks.append(d["page_content"][:MAX_CONTEXT_CHARS])

    if not matched_chunks:
        D, I = index.search(np.array([vector]), k=3)
        for rank, idx in enumerate(I[0]):
            if D[0][rank] > SIMILARITY_THRESHOLD and idx < len(docs):
                doc = docs[idx]
                text = doc[:MAX_CONTEXT_CHARS] if isinstance(doc, str) else doc.get("page_content", "")
                matched_chunks.append(text.strip())

    context = "\n\n".join(matched_chunks[:3])

    # Prompt instructions per language
    if language == "de":
        instruction = (
            "Du bist ein hilfreicher Sprachassistent für Bürgerdienste in Karlsruhe.\n"
            "Antworte klar, einfach und in 1 bis 3 kurzen Sätzen. Vermeide juristische Begriffe.\n"
            "Sprich auf Deutsch.\n"
        )
    else:
        instruction = (
            "You are a helpful voice assistant for municipal services in Karlsruhe.\n"
            "Answer clearly and briefly in 1 to 3 sentences. Avoid legal terms.\n"
            "Speak in English.\n"
        )

    prompt_text = (
        f"{instruction}\n"
        f"Question: {user_input}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    try:
        output = llm(prompt_text, max_tokens=MAX_TOKENS, stop=["User:", "Assistant:"], echo=False)
        response_text = output["choices"][0]["text"].strip()
        return response_text
    except Exception as e:
        logger.error(f"❌ RAG response error: {e}")
        return "Entschuldigung, ich konnte gerade keine Antwort generieren." if language == "de" else "Sorry, I couldn’t generate an answer right now."

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global whisper_model, tts_engine, llm, tts_service
    
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

    # Check if model file exists and is not corrupted
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    # Alternative smaller models to try if the main one fails
    alternative_models = [
        "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "models/phi-2.Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ]
    
    model_loaded = False
    for model_path in alternative_models:
        if os.path.exists(model_path):
            try:
                logger.info(f"🧠 Loading LLM model from: {model_path}")
                llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
                logger.info("✅ LLM model loaded successfully!")
                model_loaded = True
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {model_path}: {e}")
                continue
        else:
            logger.warning(f"❌ Model file not found at: {model_path}")
    
    if not model_loaded:
        logger.warning("❌ No working model found. LLM responses will be disabled.")
        logger.warning("Please download a working model file or check the model path.")
        llm = None

    yield
    
    # Shutdown
    logger.info("🛑 Shutting down VoiceBot Backend...")

app = FastAPI(title="VoiceBot Backend", version="1.0.0", lifespan=lifespan)

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

# Initialize Whisper model for transcription endpoint
import whisper
model = whisper.load_model("base")  # Or "small", "medium", "large" based on your needs

@app.post("/transcribe")
async def transcribe_audio_from_mic():
    """
    STT directly from microphone using voice_transcriber.py
    """
    try:
        text = transcribe_from_microphone()
        return {"text": text}
    except Exception as e:
        logger.error(f"❌ STT error: {e}")
        return {"error": str(e)}

@app.post("/transcribe_with_ffmpeg")
async def transcribe_audio_with_ffmpeg(file: UploadFile = File(...)):
    """Alternative endpoint that uses FFmpeg conversion"""
    try:
        logger.info(f"🎧 Received file: {file.filename}")
        audio_data = await file.read()

        # Step 1: Save the audio as a temporary .webm file
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=temp_dir) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        logger.info(f"📁 Temp file saved at: {temp_path}")

        # Step 2: Convert the .webm file to .wav using FFmpeg with resampling to 16kHz
        wav_path = temp_path.replace(".webm", ".wav")
        logger.info(f"📁 Converting to .wav at: {wav_path}")

        try:
            import ffmpeg
            # Add resampling to 16kHz
            (
                ffmpeg
                .input(temp_path)
                .output(wav_path, ar='16000')
                .run(overwrite_output=True, quiet=True)
            )
        except Exception as ffmpeg_error:
            logger.error(f"❌ FFmpeg conversion failed: {ffmpeg_error}")
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return {"text": f"Fehler bei der Audio-Konvertierung: {str(ffmpeg_error)}"}

        # Step 3: Check if .wav file exists and log its details
        if not os.path.exists(wav_path):
            logger.error(f"❌ .wav file does not exist at: {wav_path}")
            return {"text": "Fehler bei der Transkription: .wav file not found"}

        logger.info(f"📁 .wav file size: {os.path.getsize(wav_path)} bytes")

        # Step 4: Ensure that the .wav file has the correct format (RIFF header)
        try:
            import wave
            with wave.open(wav_path, "rb") as wf:
                if wf.getframerate() != 16000:  # Whisper expects 16kHz sample rate
                    logger.warning(f"Sample rate is {wf.getframerate()}Hz, expected 16kHz")
        except Exception as wave_error:
            logger.warning(f"Wave file check failed: {wave_error}")

        # Step 5: Run Whisper transcription on the .wav file
        logger.info(f"🧠 Running Whisper transcription on: {wav_path}")
        result = model.transcribe(wav_path)
        transcription = result["text"].strip()

        logger.info(f"✅ Transcription result: {transcription}")
        
        # Clean up files
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        
        return {"text": transcription}

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return {"text": f"Fehler bei der Transkription: {str(e)}"}


@app.post("/tts")        
async def text_to_speech(request: TTSRequest):
    """Text-to-speech endpoint using Coqui TTS"""
    try:
        audio_data = await tts_service.synthesize(request.text)
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="temp_files") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        return FileResponse(tmp_path, media_type="audio/wav", filename="response.wav")
    
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        return {"error": f"TTS failed: {str(e)}"}


@app.post("/respond")
async def generate_response(request: MessageRequest):
    try:
        response_text = rag_generate_response(request.message)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"❌ RAG response error: {e}")
        return {"response": "Entschuldigung, ich konnte gerade keine Antwort generieren."}




@app.get("/")
async def root():
    """Root endpoint to handle requests to localhost:8000/"""
    return {
        "message": "VoiceBot Backend is running!",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/transcribe": "Audio transcription",
            "/respond": "LLM response generation",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": {"status": "running", "model": "whisper-base"},
            "tts": {"status": "running" if tts_service else "stopped", "engine": tts_service.get_model_info()["engine"]},
            "nlp": {"status": "running" if llm else "stopped"}
        }
    }


@app.get("/status")
async def status_check():
    """Alternative status endpoint"""
    return {"status": "online", "backend": "running"}


if __name__ == "__main__":
    import uvicorn
    import socket
    
    def is_port_available(port):
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False
    
    # Try different ports if 8000 is occupied
    available_port = None
    for port in [8000, 8001, 8002, 8003, 8004, 8005]:
        if is_port_available(port):
            available_port = port
            break
        else:
            logger.warning(f"Port {port} is already in use, trying next port...")
    
    if available_port:
        logger.info(f"🚀 Starting server on port {available_port}")
        uvicorn.run(app, host="0.0.0.0", port=available_port, log_level="info")
    else:
        logger.error("Could not find an available port to run the server.")