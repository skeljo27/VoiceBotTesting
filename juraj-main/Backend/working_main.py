# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import tempfile
# import os
# import io
# import asyncio
# import logging
# import uuid

# # === New: Import LLM engine ===
# from llama_cpp import Llama

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="VoiceBot Backend", version="1.0.0")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request models
# class MessageRequest(BaseModel):
#     message: str

# class TTSRequest(BaseModel):
#     text: str

# # Initialize services
# whisper_model = None
# tts_engine = None
# llm = None

# @app.on_event("startup")
# async def startup_event():
#     global whisper_model, tts_engine, llm

#     logger.info("🚀 Starting VoiceBot Backend...")

#     try:
#         import whisper
#         logger.info("📝 Loading Whisper...")
#         whisper_model = whisper.load_model("tiny")
#         logger.info("✅ Whisper loaded!")
#     except Exception as e:
#         logger.error(f"❌ Whisper failed: {e}")

#     try:
#         import pyttsx3
#         logger.info("🔊 Loading TTS...")
#         tts_engine = pyttsx3.init()
#         tts_engine.setProperty('rate', 180)
#         tts_engine.setProperty('volume', 0.9)
#         logger.info("✅ TTS loaded!")
#     except Exception as e:
#         logger.error(f"❌ TTS failed: {e}")

#     try:
#         logger.info("🧠 Loading Mistral LLM model...")
#         llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048, n_threads=4)
#         logger.info("✅ Mistral loaded!")
#     except Exception as e:
#         logger.error(f"❌ Mistral failed to load: {e}")

# from fastapi import FastAPI, UploadFile, File

# import os
# import ffmpeg
# import tempfile
# import whisper
# import logging
# from fastapi import UploadFile, File
# import wave

# # Initialize Whisper model
# model = whisper.load_model("base")  # Or "small", "medium", "large" based on your needs

# # Initialize logger
# logger = logging.getLogger(__name__)

# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     try:
#         logger.info(f"🎧 Received file: {file.filename}")
#         audio_data = await file.read()

#         # Step 1: Save the audio as a temporary .webm file
#         temp_dir = "C:/Users/jskel/Downloads/juraj-main/juraj-main/Backend/temp_files/"
#         os.makedirs(temp_dir, exist_ok=True)

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=temp_dir) as temp_file:
#             temp_file.write(audio_data)
#             temp_path = temp_file.name
#         logger.info(f"📁 Temp file saved at: {temp_path}")

#         # Step 2: Convert the .webm file to .wav using FFmpeg with resampling to 16kHz
#         wav_path = temp_path.replace(".webm", ".wav")
#         logger.info(f"📁 Converting to .wav at: {wav_path}")

#         # Add resampling to 16kHz
#         ffmpeg.input(temp_path).output(wav_path, ar='16000').run()

#         # Step 3: Check if .wav file exists and log its details
#         if not os.path.exists(wav_path):
#             logger.error(f"❌ .wav file does not exist at: {wav_path}")
#             return {"text": "Fehler bei der Transkription: .wav file not found"}

#         logger.info(f"📁 .wav file size: {os.path.getsize(wav_path)} bytes")
#         logger.info(f"📁 Checking if .wav file exists: {os.path.exists(wav_path)}")

#         # Step 4: Ensure that the .wav file has the correct format (RIFF header)
#         with wave.open(wav_path, "rb") as wf:
#             if wf.getframerate() != 16000:  # Whisper expects 16kHz sample rate
#                 raise ValueError("Incorrect sample rate: expected 16kHz")

#         # Step 5: Run Whisper transcription on the .wav file
#         logger.info(f"🧠 Running Whisper transcription on: {wav_path}")
#         result = model.transcribe(wav_path)
#         transcription = result["text"].strip()

#         logger.info(f"✅ Transcription result: {transcription}")
#         os.unlink(temp_path)  # Clean up .webm file
#         os.unlink(wav_path)   # Clean up .wav file
#         return {"text": transcription}
           

        

#     except Exception as e:
#         logger.error(f"❌ Transcription error: {e}")
#         return {"text": f"Fehler bei der Transkription: {str(e)}"}


# @app.post("/respond")
# async def generate_response(request: MessageRequest):
#     global llm
#     try:
#         prompt = f"You are a helpful assistant. Answer the following user query as clearly and concisely as possible.\nUser: {request.message}\nAssistant:"
#         logger.info(f"🧠 Sending prompt to LLM: {request.message[:50]}...")

#         output = llm(prompt, max_tokens=200, stop=["User:", "Assistant:"], echo=False)
#         response_text = output["choices"][0]["text"].strip()

#         logger.info("✅ Response generated.")
#         return {"response": response_text}

#     except Exception as e:
#         logger.error(f"❌ LLM response error: {e}")
#         return {"response": "Entschuldigung, ich konnte gerade keine Antwort generieren."}


# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "services": {
#             "stt": {"status": "running"},
#             "tts": {"status": "running" if tts_engine else "stopped"},
#             "nlp": {"status": "running"}
#         }
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
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
from contextlib import asynccontextmanager

# === New: Import LLM engine ===
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
whisper_model = None
tts_engine = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"🎧 Received file: {file.filename}")
        audio_data = await file.read()

        # Step 1: Save the audio as a temporary file
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        # Determine file extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".webm"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=temp_dir) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        logger.info(f"📁 Temp file saved at: {temp_path}")

        # Step 2: Use Whisper directly (it can handle webm, mp3, wav, etc.)
        logger.info(f"🧠 Running Whisper transcription on: {temp_path}")
        result = model.transcribe(temp_path)
        transcription = result["text"].strip()

        logger.info(f"✅ Transcription result: {transcription}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return {"text": transcription}

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"text": f"Fehler bei der Transkription: {str(e)}"}


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
    """Text-to-speech endpoint"""
    global tts_engine
    try:
        if tts_engine is None:
            return {"error": "TTS engine not available"}
        
        # Create a temporary file for the audio output
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        
        audio_file = os.path.join(temp_dir, f"tts_{uuid.uuid4().hex}.wav")
        
        # Generate speech
        tts_engine.save_to_file(request.text, audio_file)
        tts_engine.runAndWait()
        
        # Return the audio file
        def iterfile():
            with open(audio_file, mode="rb") as file_like:
                yield from file_like
            # Clean up after sending
            if os.path.exists(audio_file):
                os.unlink(audio_file)
        
        return StreamingResponse(iterfile(), media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        return {"error": f"TTS failed: {str(e)}"}

@app.post("/respond")
async def generate_response(request: MessageRequest):
    global llm
    try:
        if llm is None:
            # Fallback: Use a simple rule-based response for testing
            user_message = request.message.lower()
            
            # Simple responses for testing
            if "hello" in user_message or "hi" in user_message:
                response_text = "Hello! How can I help you today?"
            elif "how are you" in user_message:
                response_text = "I'm doing well, thank you for asking!"
            elif "weather" in user_message:
                response_text = "I'm sorry, I don't have access to current weather data."
            elif "time" in user_message:
                from datetime import datetime
                current_time = datetime.now().strftime("%H:%M")
                response_text = f"The current time is {current_time}."
            else:
                response_text = f"You said: '{request.message}'. I understand you, but my language model is currently offline. Please check back later!"
            
            logger.info(f"✅ Fallback response generated: {response_text[:50]}...")
            return {"response": response_text}

        # If LLM is available, use it
        prompt = f"You are a helpful assistant. Answer the following user query as clearly and concisely as possible.\nUser: {request.message}\nAssistant:"
        logger.info(f"🧠 Sending prompt to LLM: {request.message[:50]}...")

        output = llm(prompt, max_tokens=200, stop=["User:", "Assistant:"], echo=False)
        response_text = output["choices"][0]["text"].strip()

        logger.info("✅ Response generated.")
        return {"response": response_text}

    except Exception as e:
        logger.error(f"❌ LLM response error: {e}")
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-07-12",
        "services": {
            "stt": {"status": "running", "model": "whisper-base"},
            "tts": {"status": "running" if tts_engine else "stopped"},
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