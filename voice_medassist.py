import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

class AudioProcessor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.vad_threshold = 0.015
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_duration = int(0.3 * sample_rate)
        self.max_silence_duration = int(0.8 * sample_rate)
        self.buffer = []
        self.is_speaking = False
        self.speech_detected = False

    def process_audio(self, indata):
        # If TTS playing, skip capturing user audio
        if self.is_speaking:
            return
        audio_level = np.abs(indata).mean() / 32768.0
        if audio_level > self.vad_threshold:
            self.speech_detected = True
            self.speech_frames += len(indata)
            self.silence_frames = 0
            self.buffer.extend(indata.tobytes())
        elif self.speech_detected:
            self.silence_frames += len(indata)
            if self.silence_frames < self.max_silence_duration:
                self.buffer.extend(indata.tobytes())

    def should_process(self):
        return (
            self.speech_detected
            and self.speech_frames >= self.min_speech_duration
            and self.silence_frames >= self.max_silence_duration
        )

    def reset(self):
        audio_data = bytes(self.buffer)
        self.buffer.clear()
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        return audio_data

class ConversationSystem:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found")

        # Websocket endpoint
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}")
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        self.streams['output'] = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16
        )
        self.streams['input'] = sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
            callback=self.audio_callback,
            blocksize=4800
        )
        self.streams['output'].start()
        self.streams['input'].start()

    async def setup_websocket_session(self, websocket):
        """
        Start a new session. We'll also send an initial user message once we see 'session.created'.
        """
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": (
                    "You are a formal and professional medical call agent named 'MedAssist.' "
                    "IMPORTANT: When responding to a greeting or initial contact, do NOT use simple responses like 'yes' or 'I am'. "
                    "Instead, always begin with: 'This is MedAssist, a medical call service. How may I assist you with your healthcare needs today?' "
                    "Maintain an empathetic, respectful tone and carefully avoid casual slang, expletives, jokes, or informal language. "
                    "Keep explanations concise and relevant to healthcare. "
                    "When appropriate, ask clarifying questions to identify the caller's intent:\n"
                    " - New symptoms (caller seeking care for a newly arising complaint).\n"
                    " - Existing condition (caller has an ongoing issue or known diagnosis).\n"
                    " - Other medical reasons (e.g., test results, referrals, or administrative inquiries).\n"
                    " - Medical emergency (urgent or serious symptoms—this leads to immediate escalation).\n"
                    "Once you identify the intent, route the conversation or proceed with the correct flow. "
                    "Remain professional, calm, and ensure the caller feels supported."
                ),
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 150,
                    "silence_duration_ms": 600
                }
            }
        }
        await websocket.send(json.dumps(session_config))

        # Wait for session creation
        while True:
            resp_str = await websocket.recv()
            data = json.loads(resp_str)
            if data["type"] == "session.created":
                print("Session created! We'll send an initial user greeting now.")
                # Let's greet as if the user said "Hello..."
                await self.send_message(websocket, "Hello, I'm calling for medical help. Are you the medical call agent?")
                # We'll handle the TTS response
                await self.handle_response(websocket)
                break
            elif data["type"] == "error":
                raise Exception(f"Session error: {data}")

    async def send_message(self, ws, text):
        payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await ws.send(json.dumps(payload))
        # Then request TTS
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def send_audio(self, ws, audio_data):
        # Send user audio for STT
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, ws):
        """
        Process server messages: TTS data, user messages, etc.
        We break once "response.done" => done with TTS chunk.
        """
        # Mute user capturing while TTS plays
        self.audio_processor.is_speaking = True
        try:
            while True:
                resp_str = await ws.recv()
                data = json.loads(resp_str)
                msg_type = data.get("type", "")

                if msg_type == "response.audio.delta":
                    audio_b64 = data.get("delta", "").strip()
                    if audio_b64:
                        # Fix padding
                        pad = -len(audio_b64) % 4
                        if pad:
                            audio_b64 += "=" * pad
                        try:
                            audio_decoded = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(audio_decoded, dtype=np.int16)
                            self.streams['output'].write(audio_array)
                        except Exception as e:
                            print(f"Audio decode error: {e}")

                elif msg_type == "conversation.item.created":
                    # This can be user speech STT or assistant text
                    item = data.get("item", {})
                    role = item.get("role", "")
                    content_list = item.get("content", [])
                    text_content = "".join(c.get("text", "") for c in content_list)

                    # If user says something, we see it transcribed here
                    if role == "user":
                        print(f"[User STT] {text_content}")

                elif msg_type == "response.done":
                    # Done with TTS chunk
                    break

        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        await self.setup_audio()
        async with websockets.connect(self.url) as ws:
            # Setup session => send initial user greeting
            await self.setup_websocket_session(ws)
            print("Session setup complete. Begin main loop...")

            while True:
                # If user has spoken enough => send audio to server
                if self.audio_processor.should_process():
                    captured = self.audio_processor.reset()
                    await self.send_audio(ws, captured)
                    # Now wait for TTS
                    await self.handle_response(ws)

                await asyncio.sleep(0.05)

if __name__ == "__main__":
    conv = ConversationSystem()
    asyncio.run(conv.run())
