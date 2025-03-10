import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
import datetime

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
        # Don't capture user audio when assistant is speaking
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
        # Return the audio so we can send it to the server
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.buffer)
        self.buffer.clear()
        return audio_data

class ConversationSystem:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found")

        # AOAI real-time endpoint
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}

    def log(self, msg):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t}] {msg}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.log(f"Audio stream status: {status}")
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        # Create output stream for TTS
        self.streams['output'] = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16
        )
        # Create input stream for user voice
        self.streams['input'] = sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
            blocksize=4800,
            callback=self.audio_callback,
        )
        # Start both streams
        self.streams['output'].start()
        self.streams['input'].start()

    async def setup_websocket_session(self, websocket):
        """
        We explicitly say:
          - you are a medical call agent
          - produce both audio and text
          - do not mention Dr Smith or booking appointments unless specifically asked
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
                    " - Medical emergency (urgent or serious symptoms—this leads to a roadblock and immediate escalation).\n"
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

        while True:
            resp_str = await websocket.recv()
            data = json.loads(resp_str)

            if data["type"] == "session.created":
                self.log("Session created. Sending initial greeting.")
                # We'll treat it as if user said hi.
                await self.send_message(websocket, "Hello, I'm calling for medical help. Are you the medical call agent?")
                # We'll handle the immediate TTS response from that.
                await self.handle_response(websocket)
                break
            elif data["type"] == "error":
                raise RuntimeError(f"Session error: {data}")

    async def send_message(self, ws, user_text):
        self.log(f"[User] {user_text}")
        payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text}
                ]
            }
        }
        await ws.send(json.dumps(payload))

        # Then ask the server to generate audio+text
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def send_audio(self, ws, audio_data):
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        # Append user audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # Then ask for TTS + text
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, ws):
        """
        Listen for:
         - response.audio.delta => TTS chunks
         - response.text.delta => partial text from assistant
         - response.text.done => final text from assistant
         - conversation.item.created => new item from user or assistant
        """
        self.audio_processor.is_speaking = True
        try:
            while True:
                resp_str = await ws.recv()
                data = json.loads(resp_str)

                # TTS audio
                if data.get("type") == "response.audio.delta":
                    audio_b64 = data.get("delta", "").strip()
                    if audio_b64:
                        pad = -len(audio_b64) % 4
                        if pad:
                            audio_b64 += "=" * pad
                        try:
                            audio_array = np.frombuffer(
                                base64.b64decode(audio_b64), dtype=np.int16
                            )
                            self.streams['output'].write(audio_array)
                        except Exception as e:
                            self.log(f"Audio decode error: {e}")

                # Assistant partial text
                elif data.get("type") == "response.text.delta":
                    partial_text = data.get("delta", "")
                    if partial_text:
                        self.log(f"[MedAssist partial] {partial_text}")

                # Assistant final text
                elif data.get("type") == "response.text.done":
                    final_text = data.get("text", "")
                    if final_text:
                        self.log(f"[MedAssist final] {final_text}")

                # If new conversation item created (often user or system message)
                elif data.get("type") == "conversation.item.created":
                    item = data.get("item", {})
                    role = item.get("role", "")
                    content_list = item.get("content", [])
                    # Possibly show transcription if user is recognized
                    if role == "user":
                        # This might be the user transcription text if server's STT is on
                        text_content = "".join(c.get("text", "") for c in content_list)
                        if text_content:
                            self.log(f"[User STT] {text_content}")
                    elif role == "assistant":
                        # Or any new assistant item
                        text_content = "".join(c.get("text", "") for c in content_list)
                        if text_content:
                            self.log(f"[MedAssist] {text_content}")

                elif data.get("type") == "response.done":
                    # This signals the end of this response chunk
                    break

                # else, you might see other message types
        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        await self.setup_audio()
        async with websockets.connect(self.url) as ws:
            # 1) Setup session => TTS and text
            await self.setup_websocket_session(ws)
            self.log("Session setup complete. Entering main loop...")

            # 2) Loop, check if user done speaking, then send audio, handle response
            while True:
                if self.audio_processor.should_process():
                    audio_data = self.audio_processor.reset()
                    # show we got user audio
                    self.log("[User] (audio captured)")
                    await self.send_audio(ws, audio_data)
                    await self.handle_response(ws)
                await asyncio.sleep(0.05)

if __name__ == "__main__":
    conv = ConversationSystem()
    asyncio.run(conv.run())
