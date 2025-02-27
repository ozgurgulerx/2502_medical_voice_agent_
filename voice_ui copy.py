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
        # don't capture user audio while the system is speaking
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

        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.call_active = True  # We'll use this to end the call gracefully.

    def log(self, message):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t}] {message}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.log(f"Audio stream status: {status}")
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        # Output stream for TTS
        self.streams['output'] = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
        )
        # Input stream for user voice
        self.streams['input'] = sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
            blocksize=4800,
            callback=self.audio_callback
        )
        self.streams['output'].start()
        self.streams['input'].start()

    async def setup_websocket_session(self, ws):
        # Enable input audio transcription with whisper-1
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
                    "Remain professional, calm, and ensure the caller feels supported."),
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 150,
                    "silence_duration_ms": 600
                },
                # crucial for user speech transcription
                "input_audio_transcription": {
                    "model": "whisper-1"  
                }
            }
        }
        await ws.send(json.dumps(session_config))

        while True:
            resp_str = await ws.recv()
            data = json.loads(resp_str)

            if data["type"] == "session.created":
                
                break
            elif data["type"] == "error":
                raise RuntimeError(f"Session error: {data}")

    async def send_message(self, ws, user_text):
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

        # Then ask for a response with TTS
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def send_audio(self, ws, audio_data):
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    def check_goodbye(self, text):
        text_lower = text.lower()
        return ("bye" in text_lower or "goodbye" in text_lower or "good bye" in text_lower)

    def check_all_questions_answered(self, text):
        # naive condition - if the assistant says 'all questions answered', we end
        # adapt logic to your needs.
        if "all your questions" in text.lower() and "answered" in text.lower():
            return True
        return False

    async def handle_response(self, ws):
        """
        Listen for messages:
         - response.audio.delta => TTS
         - response.text.delta => partial text
         - response.text.done => final text
         - conversation.item.created => user or assistant messages
        """
        self.audio_processor.is_speaking = True
        try:
            while True:
                # if user ended the call, break immediately
                if not self.call_active:
                    break

                resp_str = await ws.recv()
                data = json.loads(resp_str)
                msg_type = data.get("type", "")

                # TTS audio
                if msg_type == "response.audio.delta":
                    audio_b64 = data.get("delta", "").strip()
                    if audio_b64:
                        pad = -len(audio_b64) % 4
                        if pad:
                            audio_b64 += "=" * pad
                        try:
                            audio_array = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                            self.streams['output'].write(audio_array)
                        except Exception as e:
                            self.log(f"Audio decode error: {e}")

                # partial text from assistant
                elif msg_type == "response.text.delta":
                    partial_text = data.get("delta", "")
                    if partial_text:
                        self.log(f"[MedAssist partial] {partial_text}")

                # final text from assistant
                elif msg_type == "response.text.done":
                    final_text = data.get("text", "")
                    if final_text:
                        self.log(f"[MedAssist final] {final_text}")
                        if self.check_all_questions_answered(final_text):
                            self.log("Assistant indicated all questions answered. Ending call.")
                            self.call_active = False

                # new conversation item - possibly user STT or assistant messages
                elif msg_type == "conversation.item.created":
                    item = data.get("item", {})
                    role = item.get("role", "")
                    content_list = item.get("content", [])
                    text_content = "".join(c.get("text", "") for c in content_list)

                    if role == "user":
                        # This should be user's transcribed text
                        if text_content:
                            self.log(f"[User STT] {text_content}")
                            if self.check_goodbye(text_content):
                                self.log("User said goodbye. Ending call.")
                                self.call_active = False

                    elif role == "assistant":
                        # Assistant may also produce conversation items
                        if text_content:
                            self.log(f"[MedAssist] {text_content}")

                elif msg_type == "response.done":
                    # end this TTS response chunk
                    break

        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        await self.setup_audio()
        async with websockets.connect(self.url) as ws:
            await self.setup_websocket_session(ws)
            self.log("Session setup complete. Entering main loop.")

            while self.call_active:
                if self.audio_processor.should_process():
                    user_audio = self.audio_processor.reset()
                    self.log("[User] (audio captured)")
                    await self.send_audio(ws, user_audio)
                    await self.handle_response(ws)
                await asyncio.sleep(0.05)

            self.log("Call ended.")

if __name__ == "__main__":
    conv = ConversationSystem()
    asyncio.run(conv.run())
