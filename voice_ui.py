import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

###############################################################################
# AudioProcessor: handles the local Voice Activity Detection (VAD)
###############################################################################
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
        # If we're currently TTS-ing, don't record input
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
            # keep capturing some extra bytes to ensure tail end
            if self.silence_frames < self.max_silence_duration:
                self.buffer.extend(indata.tobytes())

    def should_process(self):
        return (
            self.speech_detected
            and self.speech_frames >= self.min_speech_duration
            and self.silence_frames >= self.max_silence_duration
        )

    def reset(self):
        """Retrieves the recorded audio buffer and resets counters."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.buffer)
        self.buffer.clear()
        return audio_data

###############################################################################
# VoiceUI: main class to handle microphone input + TTS output via Real-Time API
###############################################################################
class VoiceUI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables.")

        # Adjust the region and deployment name to match your resource
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )

        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.websocket = None

    async def initialize(self):
        """
        1) Start mic (input) + speaker (output) streams
        2) Connect to Real-Time API WebSocket
        3) Minimal session config so we can do:
           - listen() for user speech -> text
           - speak() for text -> TTS
        """
        # Start local audio streams
        self.streams['output'] = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.streams['input'] = sd.InputStream(
            samplerate=24000, channels=1, dtype=np.int16,
            callback=self._audio_callback, blocksize=4800
        )
        for s in self.streams.values():
            s.start()

        # Connect the websocket
        self.websocket = await websockets.connect(self.url)
        await self._setup_session()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}")
        self.audio_processor.process_audio(indata)

    async def _setup_session(self):
        """
        Minimal instructions so the Real-Time API won't do any conversation logic.
        We'll handle that in Python with classification + specialist.
        """
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "",
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
        await self.websocket.send(json.dumps(session_config))

        # Wait for session.created
        while True:
            resp_str = await self.websocket.recv()
            data = json.loads(resp_str)
            if data["type"] == "session.created":
                print("Real-Time API session created.")
                break
            elif data["type"] == "error":
                raise RuntimeError(f"Session error: {data}")

    async def listen(self) -> str:
        """
        Wait until we detect a chunk of speech, then transcribe it to text.
        """
        while not self.audio_processor.should_process():
            await asyncio.sleep(0.05)

        audio_data = self.audio_processor.reset()
        return await self._transcribe(audio_data)

    async def _transcribe(self, audio_data: bytes) -> str:
        """
        Send user audio to Real-Time for text transcription, and return the final text.
        """
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        # Step 1: feed to input buffer
        await self.websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        # Step 2: commit
        await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # Step 3: request text
        await self.websocket.send(json.dumps({
            "type":"response.create",
            "response":{"modalities":["text"]}
        }))

        transcript = ""
        while True:
            raw = await self.websocket.recv()
            data = json.loads(raw)
            if data["type"] == "response.text.delta":
                transcript += data.get("delta","")
            elif data["type"] == "response.done":
                break
        return transcript.strip()

    async def speak(self, text: str):
        """
        Convert 'text' -> TTS with Real-Time API.
        """
        if not text:
            return
        print(f"[Assistant -> {text}]")

        # Create assistant message
        item_msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type":"text","text":text}]
            }
        }
        await self.websocket.send(json.dumps(item_msg))

        # Request audio
        await self.websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities":["audio"]}
        }))

        self.audio_processor.is_speaking = True
        try:
            while True:
                msg_raw = await self.websocket.recv()
                data = json.loads(msg_raw)
                if data["type"] == "response.audio.delta":
                    audio_b64 = data.get("delta","").strip()
                    if audio_b64:
                        pad = -len(audio_b64) % 4
                        if pad: 
                            audio_b64 += "=" * pad
                        audio_array = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                        self.streams['output'].write(audio_array)
                elif data["type"] == "response.done":
                    break
        finally:
            self.audio_processor.is_speaking = False

    async def close(self):
        """
        Close the WebSocket and audio streams
        """
        if self.websocket:
            await self.websocket.close()
        for s in self.streams.values():
            if s and s.active:
                s.stop()
                s.close()

###############################################################################
# Simple local test
###############################################################################
if __name__ == "__main__":
    async def run_test():
        voice_ui = VoiceUI()
        await voice_ui.initialize()
        print("Voice UI is active. Speak and it will attempt to transcribe your speech to text.")
        try:
            while True:
                user_text = await voice_ui.listen()
                if user_text:
                    print(f"User said: {user_text}")
                    # If you want to TTS a response, do:
                    # final_answer = "Hello from the Python side!"
                    # await voice_ui.speak(final_answer)
        except KeyboardInterrupt:
            print("Exiting.")
        finally:
            await voice_ui.close()

    asyncio.run(run_test())
