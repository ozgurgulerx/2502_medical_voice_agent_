import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from typing import Optional, Any, List

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
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.buffer)
        self.buffer.clear()
        return audio_data

class VoiceUI:
    """
    Interface for the Real-Time API that handles voice input/output.
    """
    def __init__(self):
        """
        Initialize the Voice UI with the Azure OpenAI Real-Time API.
        """
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")

        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.websocket = None
        self.conversation_history = []
        self.current_response_text = ""
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}")
        self.audio_processor.process_audio(indata)
        
    async def initialize(self):
        """
        Initialize audio streams and WebSocket connection.
        """
        # Setup audio devices
        self.streams['output'] = sd.OutputStream(
            samplerate=24000, channels=1, dtype=np.int16
        )
        self.streams['input'] = sd.InputStream(
            samplerate=24000, channels=1, dtype=np.int16,
            callback=self.audio_callback, blocksize=4800
        )
        for s in self.streams.values():
            s.start()
            
        # Connect to WebSocket
        self.websocket = await websockets.connect(self.url)
        
        # Initialize session without the pre-programmed assistant message
        await self._setup_websocket_session()
        
    async def _setup_websocket_session(self):
        """
        Set up the WebSocket session with the Real-Time API.
        This configures the session without any specific instructions
        since we'll be using AutoGen for that.
        """
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "",  # Empty instructions - we'll use AutoGen agents
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
        
        while True:
            resp_str = await self.websocket.recv()
            data = json.loads(resp_str)
            if data["type"] == "session.created":
                print("Real-Time API session created.")
                break
            elif data["type"] == "error":
                raise Exception(f"Session error: {data}")
    
    async def listen(self) -> str:
        """
        Listen for user speech input and return the transcribed text.
        """
        print("🎤 Listening for user input...")
        
        # Reset the speech buffer in case there's anything leftover
        self.audio_processor.reset()
        
        # Wait until we detect complete speech
        while not self.audio_processor.should_process():
            await asyncio.sleep(0.05)
            
        # Get the audio data
        audio_data = self.audio_processor.reset()
        
        # Send to the API for transcription
        transcription = await self._get_transcription(audio_data)
        
        # Store in conversation history
        if transcription:
            self.conversation_history.append({"role": "user", "content": transcription})
            print(f"User said: {transcription}")
            
        return transcription
    
    async def _get_transcription(self, audio_data) -> str:
        """
        Send audio to the API and get back the transcription.
        """
        try:
            # Send the audio buffer
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            await self.websocket.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }))
            await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            # Request a text-only response (we don't want audio back for transcription)
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["text"]}
            }))
            
            # Get the transcription
            transcription = ""
            while True:
                message_raw = await self.websocket.recv()
                data = json.loads(message_raw)
                
                if data["type"] == "response.text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        transcription += delta
                
                elif data["type"] == "response.done":
                    break
                    
            return transcription.strip()
            
        except Exception as e:
            print(f"Error getting transcription: {e}")
            return ""
        
    async def speak(self, text: str) -> None:
        """
        Convert text to speech and play it through the Real-Time API.
        """
        if not text:
            return
            
        # Store in conversation history
        self.conversation_history.append({"role": "assistant", "content": text})
        
        print("\n🔊 Assistant says:")
        print(text)
        
        # Format text for voice - remove markdown, code blocks, etc.
        voice_text = self._format_for_voice(text)
        
        try:
            # Create a message from the assistant
            payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": voice_text}]
                }
            }
            await self.websocket.send(json.dumps(payload))
            
            # Request audio response
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["audio"]}
            }))
            
            # Play the audio response
            self.audio_processor.is_speaking = True
            try:
                while True:
                    message_raw = await self.websocket.recv()
                    data = json.loads(message_raw)
                    
                    if data["type"] == "response.audio.delta":
                        audio_b64 = data.get("delta", "").strip()
                        if audio_b64:
                            # handle padding
                            pad = -len(audio_b64) % 4
                            if pad: 
                                audio_b64 += "=" * pad
                            try:
                                audio_array = np.frombuffer(
                                    base64.b64decode(audio_b64), dtype=np.int16
                                )
                                self.streams['output'].write(audio_array)
                            except Exception as e:
                                print(f"Audio decode error: {e}")
                    
                    elif data["type"] == "response.done":
                        break
            finally:
                self.audio_processor.is_speaking = False
                
        except Exception as e:
            print(f"Error in speak: {e}")
    
    def _format_for_voice(self, text: str) -> str:
        """
        Format text to be more suitable for voice output.
        """
        # Remove markdown formatting, code blocks, etc.
        voice_text = text
        voice_text = voice_text.replace('*', '')
        voice_text = voice_text.replace('#', '')
        voice_text = voice_text.replace('```', '')
        
        # Remove any code blocks
        lines = voice_text.split('\n')
        filtered_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                filtered_lines.append(line)
        
        voice_text = '\n'.join(filtered_lines)
        
        return voice_text
    
    def interrupt(self) -> None:
        """
        Interrupt current speech output, if any.
        """
        self.audio_processor.is_speaking = False
    
    def get_conversation_history(self):
        """
        Return the conversation history.
        """
        return self.conversation_history
    
    async def close(self):
        """
        Close the WebSocket connection and audio streams.
        """
        if self.websocket:
            await self.websocket.close()
        
        for stream in self.streams.values():
            if stream and stream.active:
                stream.stop()
                stream.close()


# Simple test function
async def test_voice_ui():
    voice_ui = VoiceUI()
    await voice_ui.initialize()
    
    try:
        await voice_ui.speak("Hello, I'm your medical assistant. How can I help you today?")
        user_input = await voice_ui.listen()
        await voice_ui.speak(f"You said: {user_input}. Let me help you with that.")
    finally:
        await voice_ui.close()

if __name__ == "__main__":
    # Test the VoiceUI class
    asyncio.run(test_voice_ui())