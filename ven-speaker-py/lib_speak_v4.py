import numpy as np
from TTS.api import TTS
from lib_sl_text import SeleroText
import sounddevice as sd
import threading
import queue
import soundfile
import librosa

'''
WORK IN PROGRESS!
'''

class SpeakerV4:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="cpu", speaker_wav=None,
                 language="ru"):
        self.tts = TTS(model_name=model_name, progress_bar=False)
        self.tts.to(device)
        self.language = language
        self.speaker_wav = speaker_wav  # путь к .wav с голосом
        self.default_speaker = "random" if speaker_wav is None else None

    def _synthesize(self, text, speed=1.0, volume=1.0):
        # Генерация аудио
        wav = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            speaker=self.default_speaker,
            language=self.language
        )

        # Изменим громкость и скорость при необходимости
        wav = np.array(wav)
        wav = librosa.effects.time_stretch(wav.astype(np.float32), rate=speed)
        wav *= volume
        return wav

    def speak(
            self,
            text: str,
            sample_rate: int = 22050,
            speed: float = 1.0,
            volume: float = 1.0
    ):
        sl_text = SeleroText(text, to_language="ru")
        for chunk in sl_text.chunk():
            audio = self._synthesize(chunk, speed=speed, volume=volume)
            yield audio.tobytes()
        # Добавим немного тишины в конце
        yield (np.zeros(int(sample_rate * 0.25), dtype=np.float32)).tobytes()

    def to_wav(self, text: str, path: str, sample_rate: int = 22050, speed: float = 1.0, volume: float = 1.0):
        chunks = b''.join(self.speak(text=text, sample_rate=sample_rate, speed=speed, volume=volume))
        arr = np.frombuffer(chunks, dtype=np.float32)
        soundfile.write(path, arr, samplerate=sample_rate, subtype='PCM_16')


if __name__ == "__main__":
    import queue

    audio_queue = queue.Queue()


    def audio_player():
        with sd.OutputStream(samplerate=48000, channels=1, dtype='float32') as stream:
            while True:
                speech_chunk = audio_queue.get()
                if speech_chunk is None:
                    break
                audio_array = np.frombuffer(speech_chunk, dtype=np.float32)
                stream.write(audio_array)


    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()
    speaker = SpeakerV4(device="cuda", speaker_wav=r'F:\PycharmProjects\venSpeakerPy\output\output.wav')
    text = "Привет Это тест."
    for chunk in speaker.speak(text, speed=0.6, volume=0.7):
        audio_queue.put(chunk)
