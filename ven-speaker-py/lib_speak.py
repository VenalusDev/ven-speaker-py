import pathlib
import threading
import time
from typing import Generator, Literal
from scipy.signal import resample_poly

import sounddevice
import soundfile
import torch
from silero import silero_tts
import numpy as np

from lib_helper import timeit
from lib_sl_text import SeleroText


class SpeakerBase:
    #: Для синхронизации потоков озвучки текста
    _th_lock_speak = threading.Lock()

    def __init__(
            self,
            model_id,
            language: Literal["ru", "en"],
            speaker: Literal["aidar", "baya", "kseniya", "xenia", "random"],
            device="cpu",
            ) -> None:
        self.language = language
        self.model_id = model_id
        # Модель
        self.model = self._init_models(
            model_id=model_id,
            language=language,
            device=device,
            )
        # Голос
        self.speaker = speaker

    @staticmethod
    def _init_models(
            model_id: str,
            language: Literal["ru"] = "ru",
            device: Literal[
                "cpu",
                "cuda",
                "ipu",
                "xpu",
                "mkldnn",
                "opengl",
                "opencl",
                "ideep",
                "hip",
                "ve",
                "fpga",
                "ort",
                "xla",
                "lazy",
                "vulkan",
                "mps",
            ] = "cpu",
            ):
        """
        Инициализация ИИ модели

        model_id: Доступные модели https://github.com/snakers4/silero-models/blob/master/models.yml (четверку тут использовать не получится)
        """
        _device = torch.device(device)
        _model, *_any = silero_tts(language=language, speaker=model_id, device=_device)
        return _model

    def _synthesize_text(
            self, text: str, sample_rate: int, put_accent: bool = True, put_yo: bool = True
            ) -> torch.Tensor:
        """Синтезировать текст"""
        audio: torch.Tensor = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
            )
        return audio

    def _speak(
            self,
            th_name: int,
            text: str,
            audio: torch.Tensor,
            sample_rate: int,
            speed: float = 1.0,
            ):
        """Озвучить текст

        :param audio: Аудио
        :param sample_rate: Частота
        :param speed: Скорость воспроизведения [от 1.0 до 2.0]
        """
        _speed = sample_rate * speed
        # Блокировка потока. Может говорить только один поток
        with self._th_lock_speak:
            # print(f"Th: {th_name}\t| {text}")
            sounddevice.play(audio, samplerate=_speed)
            time.sleep((len(audio) / _speed + 0.02))
            sounddevice.stop()


class Speaker(SpeakerBase):
    def speak(
            self,
            text: str,
            sample_rate: int,
            speed: float = 1.0,
            *,
            volume: float = 1.0,
            put_accent: bool = True,
            put_yo: bool = False,
            normalize: bool = False,
            limiting: bool = False,
            ) -> Generator[tuple[bytes, str], None, None]:
        """
        Генерация и озвучка текста с поддержкой тегов <long:x:duration>

        :param text: Текст для озвучки (поддерживает <long:буква:время>)
        :param sample_rate: Частота дискретизации
        :param speed: Скорость воспроизведения
        :param volume: Громкость (0.0 - 2.0)
        :param put_accent: Автоматическая постановка ударения
        :param put_yo: Автоматическая постановка буквы ё
        :param normalize: Нормализация звука (вряд ли вообще понадобится, но пусть будет)
        :param limiting: Ограничение звука (вряд ли вообще понадобится, но пусть будет)
        :return: Чанки аудио в формате bytes
        """
        if not (0.0 <= volume <= 2.0):
            raise ValueError("Параметр volume должен быть в диапазоне от 0.0 до 2.0")
        sl_text = SeleroText(text, to_language=self.language)

        # Генерация чанков с выводом байтов по мере готовности
        chunks = list(sl_text.chunk())
        for i, _chunk_text in enumerate(chunks):
            audio: torch.Tensor = self._synthesize_text(
                _chunk_text,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo,
                )
            scaled_audio = (audio.numpy() / 5 * volume).astype(np.float32)

            adjusted_audio = resample_poly(scaled_audio, up=100, down=int(100 * speed)) if speed != 1 else scaled_audio

            normal = self.normalize(adjusted_audio) if normalize else adjusted_audio
            limited = self.limiter(normal) if limiting else normal

            if len(limited) > 0:
                yield adjusted_audio.tobytes(), _chunk_text

                # Если это последний чанк — добавим тишину
            if i == len(chunks) - 1:
                silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)  # 150 мс тишины
                yield silence.tobytes(), ""

    @timeit
    def to_wav(
            self,
            text: str,
            name_text: str,
            sample_rate: int,
            audio_dir: pathlib.Path | str,
            speed: float = 1.0,
            volume: float = 1.0,
            put_accent=True
            ) -> pathlib.Path:
        """
        Синтезировать и сохранить звук в WAV файл

        :param text: Текст
        :param name_text: Имя текста (используется в имени файла)
        :param sample_rate: Частота дискретизации
        :param audio_dir: Куда сохранить файл
        :param speed: Скорость речи
        :param volume: Громкость
        :return: Путь к сохраненному WAV-файлу
        """
        name_text = name_text.replace(" ", "_")
        audio_dir = pathlib.Path(audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        output_file = audio_dir / (
                name_text
                + ".wav"
        )

        if output_file.exists():
            print("Cache___Wav: 0 \t| ", output_file)
            return output_file

        full_audio = b''.join(self.speak(text=text, sample_rate=sample_rate, speed=speed, volume=volume, put_accent=put_accent, put_yo=True))

        # Сохраняем как WAV
        with soundfile.SoundFile(output_file, 'w', samplerate=sample_rate, channels=1, subtype='PCM_16') as f:
            audio_array = np.frombuffer(full_audio, dtype=np.float32)
            f.write(audio_array)

        print("Build___Wav: 0 \t| ", output_file)
        return output_file

    def normalize(self, audio, target=0.95):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * target
        return audio

    def limiter(self, audio, threshold=0.9):
        audio = np.clip(audio, -threshold, threshold)
        return audio / threshold


if __name__ == '__main__':
    import sounddevice as sd
    import queue

    audio_queue = queue.Queue()


    def audio_player():
        try:
            with sd.OutputStream(samplerate=48000, channels=1, dtype='float32') as stream:
                while True:
                    speech_chunk = audio_queue.get()
                    if speech_chunk is None:
                        break
                    audio_array = np.frombuffer(speech_chunk, dtype=np.float32)
                    stream.write(audio_array)
        except KeyboardInterrupt:
            print("Stop speaker")
        finally:
            sd.stop()


    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()
    speaker = Speaker(model_id="ru_v3", language="ru", speaker="baya", device="cpu")
    while True:
        text = input('Text: ')
        text = text.replace('ё', 'е')

        if text.startswith('!w '):
            text = text.replace('!w ', '')
            speaker.to_wav(text=text.text, sample_rate=48000, speed=1.1, volume=0.7, name_text='output', audio_dir='output')
        else:
            for chunk, text in speaker.speak(text=text.text, sample_rate=48000, speed=1.1, volume=0.7):
                audio_queue.put(chunk)
