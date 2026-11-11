import pathlib
import json
import re
import threading
import time
from typing import Generator, Iterable, Literal

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from silero import silero_tts

from .lib_cache import AudioCache
from .lib_helper import timeit
from .lib_sl_text import SeleroText


class SpeakerBase:
    """Base Silero TTS model wrapper."""

    _th_lock_speak = threading.Lock()

    def __init__(
        self,
        model_id: str,
        language: Literal["ru", "en"],
        speaker: str,
        device: str = "auto",
    ) -> None:
        self.language = language
        self.model_id = model_id
        if speaker.lower() == "lisya":
            self.voice_profile = "lisya"
            self.speaker = "baya"
            self.default_speed = 1.1
        else:
            self.voice_profile = None
            self.speaker = speaker
            self.default_speed = 1.0
        self.device = self._resolve_device(device)
        self.model = self._init_models(model_id=model_id, language=language, device=self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a CUDA GPU")
        return device

    @staticmethod
    def _init_models(model_id: str, language: Literal["ru", "en"] = "ru", device: str = "cpu"):
        target_device = torch.device(device)
        model, *_ = silero_tts(language=language, speaker=model_id, device=target_device)
        # Silero v3 returns a Python wrapper around a JIT module and ignores the
        # device argument while loading the packaged model. Move both explicitly.
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, torch.nn.Module):
            inner_model.to(target_device)
            # Silero ru_v3 creates accentor inputs on CPU and expects its
            # accentor module there; the acoustic model and vocoder use CUDA.
            accentor = getattr(inner_model, "accentor", None)
            if target_device.type == "cuda" and isinstance(accentor, torch.nn.Module):
                accentor.to("cpu")
        elif isinstance(model, torch.nn.Module):
            model.to(target_device)
        if hasattr(model, "device"):
            model.device = target_device
        return model

    def _synthesize_text(
        self, text: str, sample_rate: int, put_accent: bool = True, put_yo: bool = True
    ) -> torch.Tensor:
        return self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
        )

    def _speak(self, th_name: int, text: str, audio: torch.Tensor, sample_rate: int, speed: float = 1.0):
        try:
            import sounddevice
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Direct playback requires sounddevice and the system PortAudio library; "
                "PCM chunk generation does not."
            ) from exc

        playback_rate = sample_rate * speed
        with self._th_lock_speak:
            sounddevice.play(audio, samplerate=playback_rate)
            time.sleep(len(audio) / playback_rate + 0.02)
            sounddevice.stop()


class Speaker(SpeakerBase):
    def __init__(
        self,
        model_id: str,
        language: Literal["ru", "en"],
        speaker: str,
        device: str = "auto",
        *,
        cache_dir: str | pathlib.Path | None = None,
        cache_size_mb: float = 512,
    ) -> None:
        super().__init__(model_id=model_id, language=language, speaker=speaker, device=device)
        if cache_dir is None:
            cache_dir = pathlib.Path.home() / ".cache" / "venSpeakerPy" / "audio"
        self.cache = AudioCache(cache_dir, max_size_mb=cache_size_mb)

    def _get_audio(self, text: str, sample_rate: int, put_accent: bool, put_yo: bool) -> np.ndarray:
        key = self.cache.make_key(
            "v1", self.model_id, self.language, self.speaker, self.device,
            text, sample_rate, put_accent, put_yo,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode():
            tensor = self._synthesize_text(text, sample_rate, put_accent, put_yo)
        audio = tensor.detach().to("cpu").numpy().astype(np.float32, copy=False)
        self.cache.put(key, audio)
        return audio

    @staticmethod
    def _split_cacheable_pieces(text: str) -> list[str]:
        return [piece for piece in re.findall(r"[^,.;:!?]+[,.;:!?]*|[,.;:!?]+", text) if piece.strip()]

    def warm_cache(
        self,
        texts: Iterable[str] | str,
        sample_rate: int = 48_000,
        *,
        put_accent: bool = True,
        put_yo: bool = False,
    ) -> int:
        """Synthesize likely-to-repeat phrases ahead of time and store their audio.

        Returns the number of phrase fragments checked. Existing cache entries are
        reused, so this can safely run at each bot startup.
        """
        if isinstance(texts, str):
            texts = (texts,)
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        prepared = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("warm_cache expects strings")
            for sentence in SeleroText(text, to_language=self.language).chunk():
                for piece in self._split_cacheable_pieces(sentence):
                    self._get_audio(piece.strip(), sample_rate, put_accent, put_yo)
                    prepared += 1
        return prepared

    def speak(
        self,
        text: str,
        sample_rate: int,
        speed: float | None = None,
        *,
        norm_speed: float = 1.0,
        volume: float = 1.0,
        put_accent: bool = True,
        put_yo: bool = False,
        normalize: bool = False,
        limiting: bool = False,
        comma_pause_ms: int = 0,
        sentence_pause_ms: int = 0,
    ) -> Generator[tuple[bytes, str], None, None]:
        """Yield float32 PCM chunks and their text.

        ``speed`` changes playback rate and pitch. ``norm_speed`` changes duration
        with WSOLA, preserving pitch and waveform details more closely. Pause values are extra
        silence in milliseconds after punctuation; zero keeps the stream compact.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if speed is None:
            speed = self.default_speed
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a finite positive number")
        if not np.isfinite(norm_speed) or norm_speed <= 0:
            raise ValueError("norm_speed must be a finite positive number")
        if not np.isfinite(volume) or not (0.0 <= volume <= 2.0):
            raise ValueError("volume must be between 0.0 and 2.0")
        if comma_pause_ms < 0 or sentence_pause_ms < 0:
            raise ValueError("pause durations must be non-negative")

        chunks = SeleroText(text, to_language=self.language).chunk()
        for chunk_text in chunks:
            for piece in self._split_cacheable_pieces(chunk_text):
                audio = self._get_audio(piece.strip(), sample_rate, put_accent, put_yo)
                if speed != 1:
                    audio = resample_poly(audio, up=100, down=max(1, round(100 * speed)))
                if norm_speed != 1:
                    audio = self._time_stretch(audio, norm_speed)
                processed = audio * (volume / 5.0)
                if normalize:
                    processed = self.normalize(processed)
                if limiting:
                    processed = self.limiter(processed)
                if processed.size:
                    yield np.asarray(processed, dtype=np.float32).tobytes(), piece

                ending = piece.rstrip()[-1:] if piece.rstrip() else ""
                pause_ms = sentence_pause_ms if ending in ".!?" else comma_pause_ms if ending in ",;:" else 0
                if pause_ms:
                    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
                    yield pause.tobytes(), ""

    @timeit
    def to_wav(
        self,
        text: str,
        name_text: str,
        sample_rate: int,
        audio_dir: pathlib.Path | str,
        speed: float | None = None,
        norm_speed: float = 1.0,
        volume: float = 1.0,
        put_accent: bool = True,
        comma_pause_ms: int = 0,
        sentence_pause_ms: int = 0,
    ) -> pathlib.Path:
        if not name_text or pathlib.Path(name_text).name != name_text or name_text in (".", ".."):
            raise ValueError("name_text must be a filename without directory components")
        if speed is None:
            speed = self.default_speed
        audio_dir = pathlib.Path(audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_file = audio_dir / f"{name_text.replace(' ', '_')}.wav"
        metadata_file = output_file.with_suffix(".wav.json")
        expected_metadata = {
            "text": text,
            "model_id": self.model_id,
            "language": self.language,
            "speaker": self.speaker,
            "sample_rate": sample_rate,
            "speed": speed,
            "norm_speed": norm_speed,
            "volume": volume,
            "put_accent": put_accent,
            "comma_pause_ms": comma_pause_ms,
            "sentence_pause_ms": sentence_pause_ms,
        }
        if output_file.is_file() and metadata_file.is_file():
            try:
                if json.loads(metadata_file.read_text(encoding="utf-8")) == expected_metadata:
                    return output_file
            except (OSError, json.JSONDecodeError):
                pass

        pcm_chunks = (chunk for chunk, _ in self.speak(
            text=text, sample_rate=sample_rate, speed=speed, volume=volume,
            put_accent=put_accent, put_yo=True,
            norm_speed=norm_speed,
            comma_pause_ms=comma_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
        ))
        audio_array = np.frombuffer(b"".join(pcm_chunks), dtype=np.float32)
        with soundfile.SoundFile(output_file, "w", samplerate=sample_rate, channels=1, subtype="PCM_16") as stream:
            stream.write(audio_array)
        metadata_file.write_text(json.dumps(expected_metadata, ensure_ascii=False), encoding="utf-8")
        return output_file

    @staticmethod
    def normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return audio / peak * target if peak > 0 else audio

    @staticmethod
    def limiter(audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        return np.clip(audio, -threshold, threshold) / threshold

    @staticmethod
    def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
        """Change duration with WSOLA while keeping pitch and waveform detail."""
        if rate == 1 or audio.size < 64:
            return audio
        expected_length = max(1, round(audio.size / rate))
        window = min(1024, max(128, 2 ** int(np.floor(np.log2(audio.size)))))
        overlap = window // 2
        synthesis_hop = window - overlap
        search_radius = min(128, max(16, overlap // 4))
        search_step = 4
        padded_audio = np.pad(audio.astype(np.float32, copy=False), (0, window))
        output = np.zeros(expected_length + window, dtype=np.float32)
        output[:window] = padded_audio[:window]

        output_position = synthesis_hop
        expected_input_position = max(1, round(synthesis_hop * rate))
        fade = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)

        while output_position + window <= output.size:
            target = output[output_position:output_position + overlap]
            center = expected_input_position
            first = max(0, center - search_radius)
            last = min(len(padded_audio) - window, center + search_radius)
            candidates = range(first, last + 1, search_step)

            target_centered = target - target.mean()
            target_norm = float(np.linalg.norm(target_centered))
            best_position = min(max(center, first), last)
            best_score = -float("inf")
            for candidate_position in candidates:
                candidate = padded_audio[candidate_position:candidate_position + overlap]
                candidate_centered = candidate - candidate.mean()
                denominator = target_norm * float(np.linalg.norm(candidate_centered))
                score = float(np.dot(target_centered, candidate_centered)) / denominator if denominator else 0.0
                if score > best_score:
                    best_score = score
                    best_position = candidate_position

            frame = padded_audio[best_position:best_position + window]
            output[output_position:output_position + overlap] = (
                target * (1.0 - fade) + frame[:overlap] * fade
            )
            output[output_position + overlap:output_position + window] = frame[overlap:]

            output_position += synthesis_hop
            expected_input_position = round(expected_input_position + synthesis_hop * rate)

        return output[:expected_length]
