"""Interactive smoke test for the local venSpeakerPy package."""

import sys

from venSpeakerPy import Speaker


SAMPLE_RATE = 48_000
BLOCKSIZE = 0  # Let PortAudio select the device's optimal output block size.
CACHE_DIR = "tts-cache"
CACHE_SIZE_MB = 1024
WARM_PHRASES = (
    "Привет, чат!",
    "Спасибо за сообщение!",
    "Я тебя услышала.",
    "Рада тебя видеть!",
)


def main() -> int:
    try:
        import numpy as np
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        print(
            "Интерактивное воспроизведение требует установленных numpy, sounddevice "
            "и системной библиотеки PortAudio.",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    print("Загружаю Silero TTS модель (ru_v3, lisya, auto device)...")
    try:
        speaker = Speaker(
            model_id="ru_v3",
            language="ru",
            speaker="lisya",
            device="auto",
            cache_dir=CACHE_DIR,
            cache_size_mb=CACHE_SIZE_MB,
        )
    except Exception as exc:
        print(f"Не удалось загрузить модель: {exc}", file=sys.stderr)
        return 1

    try:
        prepared = speaker.warm_cache(WARM_PHRASES, sample_rate=SAMPLE_RATE)
        print(f"Кеш прогрет: {prepared} фрагментов. Папка: {CACHE_DIR}")
        print("Введите текст для озвучивания. Для выхода введите /exit или нажмите Ctrl+C.")
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
        ) as stream:
            while True:
                try:
                    text = input("Текст> ").strip()
                except EOFError:
                    break
                if text.lower() in {"/exit", "/quit"}:
                    break
                if not text:
                    continue

                try:
                    for audio_bytes, _chunk_text in speaker.speak(
                        text=text,
                        sample_rate=SAMPLE_RATE,
                        volume=1.0,
                        comma_pause_ms=20,
                        sentence_pause_ms=80,
                        norm_speed=1.2,
                    ):
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                        if audio.size:
                            stream.write(audio)
                except Exception as exc:
                    print(f"Не удалось озвучить текст: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nЗавершение.")
    except Exception as exc:
        print(f"Ошибка тестового аудиопотока: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
