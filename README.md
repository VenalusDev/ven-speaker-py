# venSpeakerPy

Python-библиотека для синтеза русской речи с помощью моделей Silero. Поддерживает потоковую выдачу PCM-чанков, сохранение речи в WAV, постоянный дисковый кеш и выбор CPU/CUDA.

## Возможности

- Потоковая генерация аудио: чанки можно сразу передавать в аудиопоток.
- Постоянный LRU-кеш синтезированных фрагментов. Повторные реплики читаются с диска, в том числе после перезапуска.
- Предварительный прогрев часто используемых реплик через `warm_cache()`.
- Профиль `lisya`: голос Silero `baya` со скоростью `1.1`, если параметр `speed` не задан.
- Автоматический выбор CUDA при наличии CUDA-сборки PyTorch и совместимого GPU; иначе используется CPU.
- Управление скоростью, громкостью, нормализованной скоростью и паузами после пунктуации.
- Сохранение результата в WAV.

## Установка

Установить опубликованную версию из PyPI:

```bash
python -m pip install venSpeakerPy
```

Для разработки клонируйте репозиторий и установите его в editable-режиме:

```bash
git clone https://github.com/VenalusDev/ven-speaker-py.git
cd ven-speaker-py
python -m pip install -e ".[test]"
```

## Системные требования

- Python 3.11 или новее. Проверенная конфигурация — Windows x64 и Python 3.13.
- Процессор: наличие процессора :D (CPU x64; видеокарта не обязательна).
- Рекомендуется от 4 ГБ оперативной памяти. Для длинных текстов и одновременной работы других приложений понадобится больше.
- Место на диске для PyTorch, Silero-модели и кеша. Лимит аудиокеша по умолчанию — 512 МБ. CUDA-сборка PyTorch требует несколько гигабайт свободного места на время установки.
- Аудиоустройство требуется для прямого воспроизведения. Если приложение само обрабатывает PCM-байты, аудиоустройство не обязательно.
- Интернет нужен при первой загрузке Silero-модели и данных токенизатора NLTK. После загрузки компонентов синтез выполняется локально.

NVIDIA GPU и CUDA-сборка PyTorch опциональны. На Windows установите CUDA-сборку PyTorch после библиотеки. Проверенная конфигурация — PyTorch 2.14.1 с CUDA 13.0:

```powershell
python -m pip install --upgrade "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

Проверьте доступность CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`device="auto"` выбирает CUDA, если она доступна, или CPU в противном случае. Для принудительного выбора передайте `device="cuda"` или `device="cpu"`.

## Быстрый старт

```python
from venSpeakerPy import Speaker

speaker = Speaker(
    model_id="ru_v3",
    language="ru",
    speaker="lisya",
    device="auto",
)

for audio_bytes, text_chunk in speaker.speak(
    "Привет, чат! Как дела?",
    sample_rate=48_000,
):
    # audio_bytes — mono PCM float32; передайте байты в свой аудиоплеер.
    print(text_chunk)
```

Генератор возвращает пары `(audio_bytes, text_chunk)`. Для воспроизведения через `sounddevice` держите `OutputStream` открытым между сообщениями:

```python
import numpy as np
import sounddevice as sd

with sd.OutputStream(samplerate=48_000, channels=1, dtype="float32", blocksize=0) as stream:
    for audio_bytes, _ in speaker.speak("Привет, чат!", sample_rate=48_000):
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        if audio.size:
            stream.write(audio)
```

В корне проекта есть интерактивный пример: `python test_speaker.py`. Он принимает текст с клавиатуры, использует GPU автоматически и хранит кеш в локальной папке `tts-cache/`.

## Скорость и паузы

`speed` меняет темп воспроизведения вместе с высотой голоса. Для `lisya` по умолчанию используется `speed=1.1`; явно переданный `speed` переопределяет профиль.

`norm_speed` меняет длительность с помощью WSOLA, стараясь сохранить высоту и тембр исходной записи. Например, `norm_speed=1.1` ускоряет речь, а `norm_speed=0.9` замедляет. Сильное изменение темпа может давать слышимые артефакты.

`comma_pause_ms` и `sentence_pause_ms` добавляют тишину после запятых и знаков конца предложения. По умолчанию паузы равны нулю:

```python
for audio_bytes, _ in speaker.speak(
    "Привет, чат! Спасибо за сообщение.",
    sample_rate=48_000,
    norm_speed=1.1,
    comma_pause_ms=80,
    sentence_pause_ms=180,
):
    ...
```

Для управления паузами фраза синтезируется по фрагментам, разделённым пунктуацией. Это позволяет независимо вставлять тишину после знаков.

## Постоянный кеш и частые реплики

По умолчанию кеш хранится в `~/.cache/venSpeakerPy/audio`, его лимит — 512 МБ. Путь и объём можно настроить при создании `Speaker`:

```python
speaker = Speaker(
    model_id="ru_v3",
    language="ru",
    speaker="lisya",
    device="auto",
    cache_dir="./tts-cache",
    cache_size_mb=1024,
)
```

Кешируются результаты синтеза фрагментов, разделённых пунктуацией. Ключ записи учитывает модель, язык, голос, устройство, текст фрагмента, частоту дискретизации и параметры ударения. Скорость и громкость применяются после чтения кеша, поэтому не создают копии записи. При повторном доступе запись перемещается в конец LRU-очереди; при достижении лимита вытесняются давно не использовавшиеся файлы. Значение `cache_size_mb=0` отключает запись в кеш.

Для Twitch и других сценариев с повторяющимися репликами можно заранее синтезировать часто используемые фразы:

```python
frequent_replies = [
    "Привет, чат!",
    "Спасибо за сообщение!",
    "Я тебя услышала.",
]
prepared_fragments = speaker.warm_cache(frequent_replies, sample_rate=48_000)
print(f"Проверено фрагментов: {prepared_fragments}")
```

`warm_cache()` пропускает повторный синтез записей, которые уже есть на диске. Новые предложения не собираются из слов или токенов: кеш переиспользует только совпавшие аудиофрагменты, а новые фрагменты синтезируются Silero.

## Сохранение в WAV

```python
wav_path = speaker.to_wav(
    text="Привет, чат!",
    name_text="greeting",
    sample_rate=48_000,
    audio_dir="output",
    norm_speed=1.1,
)
print(wav_path)
```

## Лицензия

MIT. См. файл [LICENSE](LICENSE).
