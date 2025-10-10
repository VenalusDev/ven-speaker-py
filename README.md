# ven-speaker-py

**Форк библиотеки `speakerpy` с модификациями и улучшениями для потоковой работы с аудио и TTS.**

![logo](https://raw.githubusercontent.com/denisxab/speakerpy/main/docs/source/_static/logo315x318.png)


## 🔹 Описание

ven-speaker-py — это Python-библиотека для синтеза речи с модификациями:

- Потоковая передача аудио в байтовом виде.
- Исправленная функция ~to_mp3~->`to_wav`.
- Поддержка гибкой настройки скорости, громкости и sample rate.
- Постепенно добавляются новые функции и улучшения под конкретные задачи.

Библиотека подходит для создания голосовых приложений, озвучивания текста в реальном времени и интеграции с другими проектами.


## Особенности и преимущества SpeakerPy

- **Транскрипция чисел и английских слов**: SpeakerPy способен озвучивать числа и английские слова через транскрипцию.
- **Синтез больших объемов текста**: SpeakerPy может синтезировать большие объемы текста без проблем.
- **Корректное деление текста на куски**: благодаря использованию библиотеки nltk, SpeakerPy корректно делит большие тексты на предложения.
- **Кеширование синтезированного текста**: SpeakerPy синтезирует и хранит текст по кускам, что позволяет кешировать уже ранее синтезированный текст и избежать повторного синтезирования. После первой генерации все последующие обычно занимают менее четверти секунды (имеется в виду не полная генерация, а момент, когда начинает зачитываться речь) на обычном процессоре!
- **Работа в автономном режиме**: SpeakerPy синтезирует звук локально, без подключения к интернету, что позволяет использовать его в автономном режиме.
- **Сохранение синтезированного текста в MP3**: SpeakerPy предоставляет возможность сохранять синтезированный текст в формате MP3.

## ⚙️ Установка через pip

`ven-speaker-py` можно установить с помощью pip:

```bash
pip install venspeakerpy
```

## 🛠️ Основные возможности

- **Преобразование текста в речь.**
- **Потоковая генерация аудио чанков.**
- **Поддержка сохранения в различные форматы, включая mp3.**
- **Настройка скорости, громкости и sample rate.**
- **Интеграция в голосовые боты и приложения реального времени.**

### 🚀 Системные требования

- 2 ГБ ОЗУ
- Наличие процессора :D

## 💡 Основные компоненты

Проект состоит из двух основных компонентов: Lib Speak и Lib SL Text.

### Lib Speak

Lib Speak включает классы `Speaker` и `SpeakerBase` для синтеза речи с использованием моделей Silero. Основные методы класса `Speaker`:

1. `speak`: ~~произнесение текста.~~ Теперь это потоковая передача звука в байтах, что позволяет использовать звук где-то еще (например, для трансляции в Discord)
2. `to_wav`: конвертация текста в аудиофайл wav.

#### 📄 Пример использования функции speak

```python
from lib_speak import Speaker
from lib_sl_text import SeleroText
import sounddevice as sd
import queue
import threading
import numpy as np

speaker = Speaker(model_id="ru_v3", language="ru", speaker="baya", device="cpu")

def text_to_speech(txt: str):
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

    for chunk, text in speaker.speak(text=text.text, sample_rate=48000, speed=1.1, volume=0.7):
        audio_queue.put(chunk)

    audio_queue.put(None)
    player_thread.join()

text = SeleroText("Пример текста для синтеза речи")
text_to_speech(text)

#p.s. я не проверял этот код, просто куски из своего проекта сюда напихал в надежде, что у вас заработает, так что удачи.

```

#### Пример использования функции to_wav

```python
from lib_speak import Speaker

text = SeleroText("Пример текста для синтеза речи")
speaker = Speaker(model_id="ru_v3", language="ru", speaker="aidar", device="cpu")
speaker.to_wav(text=text.text, sample_rate=48000, speed=1.0, volume=1.0, name_text='output', audio_dir='output')
```

### Lib SL Text

Lib SL Text содержит класс `SeleroText` для обработки текста перед синтезом речи. Основной метод класса `SeleroText`:

1. `chunk`: разделение текста на блоки.

#### Использование класса SeleroText

```python
text = SeleroText("Привет, мир!", to_language="ru")
```

Разделение текста на куски с помощью метода `chunk()`:

```python
for chunk in text.chunk():
    print(chunk)
```

## Ресурсы и ссылки

- <a href="https://github.com/snakers4/silero-models/blob/master/models.yml" target="_new">Silero Models на GitHub</a>
- <a href="https://www.sphinx-doc.org/" target="_new">Sphinx</a>
- <a href="https://github.com/readthedocs/sphinx_rtd_theme" target="_new">Sphinx RTD Theme</a>
- <a href="https://readthedocs.org/" target="_new">Read the Docs</a>
