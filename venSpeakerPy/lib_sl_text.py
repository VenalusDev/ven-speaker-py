import pathlib
import re
import threading
from string import punctuation, whitespace
from typing import Generator, Literal

import nltk
from num2words import num2words
from transliterate import translit


class SeleroText:
    """Normalize text and split it into chunks suitable for TTS."""

    download_dir = pathlib.Path.home() / ".cache" / "venSpeakerPy" / "nltk"
    _max_chunk_symbols = 1000
    _nltk_lock = threading.Lock()

    def __init__(self, text: str, to_language: Literal["ru", "en"]) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if to_language not in ("ru", "en"):
            raise ValueError("to_language must be 'ru' or 'en'")
        self.text = text
        self.to_language = to_language
        self._ensure_tokenizer()
        self._normal()

    @classmethod
    def _ensure_tokenizer(cls) -> None:
        """Load punkt data from a user cache, downloading only when missing."""
        with cls._nltk_lock:
            download_path = str(cls.download_dir)
            if download_path not in nltk.data.path:
                nltk.data.path.insert(0, download_path)
            try:
                nltk.data.find("tokenizers/punkt_tab/russian/")
            except LookupError:
                cls.download_dir.mkdir(parents=True, exist_ok=True)
                if not nltk.download("punkt_tab", download_dir=str(cls.download_dir), quiet=True):
                    raise RuntimeError("Could not download NLTK punkt_tab tokenizer data")

    def chunk(self) -> Generator[str, None, None]:
        """Yield non-empty text chunks, keeping each chunk under the size limit."""
        for sentence in self._punkt(self.text):
            if sentence:
                for part in self._split_long_string(sentence, self._max_chunk_symbols):
                    if re.search(r"\w", part):
                        yield part

    def _normal(self) -> None:
        if self.to_language == "ru":
            self.text = translit(self.text, self.to_language)
        self.text = self._nums_to_text()

    def _nums_to_text(self) -> str:
        return re.sub(
            r"(\d+)",
            lambda match: num2words(int(match.group(0)), lang=self.to_language),
            self.text,
        )

    @staticmethod
    def _punkt(text: str) -> Generator[str, None, None]:
        for sentence in nltk.sent_tokenize(text, language="russian"):
            yield re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9.,!?;:\-–'\"\s]", "", sentence)

    @staticmethod
    def _split_long_string(text: str, slice_length: int) -> Generator[str, None, None]:
        words = text.split()
        current: list[str] = []
        current_length = 0
        for word in words:
            if current and current_length + 1 + len(word) > slice_length:
                yield " ".join(current)
                current = []
                current_length = 0
            current.append(word)
            current_length += len(word) + (1 if len(current) > 1 else 0)
        if current:
            yield " ".join(current)
