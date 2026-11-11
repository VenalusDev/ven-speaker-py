import numpy as np

from venSpeakerPy.lib_cache import AudioCache


def test_audio_cache_survives_reopening(tmp_path):
    audio = np.linspace(-1, 1, 128, dtype=np.float32)
    key = AudioCache.make_key("ru_v3", "Привет!")

    first_run = AudioCache(tmp_path, max_size_mb=1)
    first_run.put(key, audio)

    second_run = AudioCache(tmp_path, max_size_mb=1)
    np.testing.assert_array_equal(second_run.get(key), audio)


def test_cache_hit_refreshes_lru_position(tmp_path):
    cache = AudioCache(tmp_path, max_size_mb=1)
    audio = np.arange(1000, dtype=np.float32)
    first, second, third = (AudioCache.make_key(value) for value in ("first", "second", "third"))

    cache.put(first, audio)
    cache.put(second, audio)
    item_size = (tmp_path / f"{first}.npy").stat().st_size
    cache.get(first)
    cache.max_size_bytes = item_size * 2
    cache.put(third, audio)

    assert (tmp_path / f"{first}.npy").exists()
    assert not (tmp_path / f"{second}.npy").exists()
    assert (tmp_path / f"{third}.npy").exists()
