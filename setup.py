from setuptools import setup, find_packages

setup(
    name="ven-speaker-py",
    version="0.1.3",
    author="VenalusDev",
    author_email="Venalusdev@mail.ru",
    description="Python TTS package using Silero models for Russian language",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VenalusDev/ven-speaker-py",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10,<2.0",
        "sounddevice>=0.4.7",
        "soundfile>=0.12.1",
        "torch>=2.1,<3.0",
        "nltk>=3.9,<4.0",
        "num2words>=0.5,<0.6",
        "transliterate>=1.10.2",
        "silero>=0.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
