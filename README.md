# Live Voice Analysis Toolkit (Version 2.0)

### Your swiss-army knife for live voice analysis.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [How it Works](#how-it-works)
- [The Helper Class](#the-helper-class)

## Introduction
This project is a pipeline that is able to take audio from the microphone and analyze the speakers voice live. It transcribes the voice, using OpenAI's [Whisper](https://github.com/openai/whisper) model, using the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementation which has been adapted for live transcription. The project also uses SpeechBrain's [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) model to generate speaker embeddings to distinguish speakers from each other.

## Requirements
This project was developed and tested on Python 3.11.7 on Windows 11 with access to an Nvidia GPU with CUDA. I cannot guarantee that it will work on other operating systems or other versions of Python. While CUDA is not required, it is highly recommended.
You will need the following libraries:

- langcodes
- librosa
- numpy
- sounddevice
- torch
- denoiser
- faster_whisper
- scipy
- speechbrain

## How to Use
Create a new instance of the VoiceAnalysis class.

Call `transcriptionGenerator = transcriptor.start()` to start the transcription and receive a generator that continuously yields an array of the words, timestamps of the current sentence spoken by the user and voice embeddings contained in the "Word" class. If a sentence is finished, it will yield this sentence, until the user starts speaking again, at which point it will yield the progress of the next sentence.

Call `transcriptor.close()` to stop the generator.

Parameters:
- microphone_index (int, default=0): The index of the microphone to use.
- speculative (bool, default=True): Should unconfirmed results be yielded by the generator? Allows you to access the data earlier, but the results may be inaccurate and may be corrected in a subsequent pass.
- whisper_model (string, default="large-v3"): Which Whisper model to use. Smaller is faster and less resource intensive, but also less accurate.
- device (string, default="cuda" ("cpu" if "cuda" is not available)): On which device the computations should be run. CUDA is highly recommended.
- voice_boost (float, default=10): How much the audio preprocessing stage should boost the volume of the user's voice.
- language (string, default=None): Which language the Whisper model should use. Will be autodetected if None, however, this may lead to a decrease in quality. Must be a valid language code, like "en", "de", "fr".
- verbose (bool, default=True): Wether debug information should be printed to the console.

Every word contains `text`, `start`, `end` and `speaker_embedding`. `start` and `end` are the timestamps of when the word is spoken relative to the beginning of the sentence. `speaker_embedding` is a `torch.FloatArray` which is a tensor representation of the users voice. Two embeddings can be checked against each other with the `compare_embeddings()` function from the [VoiceProcessingHelpers](#the-helper-class) to get a value between -1 and 1 representing how similar the embeddings and therefore the voices are.


## How it Works
The project is based on [this video](https://www.youtube.com/watch?v=_spinzpEeFM). The audio from the microphone is recorded and chunked into 1 second chunks. These chunks run through a 2 step audio preprocessing stage. In the first stage, the "denoiser" library is used to remove noise from the audio. However, the extracted voice cannot be used like this. Noise may have drowned out some of the speech. This results in audio which is low quality and won't improve the transcription results compared to the audio before the noise removal. This is why the volume of the audio without noise is boosted and it is added back to the original audio. This results in audio which, while still being noisy, has the voice of the speaker significantly increased compared to all other sounds.

In the second step of the preprocessing, the audio is passed through a VAD (Voice-Activity-Detection), specifically [this VAD](https://github.com/snakers4/silero-vad). It detects whether there is an actual voice in the audio or not. If not, the audio chunk will be used as context for later chunks, but it will not be processed further itself, as this chunk will only result in hallucinations from Whisper and a waste of processing time. However, if speech is detected, the chunk will be added to a larger chunk of audio, which represents the current sentence the user is speaking. This entire chunk will then be transcribed by Whisper. The result will be checked against the last result using a local agreement algorithm. Essentially, if a word is generated by Whisper 2 times in a row, it will be regarded as "confirmed" and cannot be changed anymore after that. This goes on until the sentence is finished and a new one is started.

The reason why this setup is so complicated is because Whisper was not trained for streaming applications, but rather it was trained to transcribe a singular sentence. This is why we always transcribe from the start of the sentence, because this will result in the best results from Whisper.

The audio of a singular word is then processed by SpeechBrain's `spkrec-xvect-voxceleb` model that generates a tensor representation of the users voice. This can be used to determine who is saying which word. This provides valuable context in applications like AI assistants.

## The Helper Class
The project contains the `VoiceProcessingHelpers` class, which provide more functionality to the system. The class contains the following functions:

- `word_array_to_string`: Takes a word-array and returns the full text as a single string.
- `compare_embeddings`: Takes two voice embeddings and compares how 'close' they are to each other. Can be used to check wether two voices are from the same person. Returns a value between -1 and 1 (higher is more similar).
- `save_to_wav`: Takes a torch.FloatTensor and a file path and saves the audio data to a wav file.
- `load_from_wav`: Takes a file path of a wav file and returns its audio data as a torch.FloatTensor.

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
