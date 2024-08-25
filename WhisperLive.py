#To add: Bandpass filter. Filter out frequencies that are not in the human voice range to reduce noise further. Add after the speech boost step.
#To add: LLM post processing.
#To add: Timestamps.

import os
import numpy as np
import sounddevice as sd
import threading
import time
import torch
import queue
from typing import Optional, Literal
from faster_whisper import WhisperModel
from denoiser import pretrained
from denoiser.dsp import convert_audio
import langcodes
from scipy.io import wavfile
import time
import librosa

os.environ["KMP_DUPLICATE_LIB_OK"] = "True" #Only necessary for windows

class Transcriptor:
    """
    A pipeline for live transcription using the Whisper model.
    """

    def __init__(self,
                microphone_index: int,
                speculative: bool = True,
                whisper_model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] = "large-v3",
                device: Literal["cpu", "cuda"] = "cuda",
                voice_boost: float = 10.0,
                language: str = None,
                verbose: bool = True
                ) -> None:
        """
        Arguments:
            microphone_index (int): The index of the microphone to use for recording.
            speculative (bool, optional): Whether to provide unconfirmed results. Allows you to access more of the transcription earlier, but the data may be inaccurate and is subject to change in the next pass. Defaults to True.
            whisper_model ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3", optional): The Whisper model to use. Defaults to "large-v3".
            device ("cpu", "cuda", optional): The device to use for the computations. Defaults to "cuda" or "cpu" if cuda is not available.
            voice_boost (float, optional): How much to boost the voice in the audio preprocessing stage. Setting it to 0 disables this feature. Defaults to 10.0.
            language (str, optional): The language to use for the transcription. Must be a valid language code. If None, the language will be autodetected. If possible, the language should be set to improve accuracy. Defaults to None.
            verbose (bool, optional): Whether to print debug information. Defaults to True.
        """

        self.__verbose = verbose

        #region Torch setup
        self.device = device
        if (device == "cuda" and not torch.cuda.is_available()):
            self.device = "cpu"
            
        self.__log(f"Using device '{self.device}'.")
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        #endregion

        #region Whisper setup
        self.model = WhisperModel(whisper_model, device="cuda", compute_type="float32")
        self.__log(f"Whisper model of size '{whisper_model}' loaded.")
        #Check if the language is valid
        if (language is not None):
            try:
                langcodes.Language.get(language)
            except:
                raise ValueError(f"{language} is not a valid language code.")
            self.__log(f"Language set to '{language}'.")
        else:
            self.__log("Language set to auto.")
        self.language = language
        #endregion

        #region Denoiser setup
        if (voice_boost != 0.0):
            self.denoiseModel = pretrained.dns64()
            self.denoiseModel.eval()
            self.__log("Denoiser model loaded.")
        else:
            self.__log("voice_boost is set to 0, skipping denoiser model loading.")
        #endregion

        #region VAD setup
        self.vadModel, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)
        self.vadModel = self.vadModel.to(self.device)
        (self.get_speech_ts, _, self.read_audio, self.VADIterator, self.collect_chunks) = utils
        self.vad_iterator = self.VADIterator(self.vadModel)
        self.__log("VAD model loaded.")
        #endregion

        #region Data setup
        self.microphone_index = microphone_index
        self.__sample_rate = 16000 
        self.__maxSilenceChunks = 3
        self.__currentSentence = []
        self.__lockedWords = 0
        self.__audio_queue = queue.Queue()
        self.__is_recording = True
        self.__voice_boost = voice_boost
        self.__speculative = speculative
        self.__recording_thread = threading.Thread(target=self.__record_audio)
        #endregion

        self.__log("Initialization complete.\n")

    #region Recording and audio preprocessing
    def __record_audio(self):
        audio_buffer = np.array([], dtype=np.float32)  
        silence_start = None
        lastTranscriptionTime = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_start, lastTranscriptionTime
            audio = np.frombuffer(indata, dtype=np.float32)  

            audio_buffer = np.concatenate((audio_buffer, audio))

            if time.time() - lastTranscriptionTime >= 1:
                lastTranscriptionTime = time.time()
                if len(audio_buffer) > 0:
                        self.__audio_queue.put(audio_buffer)
                        audio_buffer = np.array([], dtype=np.float32)  

        with sd.InputStream(callback=callback, dtype=np.float32, channels=1, samplerate=self.__sample_rate, device=self.microphone_index):
            while self.__is_recording:
                time.sleep(0.1)

        if len(audio_buffer) > 0:
            self.__audio_queue.put(audio_buffer)

    def __boost_speech(self, audio_data):
        if (self.__voice_boost == 0.0): return audio_data

        audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
        audio_tensor = audio_tensor.unsqueeze(0)
        audio = convert_audio(audio_tensor, self.__sample_rate, self.denoiseModel.sample_rate, self.denoiseModel.chin)

        with torch.no_grad():
            denoised = self.denoiseModel(audio)[0]

        denoised = denoised * self.__voice_boost
        audio_tensor = audio_tensor + denoised
        audio_tensor = audio_tensor / audio_tensor.abs().max()
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = audio_tensor.cpu()

        return audio_tensor.numpy()
    
    def __detect_voice_activity(self, audio_chunk):
        if (type(audio_chunk) == np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float().to(self.device)
        else:
            audio_tensor = audio_chunk.float().to(self.device)
        
        min_audio_length = int(16000 / 31.25)
        if audio_tensor.shape[0] < min_audio_length:
            padding = torch.zeros(min_audio_length - audio_tensor.shape[0], device=self.device)
            audio_tensor = torch.cat([audio_tensor, padding])
        
        speech_probs = []
        for i in range(0, len(audio_tensor), 512):
            window = audio_tensor[i:i+512]
            if len(window) == 512:
                speech_prob = self.vadModel(window, 16000).item()
                speech_probs.append(speech_prob)

        speech_detected = any(prob > 0.95 for prob in speech_probs)
        return speech_detected, speech_probs
    #endregion

    #region Transcription
    def __transcribe(self, audio_tensor):
        if (self.language is not None):
            segments, info = self.model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, language=self.language, condition_on_previous_text=False, word_timestamps=True)
        else:
            segments, info = self.model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, condition_on_previous_text=False, word_timestamps=True) #Leave the language undefined so whisper autodetects it
        transcription = []
        for segment in segments:
            for word in segment.words:
                transcription.append(Word(word.word, word.start, word.end))
        return transcription

    def __update_transcription(self, words):
        newLockedWords = self.__lockedWords
        for i in range(len(words)):
            if i < self.__lockedWords:
                continue
            if i < len(self.__currentSentence) and words[i].text == self.__currentSentence[i].text:
                newLockedWords += 1
            else:
                break
        
        self.__currentSentence = words[:newLockedWords] + words[newLockedWords:]
        self.__lockedWords = newLockedWords
        
        return self.__currentSentence, self.__lockedWords
    #endregion

    #region Main generator
    def start(self):
        """
        Returns a generator object that continuously yields the current sentence that is recorded from the microphone.
        if "speculative" is set to True, the generator may correct itself in the next pass, if not, it yields the words as they are coming in.
        When a sentence is finished, the generator yields the full sentence, until the user continues speaking, which will reset the sentence.
        """

        self.__recording_thread.start()

        currentAudioData = None
        firstAudioChunk = None
        silenceCounter = 0

        confirmedTranscription = ""
        speculativeTranscription = ""

        while self.__is_recording:
            if not self.__audio_queue.empty():
                audio_chunk = self.__audio_queue.get()
                audio_chunk = self.__boost_speech(audio_chunk)
                speechDetected, probabilities = self.__detect_voice_activity(audio_chunk)

                if currentAudioData is None:
                    if not speechDetected:
                        firstAudioChunk = audio_chunk
                        continue
                else:
                    if not speechDetected:
                        silenceCounter += 1
                        currentAudioData = np.concatenate((currentAudioData, audio_chunk))
                        if silenceCounter >= self.__maxSilenceChunks:
                            continue
                
                if speechDetected:
                    silenceCounter = 0

                if currentAudioData is None:
                    if firstAudioChunk is not None:
                        currentAudioData = np.concatenate((firstAudioChunk, audio_chunk))
                    else:
                        currentAudioData = audio_chunk
                else:
                    currentAudioData = np.concatenate((currentAudioData, audio_chunk))

                audio_tensor = torch.from_numpy(currentAudioData).float().to(self.device)
                
                transcription = self.__transcribe(audio_tensor)

                self.__currentSentence, self.__lockedWords = self.__update_transcription(transcription)

                confirmedTranscription = []
                speculativeTranscription = []
                for i, word in enumerate(self.__currentSentence):
                    if i < self.__lockedWords:
                        confirmedTranscription.append(word)

                    speculativeTranscription.append(word)

                if (len(confirmedTranscription) > 0):
                    if ("." in confirmedTranscription[len(confirmedTranscription) - 1].text or "!" in confirmedTranscription[len(confirmedTranscription) - 1].text or "?" in confirmedTranscription[len(confirmedTranscription) - 1].text):
                        currentAudioData = None
                        self.__currentSentence = []
                        self.__lockedWords = 0
            
            if (self.__speculative):
                yield speculativeTranscription
            else:
                yield confirmedTranscription

            time.sleep(0.1)
        
    def close(self):
        self.__is_recording = False
        self.__audio_queue = queue.Queue()
        if (self.__recording_thread.is_alive()):
            self.__recording_thread.join()
    #endregion

    #region Transcription from file
    def transcribe_from_file(self, file_path: str) -> str:
        startTime = time.time()
        audioData = self.__load_from_wav(file_path).squeeze(0)
        if (self.__detect_voice_activity(audioData)):
            transcription = self.__transcribe(audioData)
            self.__log(f"File transcribed in {round((time.time() - startTime) * 100) / 100} seconds.")
            return transcription
        else:
            self.__log("No speech detected in the audio file.")
            return ""
    #endregion

    #region Misc tools
    @staticmethod
    def word_array_to_string(word_array) -> str:
        text = ""
        for word in word_array:
            text += word.text
        return text
    #endregion

    #region Debug and helpers
    @staticmethod
    def __save_to_wav(file_path: str, audio_tensor: torch.FloatTensor, __sample_rate: int) -> None:
        audio_tensor = audio_tensor.cpu()
        audio_numpy = audio_tensor.squeeze().numpy()
        audio_int16 = (audio_numpy * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write(file_path, __sample_rate, audio_int16)

    @staticmethod
    def __load_from_wav(file_path: str) -> torch.FloatTensor:
        sample_rate, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        if sample_rate != 16000:
            audioData = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        audio_tensor = torch.FloatTensor(audioData)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor
    
    def __log(self, text: str) -> None: #Used to print debug information if verbose is set to True. Reduces if statements in the code.
        if self.__verbose:
            print(text)
    #endregion

class Word:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end
