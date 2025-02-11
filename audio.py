import os
import logging
from gtts import gTTS

# Intentar importar playsound, sino utilizar otra librería
try:
    from playsound import playsound
    USE_PLAYSOUND = True
except ImportError:
    USE_PLAYSOUND = False
    import pygame
    pygame.mixer.init()

class AudioPlayer:
    def __init__(self, lang="en"):
        self.lang = lang
    
    def play_audio(self, file_path):
        if USE_PLAYSOUND:
            try:
                from playsound import playsound
                playsound(file_path)
            except Exception as e:
                logging.error(f"Error reproduciendo audio con playsound: {e}")
        else:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                logging.error(f"Error reproduciendo audio con pygame: {e}")

    def speak(self, text):
        if text:
            logging.info(f"Preparando para hablar: {text}")
            try:
                tts = gTTS(text=text, lang=self.lang)
                filename = 'temp.mp3'
                tts.save(filename)
                self.play_audio(filename)
                os.remove(filename)
            except Exception as e:
                logging.error(f"Error en la síntesis de voz: {e}")