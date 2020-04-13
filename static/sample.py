import os
from gtts import gTTS

# Options
text_to_read = "I am sorry, I was not sure what emotion you were expressing or maybe there was an issue with a light or camera. But I still would love to chat with you!"
language = 'en'
slow_audio_speed = False
filename = 'neutural.mp3'

def reading_from_string():
    audio_created = gTTS(text=text_to_read, lang=language,
                         slow=slow_audio_speed)
    audio_created.save(filename)

if __name__ == '__main__':
    reading_from_string()