import wave
import random

files = 'sidespeaker.txt' 
merge_file = 'sidespeaker.wav'

def merge_wav(files, output):
    if isinstance(files, str):
        with open(files, 'r') as fr:
            items = [f.strip() for f in fr]
            files = random.sample(items, 1000)
            print(items)
    params = None
    data = []
    for f in files:
        with wave.open(f, 'rb') as wav:
            if not params:
                params = wav.getparams()
            data.append(wav.readframes(wav.getnframes()))

    with wave.open(merge_file, 'wb') as merged:
        merged.setparams(params)
        merged.writeframes(b''.join(data))

merge_wav(files, merge_file)
