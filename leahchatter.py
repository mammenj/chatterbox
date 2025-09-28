import torch
import time
import re
import torchaudio as ta

# from chatterbox.tts import ChatterboxTTS
from extended.chatterbox.src.chatterbox.tts import ChatterboxTTS


start = time.perf_counter()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


def getName(sentence):
    clean_sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

    words = clean_sentence.split()
    if words:
        firstword = words[0]
        lastword = words[-1]
        return firstword + "_" + lastword


def main():
    # with open("script.txt", "r") as file:
    # line = file.readline()
    # while line:
    # text += line
    # line = file.readline()
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = ChatterboxTTS.from_pretrained(device=device)

    count = 0
    with open("leah.txt", "r") as file:
        line = file.readline()
        while line:
            # text += line
            print("Processing line: ", line)
            AUDIO_PROMPT_PATH = "voicedir/leah.wav"

            # line = file.readline()
            if line.strip() != "" and line.__len__() > 3:
                # tts.tts_to_file(
                # text=line,
                # file_path=audio_path,
                # speaker="leah",
                # language="en",
                # split_sentences=False,
                # )
                count = count + 1
                audio_name = getName(line)
                audio_path = f"output/john/leah-{count:03d}-{audio_name}.wav"

                wav = model.generate(
                    line,
                    audio_prompt_path=AUDIO_PROMPT_PATH,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8,
                )
                ta.save(audio_path, wav, model.sr)
                print("Done with ", audio_path)
            line = file.readline()

    end = time.perf_counter()
    print(f"Execution time: {end - start:0.4f} seconds")


if __name__ == "__main__":
    main()
