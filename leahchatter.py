import re
import time
import argparse
import torch
import torchaudio as ta

# from chatterbox.tts import ChatterboxTTS
from extended.chatterbox.src.chatterbox.tts import ChatterboxTTS

start = time.perf_counter()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


def getArgs():
    parser = argparse.ArgumentParser(description="TTS with Chatterbox")
    parser.add_argument("script", type=str, help="Path to the script file")
    parser.add_argument("outdir", type=str, help="Output directory for audio files")
    parser.add_argument("voice", type=str, help="Voice to use for TTS")
    parser.add_argument(
        "exaggeration", type=float, help="Exaggeration level for TTS", default=0.6
    )
    parser.add_argument(
        "cfg_weight", type=float, help="CFG weight for TTS", default=0.4
    )
    args = parser.parse_args()
    return args.script, args.outdir, args.voice, args.exaggeration, args.cfg_weight


def getName(sentence):
    clean_sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

    words = clean_sentence.split()
    if words:
        firstword = words[0]
        lastword = words[-1]
        return firstword + "_" + lastword


def main():
    script, outdir, voice, exaggeration, cfg_weight = getArgs()
    if (
        script is None
        or outdir is None
        or voice is None
        or exaggeration is None
        or cfg_weight is None
    ):
        exit("Please provide script, outdir, voice, exaggeration, cfg_weight arguments")

    print(
        f"Script: {script}, Output Directory: {outdir}, Voice: {voice} , Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}"
    )

    model = ChatterboxTTS.from_pretrained(device=device)

    count = 0
    with open(script + ".txt", "r") as file:
        line = file.readline()
        while line:
            # text += line
            print("Processing line: ", line)
            AUDIO_PROMPT_PATH = f"voicedir/{voice}"

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
                audio_path = f"output/{outdir}/{script}-{count:03d}-{audio_name}.wav"

                wav = model.generate(
                    line,
                    audio_prompt_path=AUDIO_PROMPT_PATH,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=0.9,
                )
                ta.save(audio_path, wav, model.sr)
                print("Done with ", audio_path)
            line = file.readline()

    end = time.perf_counter()
    print(f"Execution time: {end - start:0.4f} seconds")


if __name__ == "__main__":
    main()
