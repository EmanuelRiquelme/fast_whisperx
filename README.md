<h1 align="center">WhisperX</h1>

## Faster Implementation of [Whisper X](https://github.com/m-bain/whisperX/tree/main)

Tested on Python 3.11.0 and Python 3.10.12 

If wishing to modify this package, clone and install in editable mode:
```
$ git clone https://github.com/EmanuelRiquelme/fast_whisperx
$ cd fast_whisperx
$ pip install -e .
```

A quick test taking advantage of all of the hardware available.
```
import whisperx
import torch
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
language = 'en'
try:
    model = whisperx.load_model("tiny.en", device, compute_type='float16',language = language)
except:
    model = whisperx.load_model("tiny.en", device, compute_type='int8',language = language)

def generate_transcript(audio):
    audio = whisperx.load_audio(audio)
    try:
      return model.transcribe(audio, batch_size=512)["segments"][0]['text']
    except:
      return model.transcribe(audio, batch_size=8)["segments"][0]['text']
```
the first time that the functions runs it takes about 3 seconds to run, from then it takes usually less than a second.
If you use this in your research, please cite the paper:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={arXiv preprint, arXiv:2303.00747},
  year={2023}
}
```
