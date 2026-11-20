# HolosTTS

Ukrainian text to speech for Home Assistant's Assist with [HolosTTS](https://huggingface.co/patriotyk/HolosTTS):
a natural Ukrainian voice that runs on the CPU (INT8 ONNX model, about 470 MB of RAM in use) and starts speaking
before the whole reply is synthesized.

This add-on is the CPU fallback. On a machine with an NVIDIA GPU, run the
[Docker image](https://github.com/nemethhh/holostts-wyoming#gpu-docker-image) instead and add it to Home Assistant
by host and port.

## Setup

1. Install and start the add-on. Home Assistant finds it: accept the **Wyoming Protocol** integration it offers
   (Settings → Devices & services).
2. Settings → Voice assistants → your Ukrainian assistant → **Text-to-speech**: pick **holostts** and a voice.

## What it reads correctly, and what not

The text goes through these steps before the model: clean-up, numbers to words, your stress fixes, dictionary
stress, IPA phonemes.

- **Numbers** agree with the unit after them: "21 °C" → "двадцять один градус", "5 хвилин", "40%" → "сорок
  відсотків", "21.5 °C" → "двадцять одна ціла п'ять десятих градуса", "15:30" → "п'ятнадцята тридцять", and
  "на 1 хвилину" → "на одну хвилину". Other case forms ("о сьомій"), dates and abbreviations are read as plain
  numbers or letters.
- **Stress** comes from a dictionary. A word with two possible stresses (за́мок / замо́к) gets the first one; a word
  missing from the dictionary gets none. Fix either with `stress_overrides`.

## Options

| option | default | |
|---|---|---|
| `voice` | `Speaker_0` | Default voice when the assistant does not choose one (27 preset voices). |
| `speed` | 1.0 | Above 1 speaks faster, below 1 slower. |
| `threads` | 2 | CPU threads per synthesis. Only one reply is synthesized at a time. |
| `stress_overrides` | none | Words with `+` after the stressed vowel, e.g. `замо+к`. Whole word, any case. |
| `debug_logging` | false | Log each reply's text and phonemes. |

## Credits and licence

MIT. HolosTTS and its models by Serhiy Stetskovych (patriotyk), MIT. Stress dictionary from
[ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) (MIT), phonemes by
[ipa-uk](https://github.com/patriotyk/ipa-uk) (MIT), number words by [num2words](https://github.com/savoirfairelinux/num2words) (LGPL).

The preset voices were taken without the consent of their owners (see the HolosTTS model card). Do not use them
commercially.
