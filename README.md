# HolosTTS for Home Assistant

[HolosTTS](https://huggingface.co/patriotyk/HolosTTS) as a [Wyoming](https://github.com/OHF-Voice/wyoming)
text-to-speech service for Assist: a natural **Ukrainian** voice, streamed sentence by sentence.

Two ways to run it, from the same code:

- **GPU Docker images** (recommended on a machine with a GPU): `ghcr.io/nemethhh/holostts-wyoming` for NVIDIA,
  `:latest-vulkan` for AMD and Intel.
- **Home Assistant add-on** (CPU): runs on the Home Assistant machine itself.

## Add-on

[![Add repository to Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fnemethhh%2Fholostts-wyoming)

Or: Settings → Add-ons → Add-on Store → ⋮ → Repositories → `https://github.com/nemethhh/holostts-wyoming`.
Then install **HolosTTS**; see its [documentation](holostts/DOCS.md).

## GPU Docker image

See [docker/README.md](docker/README.md).

## Development

`make models` downloads the pinned model files into `models/`; `make test` runs the tests (pytest);
`make image` / `make image-cuda` / `make image-cuda13` / `make image-vulkan` build the images; `make smoke` runs the add-on image and synthesizes one request.
`tools/wyoming_client.py HOST PORT TEXT OUT.wav [--stream]` talks to any Wyoming TTS server and prints the time to
first audio.

The text front end is a chain of named steps (`--steps normalize,numbers,overrides,stress,ipa`); a new step
registers itself with `@register("name")` in `src/holostts_wyoming/steps/`. The speech model sits behind the
`Engine` protocol in `src/holostts_wyoming/engines/`.

## Licence

MIT. See [holostts/DOCS.md](holostts/DOCS.md#credits-and-licence) for credits, and the warning about the preset
voices.
