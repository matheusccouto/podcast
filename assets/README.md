# Static stingers

Optional sonic-branding stinger assets live here and are committed to the
repository (no runtime music generation):

- `intro.mp3` — 3-5s instrumental intro played before the Personas speak.
- `outro.mp3` — 2-4s instrumental outro played after the last turn.

The files are created collaboratively with Lyria and enabled by pointing the
runtime settings at them:

```bash
PODCAST_INTRO_PATH=assets/intro.mp3
PODCAST_OUTRO_PATH=assets/outro.mp3
```

When a path is left unset, `OpenRouterTTS` encodes the normalized dialogue
alone. When a path is set but the file is absent, synthesis fails loudly.
