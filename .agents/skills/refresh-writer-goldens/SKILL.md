---
name: refresh-writer-goldens
description: Replace the Writer golden fixtures with fresh NPR The Indicator episodes.
disable-model-invocation: true
---

Replace or top up the fixtures in `evals/golden/writer/` with 5 episodes of NPR's *The Indicator from Planet Money* (feed: `https://feeds.npr.org/510325/podcast.xml`). Each fixture is a `GoldenCase` (`evals/golden/writer/__init__.py`): the episode's cited sources become `stories`, and its transcript becomes `expected_output`.

## Steps

1. **Select episodes.** From the feed, pick episodes of `<= 600s` (matches our under-10-minute Episode target) until you have 5 total that each have:
   - a transcript at `https://www.npr.org/transcripts/<episode_id>` featuring only the two hosts (skip episodes with third-party guest interviews or soundbite speakers),
   - an `<enclosure url="...">` MP3, and
   - 2–4 **cited sources**: external links in the episode page's `#storytext` pointing at the data or reporting the episode draws on (not npr.org, sponsors, or newsletter/app promos).

   *Done when*: all target episodes are listed, each with ID, `pubDate`, transcript URL, MP3 URL, and 2–4 cited-source URLs.

2. **Build stories.** Fetch each cited source with `ExaFetcher` (`EXA_API_KEY` in `.env`); keep those with `>= 400` chars of content. Set `title` to the source page's own headline (plain text, no Markdown links or generic `"Source N"` placeholders).

   *Done when*: every episode has `>= 2` `GoldenStory` items, each titled with a real headline.

3. **Transcribe turns.** Map the two hosts to `voice` `"persona_a"` and `"persona_b"`. Keep `text` **verbatim** speech: strip speaker labels and NPR stage noise such as `(LAUGHTER)`, `(SOUNDBITE OF MUSIC)`, and `[inaudible]`. Merge any turn under 3 words into its neighbour.

   *Done when*: every turn constructs as a `SpeechTurn` without error.

4. **Annotate from audio.** Download each MP3 to a scratch path and listen to it with `read_file`; if the audio cannot be ingested, stop and report loudly rather than guessing delivery from text. For each turn, write `instructions` (`<= 160` chars) describing the delivery you hear: tone, pacing, laughs, sighs, pauses. Leave `instructions` as `null` for neutral connecting turns. Delete the temporary MP3s afterwards.

   *Done when*: every turn has been heard, and every non-null `instructions` is **audio-grounded** (describes something audible in that turn).

5. **Write & validate.** Save each new case as `evals/golden/writer/<episode_id>.json` with `date` = the episode's `pubDate` (deleting stale fixtures only when doing a full refresh). Run:
   `uv run python -c "from evals.conftest import load_golden_dataset; assert len(load_golden_dataset().goldens) == 5"`

   *Done when*: the command exits 0.
