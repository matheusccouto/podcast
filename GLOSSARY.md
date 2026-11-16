# Podcast

Daily AI-generated news podcast curated from upstream tech sources and published as audio episodes.

## Language

**Source**:
An upstream feed or platform from which candidate news items are collected.
_Avoid_: Provider, scraper, origin

**Story**:
A raw candidate news item collected from a Source, including its metadata, linked content, and discussion.
_Avoid_: Article, post, item, submission

**Segment**:
A curated, scripted topic unit within an Episode that covers one Story or synthesizes multiple related Stories.
_Avoid_: Section, chapter, block, clip

**Script**:
The structured, speaker-labeled dialogue text and show notes generated for an Episode before audio synthesis.
_Avoid_: Transcript, prompt, draft, screenplay

**Episode**:
A single published daily audio installment composed of ordered Segments.
_Avoid_: Show, broadcast, issue, installment

**Taste Profile**:
The listener's explicit editorial preferences—topics to prioritize, topics to exclude, and desired depth—used to curate Stories.
_Avoid_: Config, prompt, user settings, filter rules

**Feed**:
The published RSS 2.0 XML and hosted Episode archive for a single language.
_Avoid_: Channel, stream, subscription, bucket

**Persona**:
A configured speaker identity in a Script, defining their name, editorial role, tone, and TTS voice.
_Avoid_: Host, voice, character, agent
