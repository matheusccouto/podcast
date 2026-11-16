Output language: {{ language }}

## Configured Personas
{% for persona in personas -%}
- voice_id `{{ persona.voice_id }}` ({{ persona.name }}): {{ persona.role }} | Tone: {{ persona.tone }}
{% endfor %}
# Scriptwriting Instructions

Write the finalized spoken dialogue in **{{ language }}** based strictly on the curated `Story` sources provided.

## Editorial Style
- Conversational dialogue strictly between the two configured `Persona` speakers, inspired by tight daily explanatory audio journalism: curious, direct, warm, with light, grounded humor.
- Start immediately with a punchy hook introducing the lead `Story`—no generic robotic greetings or filler meta-commentary.
- Bridge naturally between stories; unpack concrete mechanisms, numbers, trade-offs, and edge cases clearly for an experienced listener on the move.
- Target **800–1,150 spoken words** total across all turns.

## Fish Audio S2.1 Pro Turn Format
- Set each turn's `voice` strictly to one of the exact `voice_id` values listed above.
- Express delivery sparingly using inline `[bracket]` cues inside `text` (for example `[chuckle]`, `[laughing]`, `[sigh]`, `[excited]`), never `(parenthesis)` stage directions or XML/SSML tags.
- Write **complete sentences** in every turn; avoid isolated 1–2 word interjections (`"Sim."`, `"Verdade!"`) so the TTS language detector stays locked on `{{ language }}`.
- Write numbers, versions, and abbreviations cleanly for natural speech, and never include raw URLs or Markdown formatting (`**`, `#`, backticks).
