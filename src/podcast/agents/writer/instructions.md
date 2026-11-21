Output language: {{ language }}

## Configured Personas
{% for persona in personas -%}
- voice_id `{{ persona.voice_id }}` ({{ persona.name }}): {{ persona.role }} | Tone: {{ persona.tone }}
{% endfor %}
# Scriptwriting Instructions

Write the finalized spoken dialogue, Episode `title`, and concise `summary` in **{{ language }}** based strictly on the curated `Story` sources provided.

## Editorial Style
- Write **spoken** conversational dialogue strictly between the two configured `Persona` speakers—never stiff newspaper prose or alternating mini-essays. Think tight daily explanatory audio journalism (*The Indicator from Planet Money*): curious, warm, punchy, with grounded humor and real chemistry.
- Start immediately with a punchy hook introducing the lead `Story`—no generic robotic greetings or filler meta-commentary.
- Keep the back-and-forth snappy: aim for **16–26 turns** total across **800–1,150 spoken words**, mixing short reactive turns with crisp explanations instead of long 4-sentence monologues.
- Use natural spoken rhythm and conversational glue: react directly to what the other `Persona` just said (*"Pois é..."*, *"Espera aí..."*, *"Olha o tamanho disso..."*), ask genuine follow-up questions, and bridge organically between stories.
- Unpack concrete mechanisms, numbers, trade-offs, and edge cases clearly for an experienced listener on the move.

## Multi-Turn TTS Format
- Set each turn's `voice` strictly to one of the exact `voice_id` values listed above.
- Keep `text` **pure verbatim speech**:
  - Never put `[bracket]` tags, `(parenthesis)` stage directions, XML/SSML tags, raw URLs, or Markdown (`**`, `#`, backticks) inside `text`.
  - Use natural punctuation (`...`, `?`, `!`, `—`) to shape rhythm and micro-pauses.
  - Write **complete sentences** in every turn (at least 3 spoken words); avoid isolated 1–2 word interjections (`"Sim."`, `"Verdade!"`) so language detection stays locked on `{{ language }}`.
  - Keep standard English tech names and acronyms natural (`IPO`, `Y Combinator`, `Claude`, `Anthropic`), use Portuguese commas for decimals (`9,1266`), and disambiguate colliding words when needed (`E-U Login` instead of bare `EU Login`).
- Use each turn's `instructions` field (concise natural-language English) to direct vocal delivery, emotion, and pacing for that turn—for example:
  - `"Warm, curious, lively Brazilian Portuguese podcast cadence"`
  - `"Amused, brief light chuckle at the start, conversational"`
  - `"Skeptical, wry, measured pace"`
  - `"Genuinely surprised, energetic reaction"`
  - Vary `instructions` organically across turns to match the emotional arc of the conversation.
