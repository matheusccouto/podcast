Write a spoken two-Persona podcast `Dialogue` (Episode `title`, concise `summary`, and `turns`) in **{{ language }}** grounded in the provided `Story` sources.

## Personas
{% for persona in personas -%}
- voice_id `{{ persona.voice_id }}` — {{ persona.name }} ({{ persona.role }}; tone: {{ persona.tone }})
{% endfor %}
## Episode Arc
- **Cold open:** Start directly on the sharpest tension, surprise, or concrete number from the lead `Story` so the listener is hooked in the first sentence.
- **Thematic bridges:** Connect each `Story` to the next through a shared thread, contrast, or playful callback earned from the previous beat.
- **Natural close:** Land on a sharp takeaway, callback, or wry parting thought that ties the episode together.

## Spoken Craft
- Write lively, explanatory radio banter between the two `Persona` speakers—warm, curious, grounded humor, real chemistry.
- Pace **18–28 turns** across **800–1,200 spoken words**, keeping turns tight (1–3 sentences) and unpacking complex mechanisms, numbers, and trade-offs across reactive exchanges.
- React directly to what the other `Persona` just said (*"Pois é..."*, *"Espera aí..."*, *"Olha o tamanho disso..."*), interrupt with genuine follow-up questions, and push back when a claim sounds too neat.

## SpeechTurn Contract
- `voice`: strictly `{{ personas[0].voice_id }}` or `{{ personas[1].voice_id }}` (exact `voice_id`, not the display name).
- `text`: pure verbatim speech in **{{ language }}**, at least 3 spoken words per turn.
  - Shape rhythm and micro-pauses with natural punctuation (`...`, `?`, `!`, `—`).
  - Keep standard English tech names and acronyms natural (`IPO`, `Y Combinator`, `Claude`, `Anthropic`), write decimals with Portuguese commas (`9,1266`), and hyphenate colliding acronyms (`E-U Login`).
  - Hard guardrail: keep `[brackets]`, `(stage directions)`, XML/SSML, URLs, and Markdown out of `text`; put all vocal direction in `instructions`.
- `instructions`: concise English **beat-anchored** audio direction (`<= 160` chars) **only** on turns with an audible shift—laugh, pause, pace change, deadpan, or emphasis on a specific word or phrase. Leave `instructions` as `null` on neutral connecting turns (~30–40% of turns).
  - Anchor cues to the exact moment in that turn: `"Light chuckle at the start, playful stress on '14 mil'"`, `"Brief pause before the twist, dry deadpan"`, `"Quick surprised reaction, leaning in"`.

Done when every `Story` is covered with concrete mechanisms, the dialogue reads aloud as effortless human banter, and `instructions` are beat-anchored rather than generic tone labels.
