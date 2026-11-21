Write a spoken two-Persona podcast `Dialogue` (Episode `title`, concise `summary`, and `turns`) in **{{ language }}** for a native **{{ language }}** audience, grounded in the provided `Story` sources.

## Personas
{% for persona in personas -%}
- voice_id `{{ persona.voice_id }}` — {{ persona.name }} ({{ persona.role }}; tone: {{ persona.tone }})
{% endfor %}
## Episode Arc
- **Show intro (first 2–4 turns):** Open with a quick hook or warm greeting, introduce both `Persona` speakers by name (`{{ personas[0].name }}` and `{{ personas[1].name }}`), and frame the theme connecting today's `Episode` before diving into the lead `Story`.
- **Thematic bridges:** Connect each `Story` to the next through a shared tension, contrast, or playful callback earned from the previous beat.
- **Natural close:** Land on a sharp takeaway, callback, or wry parting thought that ties the `Episode` together.

## Conversational Craft & Localization
- **Reactive banter:** Pace **22–34 turns** across **800–1,200 spoken words** with real rhythm variety—mix crisp 1-sentence reactions, interruptions, dry jokes, and sharp follow-up questions with tight 2–3 sentence explanations. Each turn responds directly to what the other `Persona` just said, pushing back or building on their exact point.
- **Standalone journalism:** Fold any `Story` discussion seamlessly into the reporting as perspectives from developers, critics, or users in the field. Hard guardrail: keep all meta-references to threads, comment sections, forums, or news aggregators out of the dialogue.
- **Locale-native phrasing:** Write as two native **{{ language }}** speakers talking to local listeners—situate foreign countries, elections, laws, agencies, and currencies explicitly on first mention, and express every idea in natural, idiomatic **{{ language }}** speech.

## SpeechTurn Contract
- `voice`: strictly `{{ personas[0].voice_id }}` or `{{ personas[1].voice_id }}` (exact `voice_id`, not the display name).
- `text`: pure verbatim speech in **{{ language }}**, at least 3 spoken words per turn.
  - Shape rhythm and micro-pauses with natural punctuation (`...`, `?`, `!`, `—`).
  - Keep standard English tech names and acronyms natural (`IPO`, `Y Combinator`, `Claude`, `Anthropic`), format decimals in **{{ language }}** convention, and hyphenate colliding acronyms (`E-U Login`).
  - Hard guardrail: keep `[brackets]`, `(stage directions)`, XML/SSML, URLs, and Markdown out of `text`; put all vocal direction in `instructions`.
- `instructions`: concise English **beat-anchored** audio direction (`<= 160` chars) **only** on turns with an audible shift—laugh, pause, pace change, deadpan, or emphasis on a specific word or phrase. Leave `instructions` as `null` on neutral connecting turns (~30–40% of turns).
  - Anchor cues to the exact moment in that turn: `"Light chuckle at the start, playful stress on '14 mil'"`, `"Brief pause before the twist, dry deadpan"`, `"Quick surprised reaction, leaning in"`.

Done when every `Story` is covered with concrete mechanisms and every section above is satisfied.
