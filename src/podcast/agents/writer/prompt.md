Episode date: {{ episode_date }}

## Curated Stories
{% for story in stories %}
### {% if story.is_lead %}LEAD STORY{% else %}STORY {{ loop.index }}{% endif %}: {{ story.title }}
Source URL: {{ story.url or "N/A" }}
Key Facts:
{% if story.key_facts %}{% for fact in story.key_facts %}  - {{ fact }}
{% endfor %}{% else %}  - None recorded
{% endif %}Community Comments:
{% if story.comments %}{% for comment in story.comments %}  - {{ comment }}
{% endfor %}{% else %}  - None recorded
{% endif %}Article Text:
{{ story.content[:max_content_chars] }}
{% if not loop.last %}
---
{% endif %}{% endfor %}
