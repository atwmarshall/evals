### [date/time] · 5 · short title

**What happened**: sometimes llms speak to much - "ignore in one sentence" or ignore token limits and just get truncated - which can be fatal for JSON formatting as never closed.
**Why it happened**: 
**What it means**: at worst: error parsing as incomplete json. at best: answer is truncated.
**How you'd fix it**: add detector and close the JSON.