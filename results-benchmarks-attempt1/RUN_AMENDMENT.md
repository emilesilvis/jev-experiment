# Archived initial attempt

SNIPS produced the same invalid GPT probability distribution on its first example twice. The runner stopped on each invalid response. The partial suite was terminated before completion.

The final suite starts fresh with unchanged prompts, model adapters, validation rules, and snapshots. Its runner treats malformed completed model responses as incorrect final outcomes and continues; request/transport failures still stop. No responses from this archive enter the final suite statistics.

An in-flight request at termination may not have a saved cost record.

Saved attempts:
- ag_news: 187 attempts; 1 errors.
- snips: 3 attempts; 2 errors.
- trec: 264 attempts; 0 errors.
