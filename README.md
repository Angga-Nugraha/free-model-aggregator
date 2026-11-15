# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Angga-Nugraha/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Angga-Nugraha/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A server that answers OpenAI and Anthropic requests with the best free model able to serve them, across ten free-tier providers, with failover on 402/429 and an exact catalog of what it read.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
```

One file, Node 22, standard library only. The server reads the trial brief (`MAKEFUN_PROMPT_FILE`) for the provider directory, the quality table and the workload, probes each provider for its live model listing, builds one catalog, registers its port and that catalog with the environment (`MAKEFUN_ENV_URL`), then serves the environment's 20 requests one at a time.

Routing: for each request, candidates are the free models that meet its requirements (tools, vision, JSON mode, minimum context), ordered by quality rating descending. On a 402, a 429 or a 5xx the server moves to the next candidate, honours `Retry-After`, and marks a provider dead for the rest of the trial when its answer says its budget is spent. Exactly one successful provider call answers each request. Answers are translated between the OpenAI, Anthropic and Gemini wires without touching the echo fields (system text, user text, tool names, max_tokens, stop sequences, image bytes, JSON mode), and the provider's reported usage is copied into the client response exactly.

`node main.ts --list` prints the catalog document byte for byte as registered. Practice trials 0–2 grade 99/100 each (catalog 100, failover 100, served 100, wire 100; quality 99 is the ceiling, since the tools+vision requests cap at 1458 against the 1475 normaliser).

---

<sub>Entered on [make.fun](https://make.fun/?ref=2nbzqxkh&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
