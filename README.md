# Cold Opener

Give it a company's website and it writes a 3-line opener for a cold email to the founder, using only facts from that site.

Choose the AI: **Google Gemini** (`gemini-3.5-flash`) or **Groq** (`openai/gpt-oss-120b`). Both have free tiers.

```
$ python cli.py tally.so
Tally takes a unique approach by letting users build online forms just by typing like in a text document.
Offering unlimited forms and submissions for free while staying bootstrapped as a team of 10 is an impressive direction.
Would love to connect and hear how your team is thinking about scaling the platform this year.

  founder: not named on site
  facts:   ['type like you would in a doc', 'unlimited forms and submissions, completely free', 'bootstrapped team of 10']
  pages:   ['https://tally.so/', 'https://tally.so/about']
  gemini gemini-3.5-flash: 1 attempt(s), 17.1s, tokens {'input_tokens': 1128, 'output_tokens': 175}
```

## Run it

```bash
git clone <this repo> && cd cold-opener
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'GEMINI_API_KEY=your-key' >> .env     # free: aistudio.google.com
echo 'GROQ_API_KEY=your-key' >> .env       # free: console.groq.com/keys (either key is enough)
.venv/bin/python cli.py stripe.com          # command line, Gemini
.venv/bin/python cli.py --groq stripe.com   # command line, Groq
.venv/bin/streamlit run streamlit_app.py    # web page with a provider picker
```

## How it works

| Step | Where | What |
|---|---|---|
| 1. Clean the URL, block private/localhost IPs | code | `normalize_url`, `assert_public_host` |
| 2. Fetch homepage + up to 3 useful pages: about, customers, pricing, blog (10s timeout, 2 MB cap) | code | `read_site`, `find_extra_pages` |
| 3. Strip HTML down to title, meta description, headings, paragraphs (≤5k chars/page) | code | `extract` |
| 4. Write 3 different candidate openers, each built on one specific "hook" (a number, customer, unusual choice), plus the exact site phrases each relies on, as JSON | **model** (Gemini 3.5 Flash or Groq gpt-oss-120b, low reasoning, JSON schema) | `call_model` |
| 5. Check each candidate: exactly 3 lines, ≤30 words each, no clichés, every cited phrase and number actually appears on the site. Score the survivors for specificity (real numbers and names, line 1 built on a fact, ends in a question, no hype words) and show the best, with the rest as alternatives. If none pass, retry once with the reasons | code | `validate`, `score` |

If the site has almost no readable text (JavaScript-only), it stops and says so instead of guessing. If it has only a little text, it still writes an opener but warns that it may read generic. A founder name is only used in the greeting if it appears on the site.

## Cost

Measured: about 0.9k–3.3k input + ~150–200 output tokens per run. Google's pricing page doesn't list `gemini-3.5-flash` itself. For reference, Gemini 3.8 Flash is $0.75 / $3.75 per 1M tokens (through Dec 31, 2026, then double), which would be about **$0.003 per opener**, or about $3/day at 1,000 runs. On the free tier, Google may use the inputs to improve its products.

On Groq, `openai/gpt-oss-120b` is listed at $0.15 / $0.60 per 1M tokens, about **$0.0005 per opener** (about $0.50/day at 1,000 runs). It's Groq's strongest production text model that supports strict JSON-schema output.

**Free-tier limits matter more than price:** while testing, Gemini 3.8 Flash's free tier cut us off at 20 requests/day (the default is now 3.5 Flash, whose free quota isn't published on the limits page). Groq lists 1,000 requests/day and 30/minute for gpt-oss-120b.

Override models with `GEMINI_MODEL=...` / `GROQ_MODEL=...`, and the default provider with `LLM_PROVIDER=groq`.
