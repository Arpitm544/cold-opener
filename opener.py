"""Website URL -> 3-line cold email opener for the founder.

Pipeline (code does everything except the writing):
  1. normalize + safety-check the URL            (code)
  2. fetch homepage + one "about"-style page     (code)
  3. strip HTML down to the text that matters    (code)
  4. write the opener from that text only        (model: Gemini or Groq, user's choice)
  5. validate: 3 lines, length, banned phrases,
     every cited fact really appears on the site (code) -> retry once if it fails
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import groq
import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

# Load GEMINI_API_KEY from a local .env file if present (kept out of git).
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _line in open(_env).read().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PROVIDERS = {
    # name: (default model, env var for the key, where to get a free key)
    "gemini": (os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"), "GEMINI_API_KEY", "aistudio.google.com"),
    # gpt-oss-120b: Groq's strongest production text model with strict JSON-schema output, free tier included.
    "groq": (os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "GROQ_API_KEY", "console.groq.com/keys"),
}
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
USER_AGENT = "Mozilla/5.0 (compatible; ColdOpenerBot/0.1)"
FETCH_TIMEOUT = 10.0
MAX_BYTES = 2_000_000          # don't download huge pages
MAX_CHARS_PER_PAGE = 5_000     # keeps prompt (and cost) small
MAX_EXTRA_PAGES = 3            # about / customers / pricing-style pages on top of the homepage
THIN_SITE_CHARS = 1_500        # below this, warn that the opener may be generic
MIN_USEFUL_CHARS = 300         # below this the site is probably JS-rendered
MAX_WORDS_PER_LINE = 30
BANNED = [
    "hope this finds you well", "hope you're doing well", "i came across", "i noticed", "noticing",
    "i stumbled upon", "revolutionize", "game-changer", "game changer",
    "synergy", "touch base", "quick question", "just wanted to",
]
ABOUT_HINTS = ("about", "team", "company", "story", "founder", "founders", "who-we-are", "us")
# Pages most likely to hold specific, hook-worthy facts, in priority order.
PAGE_HINTS = [
    ABOUT_HINTS,
    ("customers", "case-studies", "stories", "testimonials", "wall-of-love", "showcase"),
    ("pricing", "plans"),
    ("changelog", "blog", "updates", "news"),
]


class OpenerError(Exception):
    """Something the user should see (bad URL, unreadable site, model failure)."""


class KeyProblem(OpenerError):
    """The API key is missing, invalid, or out of free quota: a different key would fix it."""


# ---------- 1. URL handling ----------

def normalize_url(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise OpenerError("Please enter a website address.")
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname or "." not in parsed.hostname:
        raise OpenerError(f"'{raw}' doesn't look like a website address.")
    return f"{parsed.scheme}://{parsed.hostname}{parsed.path or '/'}"


def assert_public_host(url: str) -> None:
    """Block localhost/private IPs so the tool can't be used to probe internal networks (SSRF)."""
    host = urlparse(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise OpenerError(f"Couldn't find a website at {host}.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise OpenerError("That address points to a private network, so it's blocked.")


# ---------- 2 + 3. Fetch and extract ----------

@dataclass
class Page:
    url: str
    title: str
    text: str


def fetch_html(client: httpx.Client, url: str) -> tuple[str, str]:
    assert_public_host(url)
    with client.stream("GET", url) as r:
        if r.status_code >= 400:
            raise OpenerError(f"{url} returned HTTP {r.status_code}.")
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype:
            raise OpenerError(f"{url} isn't an HTML page ({ctype or 'unknown type'}).")
        body = b""
        for chunk in r.iter_bytes():
            body += chunk
            if len(body) > MAX_BYTES:
                break
        return str(r.url), body.decode(r.encoding or "utf-8", errors="replace")


def extract(html: str, url: str) -> tuple[Page, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

    title = (soup.title.string or "").strip() if soup.title else ""
    parts: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        t = " ".join(t.split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            parts.append(t)

    add(title)
    for name in ("description", "og:description", "twitter:description"):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            add(tag["content"])

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer"]):
        tag.decompose()
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if len(t) >= 20:
            add(t)

    text = "\n".join(parts)[:MAX_CHARS_PER_PAGE]
    return Page(url=url, title=title, text=text), links


def _matches(link: str, base_host: str | None, hints: tuple[str, ...]) -> bool:
    # Match the first path segment ("/about", "/our-team"), not substrings ("/customers/storyblok").
    p = urlparse(link)
    segments = [x for x in p.path.lower().split("/") if x]
    first = segments[0] if segments else ""
    return p.hostname == base_host and 0 < len(segments) <= 2 and any(
        first == h or first.startswith(h + "-") or first.endswith("-" + h) for h in hints
    )


def find_extra_pages(base_url: str, links: list[str]) -> list[str]:
    """One link per category (about, customers, pricing, blog), in priority order."""
    base_host = urlparse(base_url).hostname
    picked: list[str] = []
    for hints in PAGE_HINTS:
        for link in links:
            link = link.split("#")[0].split("?")[0]
            if link.rstrip("/") == base_url.rstrip("/") or link in picked:
                continue
            if _matches(link, base_host, hints):
                picked.append(link)
                break
    return picked[:MAX_EXTRA_PAGES]


def read_site(url: str) -> list[Page]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    # Connect over IPv4: on networks with broken IPv6, httpx waits ~20s per new connection
    # before giving up on the IPv6 address (seen on pirsch.io: 20.6s vs 0.6s over IPv4).
    transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=1)
    with httpx.Client(
        headers=headers, timeout=FETCH_TIMEOUT, follow_redirects=True, transport=transport
    ) as client:
        try:
            final_url, html = fetch_html(client, url)
        except httpx.HTTPError as e:
            raise OpenerError(f"Couldn't load {url} ({type(e).__name__}).")
        home, links = extract(html, final_url)
        pages = [home]

        def fetch_extra(link: str) -> Page | None:
            try:
                extra_url, extra_html = fetch_html(client, link)
                page = extract(extra_html, extra_url)[0]
                return page if len(page.text) >= 200 else None
            except (OpenerError, httpx.HTTPError):
                return None  # extra pages are a bonus, not required

        # Fetch the extra pages in parallel: slow sites otherwise add ~5s per page.
        extras = find_extra_pages(final_url, links)
        if extras:
            with ThreadPoolExecutor(max_workers=len(extras)) as pool:
                pages += [p for p in pool.map(fetch_extra, extras) if p]

    if sum(len(p.text) for p in pages) < MIN_USEFUL_CHARS:
        raise OpenerError(
            "The site has almost no readable text (it's probably built with JavaScript). "
            "Not guessing, so no opener was written."
        )
    return pages


# ---------- 4. Model ----------

class Candidate(BaseModel):
    hook: str               # the one specific detail this version is built on
    lines: list[str]
    facts_used: list[str]


class Draft(BaseModel):
    founder_name: str | None
    what_they_do: str
    candidates: list[Candidate]


SYSTEM = """You write the first 3 lines of a cold email from one founder to another.
The reader decides in two seconds whether this was written for them or for 500 people.
Use ONLY facts in the website text. Never invent customers, numbers, dates, funding or names.

Write 3 DIFFERENT candidate openers, each built on a different hook. A good hook is the most
specific thing on the site: a number, a named customer, an unusual choice they made (bootstrapped,
open source, no free plan, built in a weird place), a strong claim, or an exact phrase they use.
Order the candidates best first.

Each candidate is exactly 3 lines, one sentence each, at most 25 words per line:
- Line 1: state the hook plainly, as a fact about THEM. Specific enough that it could not be
  sent to any other company.
- Line 2: the implication: what that choice or number probably means for them (a trade-off,
  a pressure, a consequence). Show you thought about it; don't praise it.
- Line 3: one short, specific question about that hook that a founder could answer in a line.
  No pitch, no product, no meeting request.

Style: plain words, like a founder texting a peer. No flattery or hype ("impressive", "amazing",
"unique", "remarkable", "innovative", "game-changer"). Don't start with "I noticed", "I saw",
"I came across" or "Noticing". No greeting or sign-off.

GOOD (fictional company, for style only):
  "You've kept Ledgerly at a flat $19/month for four years while adding multi-currency support."
  "Holding price that long usually means the new features are paying for themselves through retention."
  "Is the flat price a deliberate positioning call, or something you plan to revisit?"
BAD: "Your innovative platform is impressive and I'd love to connect about your growth journey."

Also return:
- founder_name: only if the text names the founder/CEO/co-founder; otherwise null.
- what_they_do: one plain sentence.
- facts_used per candidate: 1-3 SHORT phrases (3-12 words) copied word-for-word from the website text.
"""


def build_prompt(pages: list[Page]) -> str:
    blocks = [f"--- PAGE: {p.url}\n{p.text}" for p in pages]
    return "Website text:\n\n" + "\n\n".join(blocks) + "\n\nWrite the 3 candidate openers."


# JSON schema for providers that take a raw schema (Groq strict mode needs every field
# required and additionalProperties false).
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "founder_name": {"type": ["string", "null"]},
        "what_they_do": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hook": {"type": "string"},
                    "lines": {"type": "array", "items": {"type": "string"}},
                    "facts_used": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["hook", "lines", "facts_used"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["founder_name", "what_they_do", "candidates"],
    "additionalProperties": False,
}

RETRY_WAITS = (2, 5, None)  # back off on transient 5xx errors, then give up


def _call_gemini(key: str, model: str, prompt: str) -> tuple[Draft, dict]:
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0.7,
        response_mime_type="application/json",
        response_schema=Draft,
        # Short writing task: low thinking cuts latency and output-token cost a lot.
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )
    for wait in RETRY_WAITS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            break
        except genai_errors.ClientError as e:
            if e.code == 429:
                if "per day" in str(e):
                    raise KeyProblem(
                        "Gemini's free daily request limit for this model is used up. "
                        "Switch to Groq or try again tomorrow."
                    )
                raise KeyProblem("Gemini rate limit hit (free tier). Wait a minute and try again.")
            if e.code in (400, 401, 403) and "key" in str(e).lower():
                raise KeyProblem("Gemini rejected the API key. Check that it was copied correctly.")
            raise OpenerError(f"Gemini rejected the request: {e.message}")
        except genai_errors.ServerError as e:
            if wait is None:
                raise OpenerError(f"Gemini is having trouble right now ({e.code}). Try again.")
            time.sleep(wait)

    draft = resp.parsed if isinstance(resp.parsed, Draft) else Draft(**json.loads(resp.text))
    u = resp.usage_metadata
    usage = {
        "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
        "output_tokens": (getattr(u, "candidates_token_count", 0) or 0)
        + (getattr(u, "thoughts_token_count", 0) or 0),
    }
    return draft, usage


def _call_groq(key: str, model: str, prompt: str) -> tuple[Draft, dict]:
    client = groq.Groq(api_key=key, max_retries=0)
    for wait in RETRY_WAITS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.7,
                reasoning_effort="low",  # same reason as Gemini: short task, keep it fast and cheap
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "draft", "strict": True, "schema": DRAFT_SCHEMA},
                },
            )
            break
        except groq.RateLimitError as e:
            if "per day" in str(e).lower():
                raise KeyProblem("Groq's free daily limit is used up. Switch to Gemini or try again tomorrow.")
            raise KeyProblem("Groq rate limit hit (free tier). Wait a minute and try again.")
        except groq.AuthenticationError:
            raise KeyProblem("Groq rejected the API key. Check that it was copied correctly.")
        except (groq.InternalServerError, groq.APIConnectionError) as e:
            if wait is None:
                raise OpenerError(f"Groq is having trouble right now ({type(e).__name__}). Try again.")
            time.sleep(wait)
        except groq.APIStatusError as e:
            raise OpenerError(f"Groq rejected the request ({e.status_code}): {e.message}")

    draft = Draft(**json.loads(resp.choices[0].message.content))
    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,  # includes reasoning tokens
    }
    return draft, usage


def call_model(provider: str, key: str, prompt: str, feedback: str | None) -> tuple[Draft, dict]:
    if feedback:
        prompt += f"\n\nYour previous attempt was rejected: {feedback}. Fix that."
    model = PROVIDERS[provider][0]
    if provider == "groq":
        return _call_groq(key, model, prompt)
    return _call_gemini(key, model, prompt)


# ---------- 5. Validation ----------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", " ".join(s.lower().split()))


NUMBER_RE = re.compile(
    r"\$?\d[\d,.]*\s*(?:%|percent|k\b|m\b|million|billion|x\b)?", re.I
)


def _num_key(s: str) -> str:
    return re.sub(r"[\s,$]", "", s.lower()).rstrip(".")


def ungrounded_numbers(lines: list[str], source_text: str) -> list[str]:
    """Numbers/amounts in the opener that never appear on the site (the most common hallucination)."""
    src = {_num_key(m.group(0)) for m in NUMBER_RE.finditer(source_text)}
    out = []
    for line in lines:
        for m in NUMBER_RE.finditer(line):
            if _num_key(m.group(0)) not in src:
                out.append(m.group(0).strip())
    return out


# Not fatal, but each one makes an opener read like a template; used to rank candidates.
GENERIC = [
    "impressive", "amazing", "unique", "remarkable", "remarkably", "innovative", "exciting",
    "cutting-edge", "seamless", "robust", "love to connect", "would love", "journey",
    "really interesting", "great work", "kudos", "rare", "significant leap",
]
STOPWORDS = set("a an the and or of to in on for with your you our we is are it its that this by as at from".split())


def validate(c: Candidate, source_text: str) -> str | None:
    """Return why a candidate is unusable, or None if it passes every hard rule."""
    lines = [l.strip() for l in c.lines if l.strip()]
    if len(lines) != 3:
        return f"it had {len(lines)} lines instead of 3"
    for l in lines:
        if len(l.split()) > MAX_WORDS_PER_LINE:
            return f"a line was over {MAX_WORDS_PER_LINE} words"
        hit = next((b for b in BANNED if b in l.lower()), None)
        if hit:
            return f"it used the cliché '{hit}'"
    bad_numbers = ungrounded_numbers(lines, source_text)
    if bad_numbers:
        return f"these numbers are not on the site: {bad_numbers}"
    if not c.facts_used:
        return "it cited no facts from the site"
    src = _norm(source_text)
    missing = [f for f in c.facts_used if _norm(f) not in src]
    if missing:
        return f"these 'facts' are not on the site: {missing}"
    return None


def _content_words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if w not in STOPWORDS and len(w) > 2}


def score(c: Candidate, source_text: str) -> float:
    """Higher = more specific and less template-like. Only applied to candidates that passed validate()."""
    text = " ".join(c.lines)
    low = text.lower()
    pts = 0.0
    pts += 2 * len(NUMBER_RE.findall(text))                          # concrete numbers (already grounded)
    pts += 1.5 * len(re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", text))  # names mid-sentence: customers, places
    fact_words = set().union(*(_content_words(f) for f in c.facts_used)) if c.facts_used else set()
    pts += 3 * (len(_content_words(c.lines[0]) & fact_words) >= 2)  # line 1 is built on a real fact
    pts += 1 * c.lines[2].strip().endswith("?")                      # ends with a real question
    pts -= 2 * sum(g in low for g in GENERIC)                        # hype / template words
    pts -= 1 * sum(len(l.split()) > 25 for l in c.lines)             # long lines read like marketing
    return pts


# ---------- Orchestration ----------

@dataclass
class Option:
    lines: list[str]
    hook: str
    facts_used: list[str]
    score: float


@dataclass
class Result:
    url: str
    lines: list[str]
    founder_name: str | None
    what_they_do: str
    facts_used: list[str]
    pages_read: list[str]
    attempts: int
    seconds: float
    provider: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    hook: str = ""
    alternatives: list[Option] = field(default_factory=list)
    thin_site: bool = False


def draft_opener(
    raw_url: str, api_key: str | None = None, provider: str | None = None, max_attempts: int = 2
) -> Result:
    start = time.time()
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise OpenerError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    model, key_env, key_site = PROVIDERS[provider]
    key = api_key or os.environ.get(key_env) or (
        os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None
    )
    if not key:
        raise KeyProblem(f"No {provider.title()} API key. Set {key_env} (free at {key_site}).")

    url = normalize_url(raw_url)
    pages = read_site(url)
    source = "\n".join(p.text for p in pages)

    prompt = build_prompt(pages)
    feedback, total = None, {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        draft, usage = call_model(provider, key, prompt, feedback)
        for k in total:
            total[k] += usage[k]

        good: list[Option] = []
        reasons: list[str] = []
        for rank, c in enumerate(draft.candidates[:5]):
            reason = validate(c, source)
            if reason:
                reasons.append(reason)
                continue
            lines = [l.strip() for l in c.lines if l.strip()]
            # tiny tie-breaker that respects the model's own "best first" order
            good.append(Option(lines, c.hook, c.facts_used, round(score(c, source) - 0.1 * rank, 1)))

        if good:
            good.sort(key=lambda o: o.score, reverse=True)
            best = good[0]
            founder = draft.founder_name
            if founder and _norm(founder) not in _norm(source):
                founder = None  # never greet someone by a name the site doesn't contain
            return Result(
                url=url,
                lines=best.lines,
                founder_name=founder,
                what_they_do=draft.what_they_do,
                facts_used=best.facts_used,
                pages_read=[p.url for p in pages],
                attempts=attempt,
                seconds=round(time.time() - start, 1),
                usage=total,
                provider=provider,
                model=model,
                hook=best.hook,
                alternatives=good[1:],
                thin_site=len(source) < THIN_SITE_CHARS,
            )
        feedback = "; ".join(sorted(set(reasons))) or "no candidates were returned"
    raise OpenerError(f"Couldn't write a grounded opener after {max_attempts} tries ({feedback}).")
