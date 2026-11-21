"""Usage: python cli.py [--groq | --gemini] stripe.com [more-urls...]
       add --show-text to see what was scraped (no AI call)"""

import sys

from opener import OpenerError, draft_opener, normalize_url, read_site


def main(argv: list[str]) -> int:
    show_text = "--show-text" in argv
    provider = "groq" if "--groq" in argv else "gemini" if "--gemini" in argv else None
    urls = [a for a in argv if not a.startswith("--")]
    if not urls:
        print(__doc__)
        return 1
    code = 0
    for raw in urls:
        print(f"\n=== {raw}")
        try:
            if show_text:
                for p in read_site(normalize_url(raw)):
                    print(f"--- {p.url} ({len(p.text)} chars)\n{p.text[:1500]}\n")
                continue
            r = draft_opener(raw, provider=provider)
        except OpenerError as e:
            print(f"ERROR: {e}")
            code = 2
            continue
        if r.thin_site:
            print("  (warning: the site has little text, so this may read generic)")
        print("\n".join(r.lines))
        print(f"\n  hook:    {r.hook}")
        print(f"  founder: {r.founder_name or 'not named on site'}")
        print(f"  facts:   {r.facts_used}")
        for i, alt in enumerate(r.alternatives, 2):
            print(f"  option {i} (score {alt.score}): " + " / ".join(alt.lines))
        print(f"  pages:   {r.pages_read}")
        print(f"  {r.provider} {r.model}: {r.attempts} attempt(s), {r.seconds}s, tokens {r.usage}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
