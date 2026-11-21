import os

import streamlit as st

from opener import PROVIDERS, OpenerError, draft_opener

st.set_page_config(page_title="Cold Opener", page_icon="✉️")
st.title("Cold email opener")
st.caption("Paste a company's website. Get a 3-line opener for its founder, grounded in what the site says.")

LABELS = {"gemini": "Google Gemini", "groq": "Groq"}


def server_key(env_name: str) -> str:
    """Key from env/.env locally, or Streamlit secrets on the hosted demo."""
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        return st.secrets.get(env_name, "")
    except Exception:  # no secrets.toml when running locally
        return ""


provider = st.radio(
    "AI provider",
    list(PROVIDERS),
    format_func=lambda p: f"{LABELS[p]} ({PROVIDERS[p][0]})",
    horizontal=True,
)
model, key_env, key_site = PROVIDERS[provider]
default_key = server_key(key_env)

with st.expander(f"{LABELS[provider]} API key"):
    user_key = st.text_input("Key", type="password", help=f"Free at {key_site}", key=f"key_{provider}")

with st.form("go"):
    url = st.text_input("Website", placeholder="e.g. plausible.io")
    submitted = st.form_submit_button("Write opener")

if submitted:
    try:
        with st.spinner(f"Reading the site and drafting with {LABELS[provider]}..."):
            r = draft_opener(url, api_key=user_key or default_key or None, provider=provider)
    except OpenerError as e:
        st.error(str(e))
    else:
        greeting = f"Hi {r.founder_name.split()[0]}," if r.founder_name else "Hi there,"
        st.subheader("Opener")
        if r.thin_site:
            st.warning(
                "This site has very little text, so the opener may read generic. "
                "Check it before sending, or add a detail you know about them."
            )
        st.code(greeting + "\n\n" + "\n".join(r.lines), language=None, wrap_lines=True)
        st.caption(f"Built on: {r.hook}")
        if r.alternatives:
            with st.expander(f"Other options ({len(r.alternatives)})"):
                for alt in r.alternatives:
                    st.caption(f"Built on: {alt.hook}")
                    st.code(greeting + "\n\n" + "\n".join(alt.lines), language=None, wrap_lines=True)
        with st.expander("Why these lines (sources and stats)"):
            st.write(f"**What they do:** {r.what_they_do}")
            st.write("**Facts used, all checked against the site text:**")
            for f in r.facts_used:
                st.write(f"- “{f}”")
            st.write("**Pages read:** " + ", ".join(r.pages_read))
            st.write(
                f"{LABELS[r.provider]} `{r.model}` · {r.attempts} attempt(s) · {r.seconds}s · "
                f"{r.usage['input_tokens']} input / {r.usage['output_tokens']} output tokens"
            )
