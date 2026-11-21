import os

import streamlit as st

from opener import PROVIDERS, KeyProblem, OpenerError, draft_opener

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
key_issue = st.session_state.get(f"key_issue_{provider}")  # set when the last run hit a key/limit problem

# Ask for the visitor's own key when the app has none, or when the last run hit a limit.
# Drawn in a placeholder so it can be cleared once a run succeeds.
notice = st.empty()
if key_issue:
    notice.warning(
        f"{key_issue}\n\nPaste your own free {LABELS[provider]} API key below "
        f"(get one at {key_site}), then click **Write opener** again. "
        "Or switch to the other AI provider."
    )
elif not default_key:
    notice.info(f"Paste a free {LABELS[provider]} API key below to start (get one at {key_site}).")

# The key box lives inside the form so a pasted key is sent together with the website
# when the button is clicked (fields outside a form only register after Enter/blur).
with st.form("go"):
    url = st.text_input("Website", placeholder="e.g. plausible.io")
    with st.expander(
        f"Use your own {LABELS[provider]} API key" + (" (optional)" if default_key else ""),
        expanded=bool(key_issue) or not default_key,
    ):
        user_key = st.text_input(
            "Your API key", type="password", help=f"Free at {key_site}", key=f"key_{provider}"
        ).strip()
        st.caption(
            "If you add a key, it is used instead of the app's key. "
            "It stays in this browser session only and is never saved."
        )
    submitted = st.form_submit_button("Write opener")

if submitted:
    using_own_key = bool(user_key)
    try:
        with st.spinner(f"Reading the site and drafting with {LABELS[provider]}..."):
            r = draft_opener(url, api_key=user_key or default_key or None, provider=provider)
    except KeyProblem as e:
        msg = str(e)
        if using_own_key:
            msg = "Your key didn't work: " + msg
        elif default_key:
            msg = "The app's shared free key can't be used right now. " + msg
        st.session_state[f"key_issue_{provider}"] = msg
        st.rerun()  # redraw the page with the key box open and the explanation on top
    except OpenerError as e:
        st.error(str(e))
    else:
        st.session_state.pop(f"key_issue_{provider}", None)
        notice.empty()
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
                f"{LABELS[r.provider]} `{r.model}` · "
                f"{'your API key' if using_own_key else 'the app API key'} · {r.attempts} attempt(s) · {r.seconds}s · "
                f"{r.usage['input_tokens']} input / {r.usage['output_tokens']} output tokens"
            )
