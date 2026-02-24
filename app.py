import streamlit as st
import io
import base64
from gtts import gTTS
from agents import run_agent, SUPERHEROES   # your AI layer

# --- Helper: Text-to-Speech ---
def text_to_speech(text, lang="en"):
    """Convert text to speech and return audio bytes."""
    tts = gTTS(text=text, lang=lang, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# --- Streamlit UI ---
st.set_page_config(page_title="Superhero Video Call", layout="wide")
st.title("🦸‍♂️ Animated Superhero Video Call")

# Initialize session state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "current_hero": None,
        "call_active": False
    }
if "last_voiced_index" not in st.session_state:
    st.session_state.last_voiced_index = -1  # index of last message that was spoken

# Sidebar with superhero buttons
st.sidebar.header("Choose a Hero to Call")
for hero_key, hero in SUPERHEROES.items():
    if st.sidebar.button(hero["name"], key=hero_key):
        # Clear previous conversation and start a new call
        st.session_state.agent_state = {
            "messages": [],
            "current_hero": None,
            "call_active": False
        }
        # Simulate user requesting this hero
        simulated_message = f"I want to talk to {hero['name']}"
        with st.spinner(f"Calling {hero['name']}..."):
            new_state = run_agent(simulated_message, st.session_state.agent_state)
        st.session_state.agent_state = new_state
        st.session_state.last_voiced_index = -1  # reset voice tracker
        st.rerun()

# Main layout: video feed on left, chat on right
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Video Feed")
    if st.session_state.agent_state["current_hero"]:
        hero_key = st.session_state.agent_state["current_hero"]
        hero = SUPERHEROES[hero_key]
        # Display animation (GIF or video)
        media_path = f"assets/{hero_key}.gif"  # adjust as needed
        try:
            st.image(media_path, use_container_width=True)
        except:
            st.image(hero["image"], use_container_width=True)  # fallback static
    else:
        st.info("No active call. Click a hero in the sidebar to start!")

with col2:
    st.subheader("Chat")
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.agent_state["messages"]):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # Auto-play voice for new assistant messages
                if (msg["role"] == "assistant" and 
                    i > st.session_state.last_voiced_index):
                    audio_bytes = text_to_speech(msg["content"])
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    st.session_state.last_voiced_index = i

# End call button
if st.session_state.agent_state["call_active"]:
    if st.button("End Call"):
        st.session_state.agent_state = {
            "messages": [],
            "current_hero": None,
            "call_active": False
        }
        st.session_state.last_voiced_index = -1
        st.rerun()
