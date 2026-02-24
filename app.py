import streamlit as st
from agents import run_agent, SUPERHEROES   # import from your AI layer

# Add animation paths to SUPERHEROES (or define separately)
# For this example, we'll extend the dictionary locally
HERO_MEDIA = {
    "ironman": "assets/ironman.gif",          # or .mp4
    "spiderman": "assets/spiderman.gif",
    "captainamerica": "assets/captainamerica.gif"
}

# Function to display animated video
def display_hero_animation(hero_key):
    media_path = HERO_MEDIA.get(hero_key)
    if not media_path:
        # Fallback to static image if no animation found
        st.image(SUPERHEROES[hero_key]["image"], use_container_width=True)
        return

    # Check file extension
    if media_path.endswith('.gif'):
        st.image(media_path, use_container_width=True)
    elif media_path.endswith(('.mp4', '.webm', '.ogg')):
        # Embed HTML5 video with autoplay and loop
        video_html = f"""
        <video width="100%" autoplay loop muted>
            <source src="{media_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        st.components.v1.html(video_html, height=400)  # adjust height as needed
    else:
        # Fallback to static image
        st.image(SUPERHEROES[hero_key]["image"], use_container_width=True)

# Streamlit UI
st.set_page_config(page_title="Superhero Video Call", layout="wide")
st.title("🦸‍♂️ Animated Superhero Video Call")

# Sidebar with available heroes
st.sidebar.header("Available Heroes")
for hero in SUPERHEROES.values():
    st.sidebar.write(f"- {hero['name']}")

# Initialize agent state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "current_hero": None,
        "call_active": False
    }

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Video Feed")
    if st.session_state.agent_state["current_hero"]:
        hero_key = st.session_state.agent_state["current_hero"]
        display_hero_animation(hero_key)
    else:
        st.info("No active call. Type a message to start!")

with col2:
    st.subheader("Chat")
    for msg in st.session_state.agent_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Your message..."):
    with st.spinner("Thinking..."):
        new_state = run_agent(prompt, st.session_state.agent_state)
    st.session_state.agent_state = new_state
    st.rerun()

# End call button
if st.session_state.agent_state["call_active"]:
    if st.button("End Call"):
        st.session_state.agent_state = {
            "messages": [],
            "current_hero": None,
            "call_active": False
        }
        st.rerun()
