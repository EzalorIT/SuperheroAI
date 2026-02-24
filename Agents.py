import os
import google.generativeai as genai
from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# --- Gemini setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Get from aistudio.google.com
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except (ImportError, KeyError):
        pass

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Get one at https://aistudio.google.com")

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash"  # Fast, free, high limits

# --- Superhero definitions (same as before) ---
SUPERHEROES = {
    "ironman": {
        "name": "Iron Man",
        "persona": "You are Tony Stark, billionaire genius in a high‑tech suit. Witty, sarcastic, but always ready to help.",
        "image": "https://via.placeholder.com/400?text=Iron+Man",
        "aliases": ["iron man", "tony stark", "stark"]
    },
    "spiderman": {
        "name": "Spider-Man",
        "persona": "You are Peter Parker, a friendly neighbourhood Spider‑Man. Energetic, a bit nervous, but heroic.",
        "image": "https://via.placeholder.com/400?text=Spider-Man",
        "aliases": ["spider man", "spiderman", "peter parker", "peter"]
    },
    "captainamerica": {
        "name": "Captain America",
        "persona": "You are Steve Rogers, Captain America. Honest, brave, and always speaks with moral clarity.",
        "image": "https://via.placeholder.com/400?text=Captain+America",
        "aliases": ["captain america", "steve rogers", "captain", "cap"]
    }
}

# --- Graph State ---
class AgentState(TypedDict):
    messages: List[dict]
    current_hero: Optional[str]
    call_active: bool

# --- Helper: Identify hero (unchanged) ---
def identify_superhero(text: str) -> Optional[str]:
    text_lower = text.lower()
    import re
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    for key, hero in SUPERHEROES.items():
        if hero["name"].lower() in text_clean:
            return key
        for alias in hero.get("aliases", []):
            if alias in text_clean:
                return key
    return None

# --- Master Router Node (unchanged) ---
def master_router(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]["content"]
    hero_key = identify_superhero(last_msg)
    if hero_key:
        state["current_hero"] = hero_key
        state["call_active"] = True
    else:
        state["messages"].append({
            "role": "assistant",
            "content": "I'm not sure which hero you'd like to call. Please say, for example, 'I want to talk to Spider-Man'."
        })
    return state

# --- Superhero Node Factory (UPDATED for Gemini) ---
def create_hero_node(hero_key: str):
    def hero_node(state: AgentState) -> AgentState:
        hero = SUPERHEROES[hero_key]
        
        # Build conversation history for Gemini
        conversation = []
        for msg in state["messages"][-10:]:  # last 10 messages
            role = "user" if msg["role"] == "user" else "model"
            conversation.append({"role": role, "content": msg["content"]})
        
        # Create the model with system prompt
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=hero["persona"]
        )
        
        # Start chat with history
        chat = model.start_chat(history=conversation[:-1])  # all but last
        
        try:
            # Send the last user message
            response = chat.send_message(conversation[-1]["content"])
            reply = response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            reply = "Sorry, I encountered an issue. Please try again."
        
        state["messages"].append({"role": "assistant", "content": reply})
        return state
    return hero_node

# Create hero nodes
ironman_node = create_hero_node("ironman")
spiderman_node = create_hero_node("spiderman")
captainamerica_node = create_hero_node("captainamerica")

# --- Build Graph (unchanged) ---
builder = StateGraph(AgentState)
builder.add_node("master_router", master_router)
builder.add_node("ironman", ironman_node)
builder.add_node("spiderman", spiderman_node)
builder.add_node("captainamerica", captainamerica_node)
builder.set_entry_point("master_router")

def route_to_hero(state: AgentState) -> str:
    if state["call_active"] and state["current_hero"]:
        return state["current_hero"]
    return END

builder.add_conditional_edges("master_router", route_to_hero)
builder.add_edge("ironman", "master_router")
builder.add_edge("spiderman", "master_router")
builder.add_edge("captainamerica", "master_router")

graph = builder.compile()

# --- Public interface ---
def run_agent(user_input: str, state: AgentState) -> AgentState:
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append({"role": "user", "content": user_input})
    return graph.invoke(state)
