
import os
import re
import traceback
from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq, BadRequestError, AuthenticationError

# --- Groq client setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except (ImportError, KeyError):
        pass

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Set it as an environment variable or in Streamlit secrets.")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "mixtral-8x7b-32768"

# --- Superhero definitions ---
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

# --- Improved hero identification ---
def identify_superhero(text: str) -> Optional[str]:
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    for key, hero in SUPERHEROES.items():
        if hero["name"].lower() in text_clean:
            return key
        for alias in hero.get("aliases", []):
            if alias in text_clean:
                return key
    return None

# --- Master Router Node ---
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

# --- Superhero Agent Node Factory ---
def create_hero_node(hero_key: str):
    def hero_node(state: AgentState) -> AgentState:
        hero = SUPERHEROES[hero_key]
        system_msg = {"role": "system", "content": hero["persona"]}
        # Build conversation history
        conversation = [{"role": m["role"], "content": m["content"]} for m in state["messages"]]
        messages = [system_msg] + conversation[-10:]  # last 10 for context

        # --- Call Groq with error logging ---
        try:
            print(f"Calling Groq with model {MODEL_NAME}, messages count: {len(messages)}")
            chat_completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7
            )
            reply = chat_completion.choices[0].message.content
        except AuthenticationError as e:
            print("AuthenticationError:", str(e))
            reply = "I'm having trouble authenticating. Please check my API key."
        except BadRequestError as e:
            # Log the full error details
            print("BadRequestError:", str(e))
            if hasattr(e, 'response') and e.response is not None:
                print("Response status:", e.response.status_code)
                print("Response body:", e.response.text)
            reply = "Sorry, I encountered a technical issue. Please try again."
        except Exception as e:
            print("Unexpected error:", traceback.format_exc())
            reply = "Sorry, something went wrong. Please try again later."

        state["messages"].append({"role": "assistant", "content": reply})
        return state
    return hero_node

# Create hero nodes
ironman_node = create_hero_node("ironman")
spiderman_node = create_hero_node("spiderman")
captainamerica_node = create_hero_node("captainamerica")

# --- Build Graph ---
builder = StateGraph(AgentState)
builder.add_node("master_router", master_router)
builder.add_node("ironman", ironman_node)
builder.add_node("spiderman", spiderman_node)
builder.add_node("captainamerica", captainamerica_node)
builder.set_entry_point("master_router")

def route_to_hero(state: AgentState) -> str:
    if state["call_active"] and state["current_hero"]:
        return state["current_hero"]
    else:
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
    new_state = graph.invoke(state)
    return new_state
