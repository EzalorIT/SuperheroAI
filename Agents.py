import openai
from typing import List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# --- Configuration for free model (Ollama) ---
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "dummy"
MODEL_NAME = "llama3.2"   # or any model you pulled

# --- Superhero definitions ---
SUPERHEROES = {
    "ironman": {
        "name": "Iron Man",
        "persona": "You are Tony Stark, billionaire genius in a high‑tech suit. Witty, sarcastic, but always ready to help.",
        "image": "https://via.placeholder.com/400?text=Iron+Man"
    },
    "spiderman": {
        "name": "Spider-Man",
        "persona": "You are Peter Parker, a friendly neighbourhood Spider‑Man. Energetic, a bit nervous, but heroic.",
        "image": "https://via.placeholder.com/400?text=Spider-Man"
    },
    "captainamerica": {
        "name": "Captain America",
        "persona": "You are Steve Rogers, Captain America. Honest, brave, and always speaks with moral clarity.",
        "image": "https://via.placeholder.com/400?text=Captain+America"
    }
}

# --- Graph State ---
class AgentState(TypedDict):
    messages: List[dict]          # conversation history [{"role": "user"/"assistant", "content": ...}]
    current_hero: Optional[str]    # key of active hero (None if no call)
    call_active: bool              # whether a call is in progress

# --- Helper: Identify hero from user message ---
def identify_superhero(text: str) -> Optional[str]:
    text_lower = text.lower()
    for key, hero in SUPERHEROES.items():
        if hero["name"].lower() in text_lower:
            return key
    return None

# --- Master Router Node ---
def master_router(state: AgentState) -> AgentState:
    # Get the last user message
    last_msg = state["messages"][-1]["content"]
    hero_key = identify_superhero(last_msg)
    
    if hero_key:
        # If a hero is mentioned, activate the call (or switch heroes)
        state["current_hero"] = hero_key
        state["call_active"] = True
        # No need to greet here; the hero agent will respond.
        # We'll let the hero agent generate the greeting.
    else:
        # If no hero identified, ask for clarification
        state["messages"].append({
            "role": "assistant",
            "content": "I'm not sure which hero you'd like to call. Please say, for example, 'I want to talk to Iron Man'."
        })
    return state

# --- Superhero Agent Nodes (one per hero) ---
def create_hero_node(hero_key: str):
    """Factory to create a node function for a specific superhero."""
    def hero_node(state: AgentState) -> AgentState:
        hero = SUPERHEROES[hero_key]
        # Build messages for the LLM
        system_msg = {"role": "system", "content": hero["persona"]}
        # Convert state messages to the format expected by the LLM
        conversation = [{"role": m["role"], "content": m["content"]} for m in state["messages"]]
        # Keep last 10 messages for context
        messages = [system_msg] + conversation[-10:]
        
        # Call the LLM
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        
        # Append the reply to state
        state["messages"].append({"role": "assistant", "content": reply})
        return state
    return hero_node

# Create individual hero nodes
ironman_node = create_hero_node("ironman")
spiderman_node = create_hero_node("spiderman")
captainamerica_node = create_hero_node("captainamerica")

# --- Build the Graph ---
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("master_router", master_router)
builder.add_node("ironman", ironman_node)
builder.add_node("spiderman", spiderman_node)
builder.add_node("captainamerica", captainamerica_node)

# Set entry point
builder.set_entry_point("master_router")

# Conditional edges from master_router
def route_to_hero(state: AgentState) -> Literal["ironman", "spiderman", "captainamerica", "end"]:
    if state["call_active"] and state["current_hero"]:
        return state["current_hero"]   # node name matches hero key
    elif not state["call_active"]:
        # No hero identified, end this run (wait for next user input)
        return "end"
    else:
        # Should not happen, but fallback
        return "end"

builder.add_conditional_edges("master_router", route_to_hero)

# After a hero responds, go back to master router to handle next user input
builder.add_edge("ironman", "master_router")
builder.add_edge("spiderman", "master_router")
builder.add_edge("captainamerica", "master_router")

# Compile
graph = builder.compile()

# --- Public interface for Streamlit ---
def run_agent(user_input: str, state: AgentState) -> AgentState:
    # Append user message
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append({"role": "user", "content": user_input})
    
    # Run the graph
    new_state = graph.invoke(state)
    return new_state
