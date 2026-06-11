import streamlit as st
from llm_manager import ChatbotManager

# Set page layout to wide and configure titles
st.set_page_config(page_title="AI Companion", page_icon="💬", layout="wide")
st.title("💬 Modern AI Companion")

# Initialize our LangChain manager in Streamlit cache so it doesn't reload on every click
@st.cache_resource
def load_chatbot_manager():
    return ChatbotManager()

bot_manager = load_chatbot_manager()

# --- Session State Initialization ---
if "sessions" not in st.session_state:
    # Key: session_id, Value: List of dicts representing messages [{"role": "user", "content": "..."}]
    st.session_state.sessions = {"Default Session": []}

if "current_session" not in st.session_state:
    st.session_state.current_session = "Default Session"

# --- Sidebar Management ---
with st.sidebar:
    st.header("Chat Sessions")
    
    # Input box to create a brand new conversation channel
    new_session_name = st.text_input("Create new session:", placeholder="Enter session name...").strip()
    if st.button("➕ Add Session", use_container_width=True):
        if new_session_name and new_session_name not in st.session_state.sessions:
            st.session_state.sessions[new_session_name] = []
            st.session_state.current_session = new_session_name
            st.rerun()

    st.write("---")
    
    # Selection radio buttons to switch across existing sessions
    session_list = list(st.session_state.sessions.keys())
    selected_session = st.radio(
        "Select Active Session:", 
        session_list, 
        index=session_list.index(st.session_state.current_session)
    )
    
    # Update active tracking if the user switches choices
    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        st.rerun()

# --- Chat Interface (ChatGPT Aesthetic) ---
current_chat_history = st.session_state.sessions[st.session_state.current_session]

# Render historical messages from the active chat session array
for message in current_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture real-time user input using the native chat box element
if user_query := st.chat_input("Type your message here..."):
    
    # Display the user's message immediately on screen
    with st.chat_message("user"):
        st.markdown(user_query)
    current_chat_history.append({"role": "user", "content": user_query})

    # Generate reply through LangChain and stream/render under an assistant card
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The backend gets the query tied specifically to the sidebar's session name
            ai_response = bot_manager.get_response(user_query, session_id=st.session_state.current_session)
            st.markdown(ai_response)
            
    # Commit the AI reply to session history tracking arrays
    current_chat_history.append({"role": "assistant", "content": ai_response})
