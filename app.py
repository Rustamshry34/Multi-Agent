import streamlit as st
from typing import Dict, Any
from SmolManus import manager_agent
from pathlib import Path

st.set_page_config(
    page_title="Manus AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

def call_manager_agent(query: str):
    """Call the manager agent and return the result"""
    try:
        result = manager_agent.run(query)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

def render_output(content: Any):
    """Render agent output automatically depending on type"""
    # Case 1: Text / Markdown
    if isinstance(content, str):
        if content.strip().startswith("http"):
            st.markdown(f"[🔗 Link]({content})")
        elif Path(content).suffix in [".zip", ".pkl", ".csv", ".xlsx", ".json", ".txt", ".py"]:
            # File path returned
            file_path = Path(content)
            if file_path.exists():
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"📂 Download {file_path.name}",
                        data=f,
                        file_name=file_path.name
                    )
            else:
                st.warning(f"⚠️ File not found: {file_path}")
        else:
            st.markdown(content)

    # Case 2: Images
    elif "PIL" in str(type(content)):
        st.image(content, caption="Generated Image", use_container_width=True)

    # Case 3: Matplotlib figures
    elif "matplotlib" in str(type(content)):
        st.pyplot(content)

    # Case 4: Dict / JSON-like (show nicely)
    elif isinstance(content, dict):
        st.json(content)

    # Case 5: List of results (recursively render)
    elif isinstance(content, list):
        for item in content:
            render_output(item)

    # Fallback
    else:
        st.write(content)

def display_message(message: Dict[str, Any]):
    """Display chat messages with auto-rendered assistant outputs"""
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        with st.chat_message("assistant"):
            render_output(message["content"])

def main():
    """Main chatbot loop"""
    st.title("🤖 Manus AI Assistant")

    for message in st.session_state.messages:
        display_message(message)

    user_input = st.chat_input("Ask Manus AI...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_queries += 1

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_manager_agent(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                render_output(response)

if __name__ == "__main__":
    main()