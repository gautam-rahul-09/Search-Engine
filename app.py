import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Tavily search tool
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    st.error("⚠️ Please add your Tavily API key in the .env file as TAVILY_API_KEY")
search = TavilySearchResults(max_results=5)

# Streamlit app UI
st.title("🔎 LangChain - Chat with Search (Tavily Powered)")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions 
of an agent in an interactive Streamlit app.
"""

# Sidebar for setting
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter Your Groq API key: ", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar")
    else:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            streaming=True
        )

        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            except Exception as e:
                response = f"⚠️ Search failed: {e}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
