import sys

sys.path.append('..')

import langchain
import streamlit as st
# from KGChatbot_server import KGChatbot
from KGChatbot import KGChatbot
import utils
import os

# langchain.debug = True
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="KG Chatbot", page_icon="🤖")
st.header("Question answering on KG")


# Fixed list of project names
knowledge_graphs = ["DBPedia", "YAGO", "DBLP"]

if "project_name" not in st.session_state:
    st.session_state.kg_name = ""
if "kg_endpoint" not in st.session_state:
    st.session_state.kg_endpoint = ""


# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Dropdown for Project Name (stores in session state)
    st.session_state.kg_name = st.selectbox(
        "Select Knowledge Graph",
        knowledge_graphs,
        index=(
            knowledge_graphs.index(st.session_state.kg_name)
            if st.session_state.kg_name in knowledge_graphs
            else 0
        ),
    )

    # st.session_state.kg_endpoint = st.text_input(
    #     label="Enter KG Endpoint URL",
    #     value=st.session_state.kg_endpoint,
    #     placeholder="http://",
    # )


session_id = "1"

if "kgchatbot" not in st.session_state:
    kgchatbot = KGChatbot(st.session_state.kg_name.lower(), st.session_state.kg_endpoint)
    st.session_state["kgchatbot"] = kgchatbot
else:
    kgchatbot = st.session_state["kgchatbot"]

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

def get_active_kg():
    return kgchatbot.get_active_kg()

# @utils.openai_api_key_needed
# @utils.enable_chat_history

if "conv_startwith_dependent_q" not in st.session_state:
    st.session_state["conv_startwith_dependent_q"] = True

chat_summary = ""
question = st.chat_input(placeholder="Ask me anything!")
while question:
    utils.display_msg(question, "user")
    print(chat_summary)
    q_type = kgchatbot.classify_question(session_id, question)
    utils.display_msg(f"Question type: {q_type}", "assistant")

    if "non-self-contained" == q_type.strip().lower():
        if st.session_state["conv_startwith_dependent_q"]:
            utils.display_msg(
                "I couldn't understood your question, can you clarify.",
                "assistant",
            )
            question = None
            continue
        # while "non-self-contained" == q_type.strip().lower():
            # kgchatbot.update_chat_summary(session_id)
        question = kgchatbot.rephrase_question(session_id, question)
        utils.display_msg(f"Rephrased Question: {question}", "assistant")
        # q_type = kgchatbot.classify_question(session_id, question)


    st.session_state["conv_startwith_dependent_q"] = False
    answer = kgchatbot.run_query(question)
    kgchatbot.update_context(session_id, question, answer)
    utils.display_msg(f"Answer: {answer}", "assistant")

    question = None


test_questions = [
    "What is the birth place of Antony Cheng?",
    "How many former coaches does he have?",
    "What is his gender?",
    "What is his choreographer?",
    "What is his country of residence?"
]
