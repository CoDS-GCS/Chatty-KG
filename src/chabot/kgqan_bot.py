import langchain
import streamlit as st
# from KGChatbot_server import KGChatbot
from KGChatbot import KGChatbot
import utils
import os

# langchain.debug = True

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


class StreamlitKGChatbot:
    def __init__(self):
       self.openai_model = "gpt-3.5-turbo"
       self.host = st.session_state.kg_endpoint
       # self.conv_startwith_dependent_q = True
       self.kgchatbot = None


    def get_active_kg(self):
        return self.kgchatbot.get_active_kg()

    @utils.openai_api_key_needed
    @utils.enable_chat_history
    def main(self):
        self.kgchatbot = KGChatbot(st.session_state.kg_name.lower(), st.session_state.kg_endpoint)
        conv_startwith_dependent_q = True
        chat_summary = ""
        question = st.chat_input(placeholder="Ask me anything!")
        while question:
            utils.display_msg(question, "user")
            print(chat_summary)
            q_type = self.kgchatbot.classify_question(question)
            utils.display_msg(f"Question type: {q_type}", "assistant")

            if "non-self-contained" == q_type.strip().lower():
                if conv_startwith_dependent_q:
                    utils.display_msg(
                        "I couldn't understood your question, can you clarify.",
                        "assistant",
                    )
                    question = None
                    continue
                while "dependent" in q_type.lower():
                    self.kgchatbot.update_chat_summary()
                    question = self.kgchatbot.rephrase_question(question)
                    utils.display_msg(f"Rephrased Question: {question}", "assistant")
                    q_type = self.kgchatbot.classify_question(question)


            conv_startwith_dependent_q = False
            answer = self.kgchatbot.run_query(question)
            self.kgchatbot.update_context(question, answer)
            utils.display_msg(f"Answer: {answer}", "assistant")

            question = None


if __name__ == "__main__":
    obj = StreamlitKGChatbot()
    obj.main()
