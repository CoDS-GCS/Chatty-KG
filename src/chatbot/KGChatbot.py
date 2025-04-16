import sys
sys.path.append("..")
from langchain.globals import set_debug
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from prompts import CLASSIFY_QUESTION_PROMPT_2, CONDENSE_QUESTION_PROMPT_CUSTOM
from chattykg.chattykg import ChattyKG
import json
import requests

# set_debug(True)

max_Vs = 1
max_Es = 21
max_answers = 41
limit_VQuery = 600
limit_EQuery = 300
chattykg_endpoint = "http://localhost:8899"

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

class KGChatbot:
    def __init__(self, kg_name, host):
        self.openai_model = "gpt-4o"
        self.chat_summary = ""
        self.kg_name = kg_name
        self.host = host
        self.setup_resources()
        self.store = {}

    def ask_question(self, session_id, question):
        retries = 3
        original_question = question

        while retries > 0:
            q_type = self.classify_question(session_id, question)
            print(f"predicted q_type : {q_type}")
            if "non-self-contained" in q_type.lower():
                question = self.rephrase_question(session_id, question)
                print("Rephrased Question: ", question)
                retries -= 1
                continue

            answer, values_answer = self.run_query(question)
            if not answer:
                return None, None
            chat_history = self.get_by_session_id(session_id)
            chat_history.add_messages([HumanMessage(original_question), AIMessage(question)])
            return answer, values_answer
        return None, None


    def setup_llm(_self):
        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            # streaming=True,
        )
        return model

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]


    def setup_resources(self):
        self.llm = self.setup_llm()
        self.chattykg_instance = ChattyKG(n_max_answers=max_answers, n_max_Vs=max_Vs, n_max_Es=max_Es, n_limit_VQuery=limit_VQuery, n_limit_EQuery=limit_EQuery)
        # self.conv_buff_memory = ConversationSummaryBufferMemory(
        #     llm=self.llm, memory_key="chat_history", max_token_limit=150
        # )
        string_output_parser = StrOutputParser()

        # Cache the LLMChain objects
        self.classify_question_chain = CLASSIFY_QUESTION_PROMPT_2 | self.llm | string_output_parser
        # self.classify_context_question_chain = CONTEXT_CLASSIFY_QUESTION_PROMPT_2 | self.llm | string_output_parser
        # self.classify_chain_with_history = RunnableWithMessageHistory(
        #     self.classify_context_question_chain,
        #     self.get_by_session_id,
        #     input_messages_key="question",
        #     history_messages_key="chat_history",
        # )

        self.rephrase_question_chain = CONDENSE_QUESTION_PROMPT_CUSTOM | self.llm | string_output_parser
        # self.rephrase_question_with_history_chain = RunnableWithMessageHistory(
        #     self.rephrase_question_chain,
        #     self.get_by_session_id,
        #     input_messages_key="question",
        #     history_messages_key="chat_history",
        # )
        # self.rephrase_chain = LLMChain(
        #     llm=self.llm,
        #     prompt=CONDENSE_QUESTION_PROMPT_CUSTOM,
        #     memory=self.conv_buff_memory,
        #     verbose=True,
        #     output_parser=string_output_parser,
        # )

    def get_active_kg(self):
        return self.chattykg_instance

    def get_llm(self):
        return self.llm

    def classify_question(self, session_id, question):
        # q_type = self.classify_question_chain.invoke(
        #     {"chat_history": self.chat_summary, "question": question}
        # )
        # q_type = self.classify_chain_with_history.invoke(
        #     {"question": question},
        #     config={'configurable': {'session_id': session_id}}
        # )
        q_type = self.classify_question_chain.invoke(
            {"question": question},
        )
        return q_type

    # Generate a new Chat summary after adding the last question-answer pair to the context
    def update_chat_summary(self,session_id):
        # self.chat_summary = self.conv_buff_memory.predict_new_summary(
        #     self.conv_buff_memory.chat_memory.messages, ""
        # )
        self.chat_summary = self.get_by_session_id(session_id)

    def rephrase_question(self, session_id, question):
        chat_history = self.get_by_session_id(session_id)
        question = self.rephrase_question_chain.invoke(
            {"question": question, "chat_history": chat_history},
        )
        return question
    
    def send_request(self, question, kg_name):
        payload = {
            "question": question,
            "knowledge_graph": kg_name,
            "max_answers": max_answers
        }
        resp = requests.post(chattykg_endpoint, data=json.dumps(payload))
        if resp.status_code != 200:
            print(f"ERROR: {resp.status_code}")
            return None, None
        else:
            results = json.loads(resp.text)
            print(results)
            return results.get("answers"), None


    # returns the structured output for evaluation and user values for history and chatbot interface
    def run_query(self, question):
        answers, _, _, understanding_time, linking_time, execution_time, query_selection_time, num_queries_executed, is_boolean = self.chattykg_instance.ask(question_text=question, question_id=0, knowledge_graph=self.kg_name)

        # answers, is_boolean = self.send_request(question, self.kg_name)
        
        if is_boolean:
            bool_value = False
            for answer in answers:
                bool_value = answer['boolean'] or bool_value
            user_values = [bool_value]
            output = [{'boolean': bool_value}]
        else:
            all_bindings = list()
            user_values = set()
            for answer in answers:
                if answer['results'] and answer['results']['bindings']:
                    all_bindings.extend(answer['results']['bindings'])

            for binding in all_bindings:
                key = list(binding.keys())[0]
                user_values.add(binding[key]['value'])
            user_values = list(user_values)
            print(user_values)
            output = [{'results': {'bindings': all_bindings}}]
        return output, user_values

    def update_context(self, session_id, question, answer):
        # self.conv_buff_memory.save_context(
        #     {"input": f"#Question: {question}"}, {"output": f"#Answer: {answer}"}
        # )
        history = self.get_by_session_id(session_id)
        # Add a sample from the answer to prevent overflowing the context length of the LLM.
        if len(answer) > 100:
            answer = answer[:100]
        history.add_messages([{"input": f"#Question: {question}"}, {"output": f"#Answer: {answer}"}])
        print("\n\n====HISTORY:====\n\n")
        print(f"{self.store}")