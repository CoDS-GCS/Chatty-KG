from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List

from chattykg.chattykg import knowledge_graph_to_uri
from chattykg.json_logger import JsonLogger
from chattykg.sparql_end_points.EndPoint import EndPoint
from chattykg.sparql_end_points.XML_EndPoint import XML_EndPoint
from dotenv import load_dotenv
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
import os
from llm_config.llm_setup import get_llm

# from pathlib import Path

# def get_openai_key():
#     current_path = Path(__file__).resolve()

#     for parent in [current_path] + list(current_path.parents):
#         env_path = parent / ".env"
#         if env_path.exists():
#             with env_path.open("r") as f:
#                 for line in f:
#                     if line.strip().startswith("OPENAI_API_KEY="):
#                         return line.strip().split("=", 1)[1].strip().strip('"').strip("'")

#     raise ValueError("OPENAI_API_KEY not found in any .env file up the directory tree.")





class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class State:
    def __init__(self, knowledge_graph, n_limit_VQuery, n_max_Vs, n_limit_EQuery, n_max_Es, n_max_answers, filtration_enabled):

        self.string_output_parser = StrOutputParser()
        self.chatbot_llm = get_llm(model_name="gpt-4o")

        self.store = {}
        self.json_logger = JsonLogger()
        
        # Initialize all attributes before using them
        self.filtration_enabled = filtration_enabled
        self.knowledge_graph = knowledge_graph
        self.n_max_Vs = n_max_Vs
        self.n_max_Es = n_max_Es
        self.n_limit_VQuery = n_limit_VQuery
        self.n_limit_EQuery = n_limit_EQuery
        self.n_max_answers = n_max_answers
        
        # Initialize objects that depend on other attributes
        self.question = None
        self.target_variable = None
        self.sparql_end_point = self.set_up_sparql_endpoint(knowledge_graph)


    def set_up_sparql_endpoint(self, knowledge_graph):
        if knowledge_graph in ["open_citations"]:
            sparql_end_point = XML_EndPoint(
                knowledge_graph, knowledge_graph_to_uri[knowledge_graph], self.filtration_enabled
            )
        else:
            sparql_end_point = EndPoint(
                knowledge_graph, knowledge_graph_to_uri[knowledge_graph], self.filtration_enabled
            )
        return sparql_end_point

    def add_possible_answer(self, query, score, node_uris, relation_uris, triples):
        self.question.add_possible_answer(
            question=self.question.text, sparql=query, score=score, nodes=node_uris, edges=relation_uris,
            triples=triples
        )

    def set_question(self, question):
        self.question = question


    def set_target_variable(self, target_variable):
        self.target_variable = target_variable

    def get_chatbot_llm(self):
        return self.chatbot_llm

    def get_string_output_parser(self):
        return self.string_output_parser

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]

    def get_json_logger(self):
        return self.json_logger

    def get_query_graph(self):
        return self.question.query_graph

    def get_knowledge_graph(self):
        return self.knowledge_graph

    def get_n_limit_VQuery(self):
        return self.n_limit_VQuery

    def get_n_max_Vs(self):
        return self.n_max_Vs

    def get_n_limit_EQuery(self):
        return self.n_limit_EQuery

    def get_n_max_Es(self):
        return self.n_max_Es

    def get_sparql_end_point(self):
        return self.sparql_end_point

    def get_target_variable(self):
        return self.target_variable

    def get_question_text(self):
        return self.question.text

    def get_answers(self):
        return self.question.possible_answers[: self.n_max_answers]

    def get_answers_at_index(self, index):
        return self.question.possible_answers[index]

    def get_answer_datatype(self):
        return self.question.answer_datatype

    #  Adding this to not break the code, but it looks like that we only use the boolean datatype in our current version of the system.
    def detect_question_and_answer_type(self):
        if not self.question.answer_datatype:
            self.question.answer_type = "string"
            self.question.answer_datatype = "string"

        if self.question.text.lower().startswith("who was"):
            self.question.answer_type = "person"
            self.question.answer_datatype = "resource"
        elif self.question.text.lower().startswith("who is "):
            self.question.answer_type = "person"
            self.question.answer_datatype = "resource"
        elif self.question.text.lower().startswith("are "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("is "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("did "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("do "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("does "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("was "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("were "):
            self.question.answer_type = "boolean"
            self.question.answer_datatype = "boolean"
        elif self.question.text.lower().startswith("who are "):
            self.question.answer_type = "person"
            self.question.answer_datatype = "list"
        elif self.question.text.lower().startswith("who "):  # Who [V]
            self.question.answer_type = "person"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("whom "):  # Who [V]
            self.question.answer_type = "person"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("how many "):
            self.question.answer_type = "count"
            self.question.answer_datatype = "number"
        elif self.question.text.lower().startswith("how much "):
            self.question.answer_type = "price"
            self.question.answer_datatype = "number"
        elif self.question.text.lower().startswith(
            "when did "
        ) or self.question.text.lower().startswith("when was "):
            self.question.answer_type = "date"
            self.question.answer_datatype = "date"
        elif self.question.text.lower().startswith("when "):
            self.question.answer_type = "date"
            self.question.answer_datatype = "date"
        elif self.question.text.lower().startswith("which airports "):  # where do
            self.question.answer_type = "place"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("which languages "):  # where do
            self.question.answer_type = "language"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("what languages "):  # where do
            self.question.answer_type = "language"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("which countries "):  # where do
            self.question.answer_type = "place"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith(
            "in which "
        ):  # In which [NNS], In which city
            self.question.answer_type = "place"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith(
            "which "
        ):  # which [NNS], which actors
            self.question.answer_type = "other"
            self.question.answer_datatype = "list"  # of list
        elif self.question.text.lower().startswith("where "):  # where do
            self.question.answer_type = "place"
            self.question.answer_datatype = "resource"  # of list
        elif self.question.text.lower().startswith("show "):  # Show ... all
            self.question.answer_type = "other"
            self.question.answer_datatype = "list"  # of list
        else:
            pass

