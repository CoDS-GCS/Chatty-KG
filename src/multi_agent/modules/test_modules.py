import unittest
import sys
import os

# Add the src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, src_dir)

from .modules import (
    classify_question,
    get_qir_from_question,
    rephrase_question,
    perform_linking,
    query_selection_execution
)
from .State import State
from langchain_core.messages import HumanMessage, AIMessage


class TestModules(unittest.TestCase):


    def setUp(self):
        # Initialize State with required parameters
        self.state = State(
            knowledge_graph="dbpedia",  # Using dbpedia as it's a valid knowledge graph
            n_limit_VQuery=3,
            n_max_Vs=3,
            n_limit_EQuery=3,
            n_max_Es=3,
            n_max_answers=3,
            filtration_enabled=True
        )

    def test_classify_question_self_contained(self):
        # Test case 1: Self-contained question
        question = "Who is the author of the book 'The Great Gatsby'?"
        result = classify_question(question, self.state).lower().strip()
        print(f"\n{'='*150}")
        print(f"Test case 1 - Question: {question}")
        print(f"Returned value: {result}")
        print(f"Expected: 'self-contained'")
        print(f"Status: {'PASSED' if result == 'self-contained' else 'FAILED'}")
        print(f"{'='*150}")
        self.assertEqual(result, "self-contained")

    def test_classify_question_non_self_contained(self):
        # Test case 2: Non-self-contained question
        question = "What did he say about it?"
        result = classify_question(question, self.state).lower().strip()
        print(f"\n{'='*150}")
        print(f"Test case 2 - Question: {question}")
        print(f"Returned value: {result}")
        print(f"Expected: 'non-self-contained'")
        print(f"Status: {'PASSED' if result == 'non-self-contained' else 'FAILED'}")
        print(f"{'='*150}")
        self.assertEqual(result, "non-self-contained")

    def test_rephrase_question(self):
        session_id = "12345"
        # First add some chat history
        chat_history = self.state.get_by_session_id(session_id)
        chat_history.add_messages([
            HumanMessage(content="Who wrote The Great Gatsby?"),
            AIMessage(content="F. Scott Fitzgerald wrote The Great Gatsby."),
            HumanMessage(content="What did he say about it?"),
        ])
        
        # Now test rephrasing
        question = "What did he say about it?"
        result = rephrase_question(session_id, question, self.state)
        print(f"\n{'='*150}")
        print(f"Test rephrase_question")
        print(f"Chat history:")
        for msg in chat_history.messages:
            print(f"- {msg.type}: {msg.content}")
        print(f"Original question: {question}")
        print(f"Rephrased question: {result}")
        print(f"{'='*150}")

    def test_get_qir_from_question(self):
        
        session_id = "12345"
        # First add some chat history
        chat_history = self.state.get_by_session_id(session_id)
        chat_history.add_messages([
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris is the capital of France."),
            HumanMessage(content="What is its population?"),
        ])
        
        # Test with a follow-up question
        question = "What is Paris's population?"
        question_id = "1"
        result = get_qir_from_question(question, question_id, self.state)
        
        print(f"\n{'='*150}")
        print(f"Test get_qir_from_question")
        print(f"Chat history:")
        for msg in chat_history.messages:
            print(f"- {msg.type}: {msg.content}")
        # print(f"Question: {result}") # Not needed. Just the Query Graph is needed.
        query_graph = self.state.get_query_graph()
        print(f"[GRAPH NODES WITH URIs:] {query_graph.nodes(data=True)}")
        print(f"[GRAPH EDGES WITH URIs:] {query_graph.edges(data=True)}")
        print(f"Query graph: {query_graph}")
        print(f"{'='*150}")
        
    def test_perform_linking(self):
        # First get QIR to set up the query graph
        session_id = "12345"
        # First add some chat history
        chat_history = self.state.get_by_session_id(session_id)
        chat_history.add_messages([
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris is the capital of France."),
            HumanMessage(content="What is its population?"),
        ])
        
        # Test with a follow-up question
        question = "What is Paris's population?"
        question_id = "1"
        result = get_qir_from_question(question, question_id, self.state)
        
        
        # Then perform linking
        perform_linking(self.state)
        print(f"\n{'='*150}")
        print(f"Test perform_linking")
        query_graph = self.state.get_query_graph()
        print(f"[GRAPH NODES WITH URIs:] {query_graph.nodes(data=True)}")
        print(f"[GRAPH EDGES WITH URIs:] {query_graph.edges(data=True)}")
        print(f"{'='*150}")
        

    # Generate Queries, Select Queries, Execute Queries
    def test_query_selection_execution(self):
        # First get QIR and perform linking
        question = "What is the capital of France?"
        question_id = "test_3"
        # Fill the Query Graph with the QIR and URIs
        get_qir_from_question(question, question_id, self.state)
        perform_linking(self.state)
        
        # Then execute query selection
        query_selection_execution(self.state)
        print(f"\n{'='*150}")
        print(f"Test query_selection_execution")
        print(f"Number of possible answers: {len(self.state.get_answers())}")
        print(f"{'='*150}")

if __name__ == '__main__':
    unittest.main(verbosity=2)  # Added verbosity=2 for more detailed output 