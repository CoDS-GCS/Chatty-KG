from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from chabot.prompts import CLASSIFY_QUESTION_PROMPT_2, CONTEXT_CLASSIFY_QUESTION_PROMPT, CONDENSE_QUESTION_PROMPT_CUSTOM
from kgqan.kgqan import KGQAn

max_Vs = 1
max_Es = 21
max_answers = 41
limit_VQuery = 600
limit_EQuery = 300

class KGChatbot:
    def __init__(self, kg_name, host):
        self.openai_model = "gpt-3.5-turbo"
        self.chat_summary = ""
        self.kg_name = kg_name
        self.host = host
        self.setup_resources()

    def ask_question(self, question):
        q_type = self.classify_question(question)
        if "non-self-contained" in q_type.lower():
            self.update_chat_summary()
            question = self.rephrase_question(question)
            q_type = self.classify_question(question)

        answer = self.run_query(question)
        self.update_context(question, answer)
        return answer

    # def setup_kgqan(_self, kg_name):
    #     return KGQAn(_self.host, kg_name)

    def setup_llm(_self):
        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            # streaming=True,
        )
        return model

    def setup_resources(self):
        self.llm = self.setup_llm()
        self.kgqan_instance = KGQAn(n_max_answers=max_answers, n_max_Vs=max_Vs, n_max_Es=max_Es, n_limit_VQuery=limit_VQuery, n_limit_EQuery=limit_EQuery)
        self.conv_buff_memory = ConversationSummaryBufferMemory(
            llm=self.llm, memory_key="chat_history", max_token_limit=150
        )
        string_output_parser = StrOutputParser()

        # Cache the LLMChain objects
        self.classify_question_chain = CLASSIFY_QUESTION_PROMPT_2 | self.llm | string_output_parser
        # self.classify_question_chain = LLMChain(
        #     llm=self.llm,
        #     prompt=CLASSIFY_QUESTION_PROMPT_2,
        #     verbose=True,
        #     output_parser=string_output_parser,
        # )
        self.classify_context_question_chain = CONTEXT_CLASSIFY_QUESTION_PROMPT | self.llm | string_output_parser
        self.classify_context_question_chain = LLMChain(
            llm=self.llm,
            prompt=CONTEXT_CLASSIFY_QUESTION_PROMPT,
            memory=self.conv_buff_memory,
            verbose=True,
            output_parser=string_output_parser,
        )

        self.rephrase_chain = LLMChain(
            llm=self.llm,
            prompt=CONDENSE_QUESTION_PROMPT_CUSTOM,
            memory=self.conv_buff_memory,
            verbose=True,
            output_parser=string_output_parser,
        )

    def get_active_kg(self):
        return self.kgqan_instance

    def get_llm(self):
        return self.llm

    def classify_question(self, question):
        q_type = self.classify_question_chain.invoke(
            {"chat_history": self.chat_summary, "question": question}
        )
        return q_type

    def update_chat_summary(self):
        self.chat_summary = self.conv_buff_memory.predict_new_summary(
            self.conv_buff_memory.chat_memory.messages, ""
        )

    def rephrase_question(self, question):
        question = self.rephrase_chain.run(question=question)
        return question

    def run_query(self, question):
        answers, _, _, understanding_time, linking_time, execution_time, query_selection_time, num_queries_executed \
            = self.kgqan_instance.ask(question_text=question,
                          question_id=0, knowledge_graph=self.kg_name)
        all_bindings = list()
        for answer in answers:
            if answer['results'] and answer['results']['bindings']:
                all_bindings.extend(answer['results']['bindings'])
        return all_bindings

    def update_context(self, question, answer):
        self.conv_buff_memory.save_context(
            {"input": f"#Question: {question}"}, {"output": f"#Answer: {answer}"}
        )