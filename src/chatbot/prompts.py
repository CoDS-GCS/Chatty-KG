from langchain.prompts import PromptTemplate

CLASSIFY_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Classify a given question as either 'independent' or 'dependent.' In this context, 'independent' questions are those that can be understood and answered without needing additional context or information, while 'dependent' questions require additional context or information to be answered. It's crucial for the model to identify ambiguity and resolve unknowns to determine the classification correctly.

        Examples:
        Question: 'What is the capital of France?'
        Classification: independent

        Question: 'Who is she talking about?'
        Classification: dependent

        Question: '{question}'
        Classification:""",
)

CLASSIFY_QUESTION_PROMPT_2 = PromptTemplate(
    input_variables=["question"],
    template="""Classify a given question as either 'self-contained' or 'non-self-contained' while considering the context of resolving pronouns and references. 
    In this task, 'self-contained' questions are those that can be understood and answered without needing additional context or information, even when they refer to entities or concepts mentioned in prior conversation. 'Non-self-contained' questions are those that, due to the presence of pronouns or references to prior conversation, require additional context or information to be answered correctly.

    Question: '{question}'
    Classification:""",
)


CONTEXT_CLASSIFY_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the chat_history and user question below, classify question as either INDEPENDENT or DEPENDENT.

        Do not respond with more than one word.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>

        Classification:""",
)

CONTEXT_CLASSIFY_QUESTION_PROMPT_2 = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Classify a given question as either 'self-contained' or 'non-self-contained' while considering the context of resolving pronouns and references. 
        In this task, 'self-contained' questions are those that can be understood and answered without needing additional context or information, even when they refer to entities or concepts mentioned in prior conversation. 'Non-self-contained' questions are those that, due to the presence of pronouns or references to prior conversation, require additional context or information to be answered correctly.

        Examples:
        Question: 'Name the author of the paper of title "MIMO Systems with Intentional Timing Offset"'
        Classification: self-contained

        Question: 'What was published by him in 1995?'
        Classification: non-self-contained

        Do not respond with more than one word.

        <chat_history>
        {chat_history}
        </chat_history>
        Question: '{question}'
        Classification:""",
)

CONDENSE_QUESTION_PROMPT_CUSTOM = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Preserve the original question in the answer sentiment during rephrasing.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""",
)