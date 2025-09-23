from langchain_core.prompts.prompt import PromptTemplate
from chattykg.llms.llm_creation import llm_type, get_llm

def translate_question(question):
    # translate_prompt_template = """Translate the following question to English:
    # Question: {question}
    # Translation:"""
    translate_prompt_template = """If the following question is in English, output it exactly as it is.  
If it is in another language, translate it into English.  

Question: {question}  
Output:
"""

    prompt = PromptTemplate(
            input_variables=["question"],
            template=translate_prompt_template,
        )
    llm = get_llm()
    chain = prompt | llm
    output = chain.invoke({"question": question})
    output = output.content
    print(f"Original question: {question}\nTranslated question: {output}\n")
    return output