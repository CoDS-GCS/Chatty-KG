from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = ""

def get_openAI_llm():
    # Choose from [gpt-3.5-turbo, 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
    model_name = "gpt-4o"
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm

def detect_question_type(question):
    template = """ Identify if the given questions is a Yes/No question or a normal question. Return only of these values [Yes/No, Normal]
    
    Question: {question}
    Output: 
    """
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    # final_prompt = prompt.format(question=question)
    # print(final_prompt)
    llm = get_openAI_llm()
    chain = prompt | llm
    output = chain.invoke({"question": question})
    # print(output.content)
    output = output.content
    return output

def extract_triples_from_boolean(question):
    template = """ 
    Identify the triples within the provided question. Follow the following guidelines:
        1. Each triple should follow this format: (<entity>, <relation>, <entity>)
        2. Each fact should be included in only 1 triple.
        3. Return a list of triples in this format: [(), ()]
        4. Entity is either extracted from the question or is a variable
        5. variables must be returned as ?var followed by a unique id e.g. ?var1, ?var2, ..., etc
        6. relations should be extracted from the question if possible


        Question: {question}
        Output:  """
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    final_prompt = prompt.format(question=question)
    print(final_prompt)
    llm = get_openAI_llm()
    chain = prompt | llm
    output = chain.invoke({"question": question})
    print(output.content)
    output = output.content
    return output

def extract_triples_from_question(question):
    template_v1 = """
        Identify the triples within the provided question. Follow the following guidelines:
        1. Each triple should follow this format: (<entity>, <relation>, <entity>)
        2. Each fact should be included in only 1 triple.
        3. Return a list of triples in this format: [(), ()]
        4. Entity is either extracted from the question or is a variable
        5. variables must be returned as ?var followed by a unique id e.g. ?var1, ?var2, ..., etc
        6. Each question must contain at least one triple and one variable
        7. ?var1 represents the answer to the question
        8. relations should be extracted from the question if possible

        Question: {question}
        Output:  
        """
    template = """
    Identify the triples within the provided question. Follow these guidelines:

    1. Prioritize the answer to the question in the first triple.
    2. Each triple should follow this format: (<subject>, <predicate>, <object>)
    3. Each fact should be included in only 1 triple.
    4. Return a list of triples in this format: [(), ()]
    5. Entities can be extracted from the question or represented as variables.
    6. Variables must be returned as ?var followed by a unique id (e.g., ?var1, ?var2, ...).
    7. Each question must contain at least one triple and one variable.
    8. Predicates should be extracted from the question if possible.
    
    Examples:
    - Who wrote the book Pride and Prejudice?: (?author, wrote, Pride and Prejudice)
    - When was the Eiffel Tower built?:  (Eiffel Tower, built in, ?year)

    Question: {question}
    Output:
    """
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    final_prompt = prompt.format(question=question)
    print(final_prompt)
    llm = get_openAI_llm()
    chain = prompt | llm
    output = chain.invoke({"question": question})
    print(output.content)
    output = output.content
    return output

def remove_unneeded_chars(text):
    text = text.strip().strip("\'").strip('\"')
    if text.startswith("?"):
        text = text.replace(" ", "_")
    return text

def post_process(triples):
    triples = triples.replace('```', '')
    triples = triples.replace('\r', '')
    triples = triples.replace('\n', '')
    triple_list = list()
    while triples:
        try:
            start = triples.index('(')
            end = triples.index(')')
            one_triple = triples[start + 1:end]
            if one_triple.count(',') > 2:
                last_index = one_triple.rindex(',')
                obj = one_triple[last_index+1 :]
                one_triple = one_triple[:last_index]
                last_index = one_triple.rindex(',')
                pred = one_triple[last_index + 1:]
                subject = one_triple[:last_index].replace(',', '')
                triple_list.append(
                    {"subject": remove_unneeded_chars(subject), "predicate": remove_unneeded_chars(pred),
                     "object": remove_unneeded_chars(obj)})
            else:
                triple = one_triple.split(",")
                triple_list.append({"subject": remove_unneeded_chars(triple[0]), "predicate": remove_unneeded_chars(triple[1]), "object": remove_unneeded_chars(triple[2])})
            triples = triples[end + 2:]
        except:
            if ',' in triples and len(triples.split(',')) == 3:
                start = triples.index('[')
                end = triples.index(']')
                triple = triples[start + 1:end].split(",")
                triple_list.append(
                    {"subject": remove_unneeded_chars(triple[0]), "predicate": remove_unneeded_chars(triple[1]),
                     "object": remove_unneeded_chars(triple[2])})
            else:
                print("Error in post_process ", triples)
            break
    return triple_list

def extract_triples(question):
    question_type = detect_question_type(question)
    if question_type.lower() == "normal":
        output = extract_triples_from_question(question)
    elif question_type.lower() == "yes/no":
        output = extract_triples_from_boolean(question)
    else:
        raise Exception(f"Error: unidentified question type: {question_type}")
    triple_list = post_process(output)
    return triple_list

if __name__ == '__main__':
    questions = ['Give some publications of Gunter Saake from the book Grundlagen von Datenbanken?',
                 # 'Ordinal-Measure Based Shape Correspondence was published in which journal volume?',
                 # 'How many researchers worked on DBPal: Weak Supervision for Learning a Natural Language Interface to Databases?',
                 # 'Give researchers who worked on Creating the electronic new Oxford english dictionary?',
                 'How many researchers created Handbook of Semantic Web Technologies?',
                 'Are Taiko some kind of Japanese musical instrument?',
                 # 'Return the journal volume that published Machine Learning Methods in the Computational Biology of Cancer.',
                 # 'On the Accuracy of Fiber Tractography was accepted as a thesis in which school?',
                 # 'Name the researcher that published Distributed Symbolic Representation of Visual Shape?',
                 # 'Active perception to deep learning was published in which journal volume?',
                 # 'Name the institute that accepted Computational Approach to Binding Theory as a thesis?'
                 ]
    for question in questions:
        extract_triples(question)

    # x = 1
    # 1) Prompt to differentiate between boolean questions and non-boolean questions
    # 2) depending on this value either call a boolean prompt or a question prompt.