##==============================Question Understanding==============================##

# Initial template for QU
# Baseline
question_understanding_template_v1 = """
### Instruction:
Extract triples from the given question. Follow these rules:
1. **Output Format:** Return the result as a JSON object using the exact format: {{ "triples": [] }}
2. **Triple Format:** Each triple must follow this exact structure: [<subject>, <predicate>, <object>].
3. **Single Fact per Triple:** Each triple must represent a unique fact. Do not include duplicate triples or redundant facts.
4. **Entities Extraction:** Entities must be extracted from the question in the following ways:
    - Explicit Entities: Extract entities directly mentioned in the question whenever possible.
    - Variables for Missing Entities: Use variables (starting with ?, e.g., ?var) if either the subject or object is not explicitly mentioned in the question.
    - Answer Representation: At least one variable representing the answer (e.g., ?answer) must be included, except for yes/no questions
5. **Ambiguity Handling::** If the question is ambiguous or incomplete, produce at least one meaningful triple with a variable as the subject or object.
6. **Predicate Extraction:** Extract predicates directly from the question text. Avoid inventing predicates not explicitly implied.
7. **Mandatory Triple Inclusion:** Ensure there is at least one triple in the output. The triples array must never be empty.

#### Examples:
1. Question: Who wrote Pride and Prejudice?
Triples:  [ ["?author", "wrote", "Pride and Prejudice"] ]

2. Question: When was the Eiffel Tower built?
Triples:  [ ["Eiffel Tower", "built in", "?year"] ]

### Input:
Question: {question}

### Response:```json
"""

# Tried to change the wording of v1, not a significant improvement detected
# 41.9
question_understanding_template_v2 = """
    Extract triples from the given question. Follow these rules:

1. **Output Format:** Return the result as a JSON object using the exact format: {{ "triples": [] }}.
2. **Triple Format:** Each triple must follow this exact structure: [<subject>, <predicate>, <object>].
3. **Single Fact per Triple:** Each triple must represent a unique fact. Do not include duplicate triples or redundant facts.
4. **Entities Extraction:**
   - Extract entities explicitly mentioned in the question whenever possible.
   - Use variables (starting with ?, e.g., ?var) if either the subject or object is missing from the question.
   - Ensure at least one variable representing the answer (e.g., ?answer) is included, except for yes/no questions.
5. **Predicate Extraction:**
   - Extract predicates directly from the question text.
   - DO NOT extract any helping verbs.
   - If a question asks about count, the predicate must include what is being counted.
6. **Mandatory Triple Inclusion:** Always return at least one triple in the output.
7. **Ambiguity Handling:** If the question is ambiguous or incomplete, produce at least one meaningful triple with a variable.

### **Examples:**
1. **Question:** Who wrote Pride and Prejudice?  
   **Triples:** `[ ["?author", "wrote", "Pride and Prejudice"] ]`

2. **Question:** When was the Eiffel Tower built?  
   **Triples:** `[ ["Eiffel Tower", "built in", "?year"] ]`

### **Input:**
Question: {question}

### **Response:**
```json
    """

# Latest working version using chain of thought prompting
question_understanding_template_v3_cot = """
Extract triples from the given question by following these steps carefully. **Let's think step by step.**

### **Step 1: Extract Key Entities**
- Identify explicit entities mentioned in the question.
- If an entity is missing but implied, introduce a variable (e.g., ?var).

### **Step 2: Identify the Predicates**
- **Prioritize nouns that represent the core relationship being questioned.** These nouns often describe the key property or attribute being asked about (e.g., "capital," "author," "moons").
- **Explicitly exclude generic verbs like "has," "is," "are," "was," "were," "do," "does," "did" when a more specific noun or verb can express the relationship.**
- If no such core noun exists, then consider other verbs, excluding the helper verbs listed above.
- Extract the most relevant predicate from the question.
- The predicate should be specific and relates two entities.

### **Step 3: Construct the Triples**
- Formulate the triple using the extracted subject, predicate, and object.
- If the object is unknown, use a variable (e.g., ?answer).
- Ensure at least one meaningful triple is produced.

### **Step 4: Output the Result**
- Return the final output as a JSON object in this exact format: `{{ "triples": [] }}`.

**Let's apply this step-by-step approach to extract the correct triple.**

### **Examples:**
#### **Example 1**
**Question:** Who wrote Pride and Prejudice?  
**Step 1 - Extract Entities:** ["Pride and Prejudice", "?author"]  
**Step 2 - Identify Predicates:** "wrote"  
**Step 3 - Construct Triples:** `[ ["?author", "write", "Pride and Prejudice"] ]`  

#### **Example 2**
**Question:** What is the capital of France?
**Step 1 - Extract Entities:** ["France", "?capital"]
**Step 2 - Identify Predicates:** "capital"
**Step 3 - Construct Triples:** `[ ["France", "capital", "?capital"] ]`

### **Input:**
Question: {question}

### **Response:**
```json
"""

##==============================Vertex Linking==============================##
vertex_linking_template_v1 = """
    ### Instruction:
Given an entity and a list of labels, choose the label that is the most semantically similar to the entity.
1. If there is an exact match in the list, **choose it**.
2. If there is no exact match, select the label that is closest in meaning.
Only Return the label in a **JSON object** in the following format: {{"value": label }}. Do not add any explanations.

### Inputs:
Entity: {entity}
List of labels: [{vertex_label_list}]

### Response:```json
        """

vertex_linking_template_v2 = """### Instruction:
Given an entity and a list of labels, return the label that is the most semantically similar.
1. If an exact match (case-insensitive) exists in the list, return it immediately.
2. If no exact match is found, select the label with the closest meaning.
**Important Notes**:
- Exact match means the entity must match a label character-for-character, ignoring case.
- Do not modify, append, or alter the exact-match label.
Only Return the label in a **JSON object** in the following format: {{"value": label }}. Do not add any explanations.

### Inputs:
Entity: {entity}
List of labels: [{vertex_label_list}]

### Response:```json
        """

vertex_linking_template_v3 = """#### Instruction:
Given an entity and a list of labels, return the most similar label.
1. If an exact match (case-insensitive) exists in the list, return that label immediately and exclusively, ignoring any other labels.
2. If no exact match is found, select the most semantically similar label based on meaning, context, and common associations.

**Important Notes**:
- An **exact match** means the entity must match a label character-for-character, ignoring case.
- Do not modify, append, 3or alter the exact-match label.
- If an exact match exists, it must be returned even if a semantically similar label is also present.
- Prioritize labels that closely match the entity’s meaning, not just syntactic similarity.

Only Return the label in a **JSON object** in the following format: {{"value": label}}. Do not add any explanations.
### Inputs:
Entity: {entity}
List of labels: [{vertex_label_list}]

### Response:```json
        """

##==============================Filtration==============================##
# template without None
query_selection_template_v1 = ("Task: Select all keywords from the provided list below that are relevant to answer the question: "
                "{question}\nList of Keywords:\n{predicate_list}\nInstructions: Please choose only the keywords from "
                "the list above that you believe are relevant to answering the question accurately. Ensure that the "
                "selected keywords belong to the provided list. If none of the keywords is relevant, choose the most similar one. Do not return any explanations. \nOutput: ")

# 45 on qald, # best reults for solution 13(m)
# No Markdown
query_selection_template_v2 = ("Task: Select all keywords from the provided list below that are relevant to answer the question: "
                "{question}\nList of Keywords:\n{predicate_list}\nInstructions: Please choose only the keywords from "
                "the list above that you believe are relevant to answering the question accurately. Ensure that the "
                "selected keywords belong to the provided list. If none of the keywords are applicable, return None. \nOutput: ")

query_selection_template_v3 = """### Instruction:
select all keywords from the provided list below that are relevant to answer the question.
Please choose only the keywords from the list that you believe are relevant to answering the question accurately. Ensure that the selected keywords belong to the provided list. If none of the keywords are applicable, return None.
Only Return the keywords in a JSON object in the following format: {{"keywords": [] }}.

### Inputs:
Question: {question}
List of Keywords: [{predicate_list}]

### Response:```json
    """

query_selection_template_v4 = """
### Instruction:
Select all possible keywords from the provided list below that are relevant to answering the question.
- If multiple variations of a keyword exist in the list (e.g., singular, plural, or synonymous forms), ensure that all are included. 
- Only choose keywords from the provided list - do not generate new ones.
- If none of the keywords are applicable, return None.
 Only Return the keywords in a JSON object in the following format: {{"keywords": [] }}.

### Inputs:
Question: {question}
List of Keywords: [{predicate_list}]

### Response:```json
"""
# Best version with gpt4o - 49
query_selection_template_v5 = """### Instruction:
Let’s think step by step about what the question is asking and what the answer would look like.
    - Evaluate each keyword's relevance to retrieving the correct answer. Focus on keywords that are directly related to answering the question.
    - Only include keywords that directly contribute to retrieving the correct answer.
    - Exclude keywords related to the item or context, and focus solely on those that provide the key information needed for the answer.

Only choose keywords from the provided list - do not generate new ones.
Return the keywords in a JSON object in the following format:
{{ "keywords": [] }}.

### Inputs:
Question: {question}
List of Keywords: [{predicate_list}]

### Response:```json
"""

# 1. Determine what type of answer the question is asking for (e.g., a person, a location, an event, a date, etc.).
# 2. Select all relevant keywords from the provided list that are related to the expected answer type.
#     - If multiple keywords could apply, include all of them.
#     - Do not exclude a keyword just because another one seems more specific.

query_selection_template_v6 = """### Instruction:
Let’s think step by step about what the question is asking and what the answer would look like.
1. Identify the core concept the question is asking for.
2. Evaluate each keyword in the provided list and determine if it is directly relevant to retrieving the answer.
3. **Select only the keywords from the list** that essential or finding the answer. 
   - **Do not generate or modify keywords.**  
   - **Do not create variations or rephrase words.**  
4. **Exclude keywords that describe entities, or general context.**  
   - Focus **only** on keywords that contribute to answering the question.  
   - Avoid selecting keywords that relate to the entity itself rather than the answer.  
5. Exclude any keyword that is unrelated to the answer or too general.

Return the keywords in a JSON object in the following format:
{{ "keywords": [] }}.

### Inputs:
Question: {question}
List of Keywords: [{predicate_list}]

### Response:```json
"""


