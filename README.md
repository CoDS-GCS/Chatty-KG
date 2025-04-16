 ChattyKG: An LLM-Powered Dialogue Platform for KGs 
 ---
 - - - - -

Abstract
-------
Recent conversational AI systems have shown proficiency in general question answering, but face challenges in domain-specific answer retrieval, particularly when data is private. To address this,we introduce Chatty-KG, a novel LLM-based dialogue platform that enables real-time conversational question answering over arbitrary knowledge graphs (KGs) without requiring domain-specific training or pre-processing. Chatty-KG leverages retrieval augmented generation (RAG) and prompting techniques, including zero-shot learning and chain-of-thought reasoning, to classify and disambiguate questions, extract key entities and relations, and efficiently retrieve information via SPARQL endpoints to generate precise answers. Our comprehensive evaluation, using four real-world knowledge graphs and various LLMs, demonstrates that Chatty-KG significantly outperforms state-of-the-art systems in both answer correctness and response time. Notably, we show that carefully designed prompts enable open-source LLMs to achieve competitive performance with commercial LLMs in this task.
![GitHub Logo](chatty_kg.png)


Running ChattyKG
-------------
- - - - 

### Installation
 
* Clone the repo
* Create `venv` Conda environment (Python >= 3.10) and install pip requirements.
```
conda create --name venv
conda activate venv
pip install -r requirements.txt
```

### Download the required Embedding:
- Download wiki-news-300d-1M.zip from this [link](https://fasttext.cc/docs/en/english-vectors.html)
- Extract the downloaded script
- Create Data directory
- Move the extracted file to the data directory

### Running ChattyKG

ChattyKG uses a similarity module for relation linking, a pre-requisite step is to run the module server using the following command
 ```
 conda activate venv
 python word_embedding/server.py 127.0.0.1 9600
 ```

ChattyKG takes as an input a JSON file containing all questions in the following format:
```
{
    "question": 
    [
        {
            "string": question text
            "language": "en"
        }
   ],
   "id": question id,
   "answers": []
}
```
* To run ChattyKG in this mode, you need a script that opens the questions' file then calls ChattyKG module to answer the questions.
* To call the ChattyKG module you should use the following code:

```python
from chattykg import ChattyKG

my_chattykg = ChattyKG()
answers = my_chattykg.ask(question_text=question_text, question_id=question['id'], knowledge_graph=knowledge_graph)
```

