
# Running WSL, Ollama, and Python Virtual Environment (Repeat Each Time)

## 1. Start Ubuntu via WSL

Open a terminal and run:

```bash
wsl.exe -d Ubuntu
```

- **Username**: `aorogat`  
- **Password**: `ahamo@2020936`

---

## 2. Activate Python Virtual Environment

Navigate to your project folder and activate the virtual environment:

```bash
cd ~/CHATTY-KG
source agentEnv/bin/activate
source ~/Chatty-KG/agentEnv/bin/activate # for ne folder structure
```
```bash
git
token: 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
```


---

## 3. Run Ollama with a Model

### 3.1 Start the Model

```bash
ollama run deepseek-llm:7b
```

This keeps the model ready to accept requests.

### 3.2 Test the Model with a Prompt

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-llm:7b",
  "prompt": "Explain in one paragraph how DFS algorithm works with a binary tree."
}'
```

---

## 4. (Optional) Install Python Requirements

If you've updated `requirements.txt`, reinstall packages:

```bash
pip install -r requirements.txt
```


## Run the BERT server
```bash
cd word_embedding
python server.py
```


## 5. Run the code
```bash
cd src
python -m multi_agent.main
python -m multi_agent.modules.test_modules
```