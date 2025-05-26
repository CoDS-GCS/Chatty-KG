
# Setting Up WSL, Ollama, and Python for Local LLMs on Windows

## 1. Install WSL (Windows Subsystem for Linux)

To run Linux on Windows, you need to install WSL. Use the following command to set it up:

```
wsl --install
```

### 1.1 Run WSL with Ubuntu

Once WSL is installed, you can run it by specifying the Linux distribution (in this case, Ubuntu). Execute this command:

```
wsl.exe -d Ubuntu
```

- **Username**: `YOUR_USERNAME`
- **Password**: `YOUR_PASSWORD`

---

## 2. Install Ollama for Local LLMs

Ollama is a tool that allows you to run local large language models (LLMs) on your machine. To install Ollama, use the following command:

```
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 3. Running a Model in Ollama

Once Ollama is installed, you can run a model using the following commands. For example, let's run **DeepSeek-LLM:7b**:

### 3.1 Start the Model

```
ollama run deepseek-llm:7b
```

This will start the **DeepSeek-LLM** model locally.

### 3.2 Test the Model with a Prompt

To interact with the model, use a `curl` request. Here’s an example to summarize the plot of **Romeo and Juliet**:

```
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-llm:7b",
  "prompt": "Explain in one paragraph how dfs algorithm work with binary tree."
}'
```

This will generate a response from the model based on your input.

---

## 4. Python Setup

### 4.1 Create a Python Environment for Your Project

It's important to create a **virtual environment** to isolate your project dependencies. You can do this by running the following commands:

```
sudo apt update
sudo apt install python3 python3-pip -y
python3 --version
sudo apt install python3.12-venv
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
# Create a folder in the Linux file system
mkdir -p ~/agents
cd ~/agents

# Create the virtual environment inside Linux (this avoids the permission issue)
sudo python3 -m venv agentEnv
sudo chown -R $USER:$USER ~/Chatty-KG/agentEnv

# Activate the environment
source agentEnv/bin/activate

# Now install your requirements
pip install -r requirements.txt

```

### 4.2 Install Dependencies from Requirements File

If your project has a `requirements.txt` file for dependencies, you can install them using pip:

```
pip install -r requirements.txt
```

This will install all the necessary packages and libraries listed in the requirements file.

---

## Conclusion

By following these steps, you will have WSL, Ollama, and a Python environment set up on your Windows machine, ready to run **local LLMs** like DeepSeek-LLM. You can further expand this setup by adding more models or customizing it based on your specific project needs.

