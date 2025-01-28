# Browser Agent

This project utilizes AI models to interact with the web through your browser.

## Features

- Integrates AI models for web browsing tasks.
- Provides a user-friendly interface for seamless interactions.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Python 3.11 or higher](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/myronlou/browser-agent.git
   cd browser-agent

   
2. **Set Up a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. **Install Dependencies**

Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**

Duplicate the .env.example file and rename the copy to .env. (currently only tested locally run model with ollama)

Open the .env file and set the necessary environment variables. For example:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL_NAME=deepseek-r1:7b
```
- [How to install ollama](https://github.com/ollama/ollama)

##Usage
To start the application, run:

```bash
python webui.py
```



