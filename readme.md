# CodebaseAnalyzer

## What is CodebaseAnalyzer?

CodebaseAnalyzer is an AI-powered tool that analyzes GitHub repositories to understand their structure, functionality, and potential issues. It helps developers quickly grasp the purpose and key components of a codebase.

## Features

- **Download and analyze any GitHub repository**
- **Identify key parts of the code**
- **Summarize the project's purpose**# CodebaseAnalyzer

## What is this?

CodebaseAnalyzer is a tool that looks at GitHub code projects and tells you what they do and how they work. It uses AI to read and understand code.

## What can it do?

- Download and study any GitHub project
- Find the most important parts of the code
- Tell you what the project is trying to do
- Find possible problems in the code
- Create a report you can read later

## What you need

- Python 3.8 or newer
- Git
- Ollama with the llama3.2 model
- Some Python packages

## How to install

1. Get the code:
   ```
   git clone https://github.com/yourusername/CodebaseAnalyzer.git
   cd CodebaseAnalyzer
   ```

2. Install needed packages:
   ```
   pip install langchain chromadb git-python huggingface-hub sentence-transformers ollama
   ```

3. Make sure you have Ollama and the llama3.2 model:
   ```
   # Get Ollama from https://ollama.ai/
   ollama pull llama3.2
   ```

## How to use it

```python
from codebase_analyzer import CodebaseAnalyzer

# Tell it which GitHub project to analyze
analyzer = CodebaseAnalyzer(
    repo_url="https://github.com/username/repository",
    repo_name="LocalName",
    db_directory="./chroma_db"
)

# Run the analysis
report = analyzer.run_analysis()

# It saves the report as a JSON file
print("Analysis done!")
```

## Example

```python
# Look at the SakilaProject
analyzer = CodebaseAnalyzer(
    repo_url="https://github.com/janjakovacevic/SakilaProject",
    repo_name="SakilaProject",
    db_directory="./chroma_sakila_db"
)

report = analyzer.run_analysis()
```

## How it works

1. **Downloads the code**: Gets the GitHub project to your computer
2. **Reads the files**: Looks at all the code files
3. **Uses AI**: Analyzes the code using AI to understand it
4. **Creates a report**: Makes a JSON file with what it found

## Limits

- Works best with smaller projects (under 100MB)
- Works better with Java projects, but can handle others
- Depends on how good the AI model is
- Might miss things in very complex code

## Need help?

Feel free to ask questions or suggest improvements!
- **Detect potential issues in the code**
- **Generate a detailed report for later reference**

## Requirements

To use CodebaseAnalyzer, ensure you have the following installed:

- Python **3.8** or newer
- Git
- **Ollama** with the **llama3.2** model
- The following Python packages:

  ```sh
  pip install langchain chromadb git-python huggingface-hub sentence-transformers ollama
