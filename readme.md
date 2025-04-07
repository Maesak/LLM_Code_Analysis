# Codebase Analyzer

This tool helps you understand GitHub projects by analyzing their code automatically.

## What Does It Do?

It takes a GitHub link, downloads the code, and uses AI to figure out:
- What the project is about
- What technologies it uses
- How the code is organized
- The main parts of the project
- How complex the code is

## How to Set It Up

1. Make sure you have:
   - Python installed on your computer
   - Git installed on your computer
   - Ollama - LLAMA3.2 (an AI tool) running on your computer

2. Install the needed packages:
   ```
   pip install gitpython langchain langchain_community tqdm
   ```

## How to Use It

Here's a simple example:

```python
from codebase_analyzer import CodebaseAnalyzer

# Point it to a GitHub repository
analyzer = CodebaseAnalyzer(
    repo_url="https://github.com/username/repository"
)

# Run the analysis
if analyzer.run_analysis():
    # Save the results as a JSON file
    analyzer.save_results("results.json")
    print("Done! Check results.json")
else:
    print("Something went wrong")
```

## What You Get

You'll get a JSON file with:
- An overview of the project
- Breakdown of the main components
- Analysis of how the code is structured

## Limits

- It might take a while for big projects
- It works best with well-organized code
- The quality depends on the AI model it uses

## Example

Try it on your favorite GitHub project:

```python
analyzer = CodebaseAnalyzer(repo_url="https://github.com/janjakovacevic/SakilaProject")
analyzer.run_analysis()
analyzer.save_results("analysis.json")
```