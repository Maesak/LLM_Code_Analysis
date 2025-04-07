import os
import json
import git
import glob
from typing import Dict, List, Any, Optional
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import logging
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    def __init__(self, repo_url: str, repo_name: str, db_directory: str = "chroma_db"):
        """
        Initialize the CodebaseAnalyzer.

        Args:
            repo_url: URL of the GitHub repository to analyze
            repo_name: Name to use for the local repository clone
            db_directory: Directory to store the Chroma vector database
        """
        self.repo_url = repo_url
        self.repo_name = repo_name
        self.db_directory = db_directory
        self.local_path = f"./{repo_name}"
        self.llm = Ollama(model="llama3.2")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def clone_repository(self) -> None:
        """Clone the GitHub repository if it doesn't exist locally."""
        if not os.path.exists(self.local_path):
            logger.info(f"Cloning repository {self.repo_url} to {self.local_path}")
            git.Repo.clone_from(self.repo_url, self.local_path)
        else:
            logger.info(f"Repository already exists at {self.local_path}")

    def get_file_paths(self, extensions: List[str] = None) -> List[str]:
        """
        Get paths to all files in the repository with specified extensions.

        Args:
            extensions: List of file extensions to include (e.g., ['.java', '.sql'])
                        If None, include all files

        Returns:
            List of file paths
        """
        if extensions is None:
            # Default to common code file extensions
            extensions = [
                ".java",
                ".sql",
                ".xml",
                ".properties",
                ".md",
                ".txt",
                ".py",
                ".js",
                ".html",
                ".css",
            ]

        file_paths = []
        for ext in extensions:
            pattern = os.path.join(self.local_path, f"**/*{ext}")
            file_paths.extend(glob.glob(pattern, recursive=True))

        logger.info(f"Found {len(file_paths)} files with extensions {extensions}")
        return file_paths

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load files into LangChain documents.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of LangChain Document objects
        """
        documents = []

        for file_path in file_paths:
            try:
                # Skip binary files and very large files
                if os.path.getsize(file_path) > 1000000:  # Skip files larger than 1MB
                    logger.warning(f"Skipping large file: {file_path}")
                    continue

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Create metadata with file path and extension
                relative_path = os.path.relpath(file_path, self.local_path)
                _, extension = os.path.splitext(file_path)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": relative_path,
                        "extension": extension,
                        "file_name": os.path.basename(file_path),
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for processing.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        Create a Chroma vector store from the documents.

        Args:
            documents: List of documents to add to the vector store
        """
        logger.info(f"Creating vector store in {self.db_directory}")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_directory,
        )
        self.vectorstore.persist()
        logger.info("Vector store created and persisted")

    def load_vectorstore(self) -> None:
        """Load the vector store from disk if it exists."""
        if os.path.exists(self.db_directory):
            logger.info(f"Loading existing vector store from {self.db_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.db_directory, embedding_function=self.embeddings
            )
        else:
            logger.warning(f"Vector store not found at {self.db_directory}")

    def analyze_project_overview(self) -> Dict[str, Any]:
        """
        Generate a high-level overview of the project using the LLM.

        Returns:
            Dictionary containing project overview information
        """
        logger.info("Analyzing project overview")

        # Query for README files and documentation
        query = "README OR documentation OR project description"
        docs = self.vectorstore.similarity_search(query, k=5)

        # Concatenate relevant documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt for project overview
        prompt_template = """
        You are an expert code analyst. Based on the following code excerpts from a project, 
        provide a comprehensive overview of the project.
        
        CODE EXCERPTS:
        {context}
        
        Please analyze the code and provide the following information in JSON format:
        1. Project name
        2. Main purpose and functionality
        3. Technologies and frameworks used
        4. Overall architecture (if discernible)
        5. Key features
        
        Format your response as a valid JSON object with these fields. Do not include any explanations or text outside the JSON structure.
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.run(context=context)

        try:
            # Clean the response to ensure it's valid JSON
            json_str = response.strip()
            # Remove markdown code blocks if present
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "", 1)
            if json_str.endswith("```"):
                json_str = json_str.replace("```", "", 1)

            json_str = json_str.strip()
            overview = json.loads(json_str)
            logger.info("Successfully generated project overview")
            return overview
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {
                "project_name": "Unknown",
                "main_purpose_and_functionality": "Error parsing LLM response",
                "technologies_and_frameworks": [],
                "overall_architecture": "Error parsing LLM response",
                "key_features": []
            }

    def extract_key_methods(self) -> List[Dict[str, str]]:
        """
        Extract key methods and their descriptions from the codebase.

        Returns:
            List of dictionaries containing method information
        """
        logger.info("Extracting key methods")

        # Query for Java files (assuming it's a Java project)
        java_docs = self.vectorstore.similarity_search(
            "class public private void function method",   # similar 5 -> key words search
            k=20,
            filter={"extension": ".java"},
        )

        # Concatenate relevant documents
        context = "\n\n".join(
            [f"File: {doc.metadata['source']}\n{doc.page_content}" for doc in java_docs]
        )

        # Create prompt for method extraction
        prompt_template = """
        You are a code analysis expert. Analyze the following Java code from the SakilaProject and extract the key methods.
        
        CODE:
        {context}
        
        Extract the most important methods (up to 15 methods) and provide a JSON response with the following structure:
        [
            {{
                "class_name": "Name of the class",
                "method_name": "Name of the method",
                "signature": "Full method signature",
                "description": "A concise description of what the method does",
                "complexity": "An estimate of the method's complexity (Low, Medium, High)",
                "file_path": "Path to the file containing the method"
            }}
        ]
        
        Focus on public methods that appear to be central to the project's functionality.
        Ensure your response is only valid JSON with no additional text.
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.run(context=context)

        try:
            # Clean the response to ensure it's valid JSON
            json_str = response.strip()
            # Remove markdown code blocks if present
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "", 1)
            if json_str.endswith("```"):
                json_str = json_str.replace("```", "", 1)

            json_str = json_str.strip()
            methods = json.loads(json_str)
            logger.info(f"Successfully extracted {len(methods)} key methods")
            return methods
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def analyze_code_complexity(self) -> Dict[str, Any]:
        """
        Analyze the overall code complexity and structure.

        Returns:
            Dictionary containing complexity analysis
        """
        logger.info("Analyzing code complexity")

        # Get a variety of files for a broader analysis
        docs = self.vectorstore.similarity_search("code structure complexity", k=10)

        # Concatenate relevant documents
        context = "\n\n".join(
            [f"File: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
        )

        # Create prompt for complexity analysis
        prompt_template = """
        You are a code analysis expert. Analyze the following code from the SakilaProject and provide an assessment of its complexity and structure.
        
        CODE SAMPLES:
        {context}
        
        Provide a JSON response with the following structure:
        {{
            "overall_complexity": "An assessment of the overall codebase complexity (Low, Medium, High)",
            "code_quality": "An assessment of the overall code quality",
            "notable_patterns": ["List of design patterns or architectural patterns observed"],
            "potential_issues": ["List of potential issues or areas for improvement"],
            "dependencies": ["List of key dependencies or technologies used"]
        }}
        
        Ensure your response is only valid JSON with no additional text.
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = LLMChain(llm=self.llm, prompt=prompt)

        response = chain.run(context=context)

        try:
            # Clean the response to ensure it's valid JSON
            json_str = response.strip()
            # Remove markdown code blocks if present
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "", 1)
            if json_str.endswith("```"):
                json_str = json_str.replace("```", "", 1)

            json_str = json_str.strip()
            complexity_analysis = json.loads(json_str)
            logger.info("Successfully generated complexity analysis")
            return complexity_analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {
                "overall_complexity": "Unknown",
                "code_quality": "Unknown",
                "notable_patterns": [],
                "potential_issues": [],
                "dependencies": [],
            }
            
    def answer_question(self, question: str, k: int = 8) -> Dict[str, Any]:
        """
        Answer a specific question about the codebase using the LLM.
        
        Args:
            question: The question to answer about the codebase
            k: Number of relevant documents to retrieve
            
        Returns:
            Dictionary containing the answer and supporting code excerpts
        """
        logger.info(f"Answering question: {question}")
        
        # Check if vectorstore is loaded
        if not self.vectorstore:
            logger.error("Vector store not loaded. Cannot answer questions.")
            return {
                "question": question,
                "answer": "Error: Vector store not loaded. Please run analysis first.",
                "sources": []
            }
            
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Extract sources
        sources = [
            {
                "file_path": doc.metadata["source"],
                "file_name": doc.metadata["file_name"],
                "extension": doc.metadata["extension"]
            }
            for doc in docs
        ]
        
        # Concatenate relevant documents
        context = "\n\n".join(
            [f"File: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
        )
        
        # Create prompt for question answering
        prompt_template = """
        You are a code analysis expert. Based on the following code excerpts from a project,
        answer the question as accurately and comprehensively as possible.
        
        CODE EXCERPTS:
        {context}
        
        QUESTION:
        {question}
        
        Provide a detailed answer to the question, referencing specific parts of the code where relevant.
        If the question cannot be answered based on the provided code, explain why.
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = chain.run(context=context, question=question)
        
        return {
            "question": question,
            "answer": response.strip(),
            "sources": sources
        }

    def generate_final_report(self) -> Dict[str, Any]:
        """
        Generate the final comprehensive report in JSON format.

        Returns:
            Dictionary containing the full analysis report
        """
        logger.info("Generating final report")

        # Get project overview
        overview = self.analyze_project_overview()

        # Extract key methods
        methods = self.extract_key_methods()

        # Analyze code complexity
        complexity = self.analyze_code_complexity()

        # Combine all analyses into a single report
        report = {
            "project_overview": overview,
            "key_methods": methods,
            "complexity_analysis": complexity,
            "metadata": {
                "analysis_date": "2025-04-07",
                "analyzer": "CodebaseAnalyzer",
                "model": "llama3.2",
            },
        }

        return report

    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Returns:
            Dictionary containing the full analysis report
        """
        # Clone the repository
        self.clone_repository()

        # Check if vector store exists
        if os.path.exists(self.db_directory) and os.listdir(self.db_directory):
            # Load existing vector store
            self.load_vectorstore()
            logger.info("Using existing vector store")
        else:
            # Process files and create vector store
            file_paths = self.get_file_paths()
            documents = self.load_documents(file_paths)
            split_docs = self.split_documents(documents)
            self.create_vectorstore(split_docs)

        # Generate the final report
        report = self.generate_final_report()

        # Save the report to a JSON file
        with open("sakila_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Analysis complete. Report saved to sakila_analysis_report.json")

        return report


# FastAPI Models
class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 8

class AnalysisRequest(BaseModel):
    repo_url: str
    repo_name: str
    db_directory: Optional[str] = None

# FastAPI App
app = FastAPI(
    title="Codebase Analyzer API",
    description="API for analyzing GitHub repositories and answering questions about codebases",
    version="1.0.0"
)

# Global analyzer instance
analyzer_instance = None

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_repository(request: AnalysisRequest):
    """
    Analyze a GitHub repository and generate a comprehensive report.
    """
    global analyzer_instance
    
    try:
        # Create a new analyzer instance
        db_directory = request.db_directory or f"./chroma_{request.repo_name}_db"
        analyzer_instance = CodebaseAnalyzer(
            repo_url=request.repo_url,
            repo_name=request.repo_name,
            db_directory=db_directory
        )
        
        # Run the analysis
        report = analyzer_instance.run_analysis()
        
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the previously analyzed codebase.
    """
    global analyzer_instance
    
    if not analyzer_instance:
        raise HTTPException(
            status_code=400, 
            detail="No repository has been analyzed yet. Please call /analyze endpoint first."
        )
        
    try:
        # Answer the question
        answer = analyzer_instance.answer_question(
            question=request.question,
            k=request.k
        )
        
        return {
            "status": "success",
            "result": answer
        }
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.get("/status", response_model=Dict[str, Any])
async def get_status():
    """
    Get the current status of the analyzer.
    """
    global analyzer_instance
    
    if analyzer_instance:
        return {
            "status": "ready",
            "repo_name": analyzer_instance.repo_name,
            "repo_url": analyzer_instance.repo_url,
            "db_directory": analyzer_instance.db_directory
        }
    else:
        return {
            "status": "not_initialized",
            "message": "No repository has been analyzed yet."
        }


if __name__ == "__main__":
    import uvicorn
    
    # Initialize the analyzer with the SakilaProject repository
    analyzer = CodebaseAnalyzer(
        repo_url="https://github.com/janjakovacevic/SakilaProject",
        repo_name="SakilaProject",
        db_directory="./chroma_sakila_db",
    )

    # Run the analysis
    report = analyzer.run_analysis()

    # Example question
    question_result = analyzer.answer_question("What are the main database tables used in this project?")
    print("\n=== QUESTION ANSWERED ===")
    print(f"Q: {question_result['question']}")
    print(f"A: {question_result['answer']}")
    
    # Print summary of findings
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Project: {report['project_overview'].get('project_name', 'Unknown')}")
    print(f"Main purpose: {report['project_overview'].get('main_purpose_and_functionality', 'Unknown')}")
    print(f"Complexity: {report['complexity_analysis'].get('overall_complexity', 'Unknown')}")
    print(f"Key Methods Found: {len(report['key_methods'])}")
    print("Full report saved to sakila_analysis_report.json")
    
    # Start the FastAPI server
    print("\n=== STARTING API SERVER ===")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)