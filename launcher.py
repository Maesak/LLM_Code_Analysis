import uvicorn
  
if __name__ == "__main__":
    print("Starting CodebaseAnalyzer API server...")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run("codebase_analyzer:app", host="0.0.0.0", port=8000, reload=True)
