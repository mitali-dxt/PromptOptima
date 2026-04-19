import os
import requests
from dotenv import load_dotenv

from generator import PromptGenerator
from evaluator import OllamaEvaluator
from optimizer import PromptOptimizer
from dataset_loader import load_benchmark

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")

    # Verify Ollama connection
    ollama_url = "http://localhost:11434/api/generate"
    try:
        requests.get("http://localhost:11434/")
    except requests.exceptions.ConnectionError:
        print(f"WARNING: Could not connect to Ollama at {ollama_url}. Please ensure Ollama is running.")
        
    # Load hardcoded benchmark (Options: "gsm8k", "logic")
    dataset = load_benchmark(name="gsm8k", num_samples=5)

    # Initialize modules
    generator = PromptGenerator(api_key=api_key)
    evaluator = OllamaEvaluator(model_name="phi3", url=ollama_url)
    optimizer = PromptOptimizer(generator, evaluator, dataset)

    # Run optimizer
    base_task = "Solve the given grade-school math word problem step by step and provide the final numerical answer."
    optimizer.run(base_task=base_task, num_variations=5)

if __name__ == "__main__":
    main()
