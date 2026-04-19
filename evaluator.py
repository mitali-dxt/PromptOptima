import requests
import re

class OllamaEvaluator:
    def __init__(self, model_name: str = "phi3", url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.url = url

    def run_model(self, prompt: str, max_tokens: int = 150) -> str:
        """Runs a query against the local Ollama model with length limits."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return ""

    def evaluate_with_judge(self, question: str, generated_answer: str, expected_answer: str) -> float:
        """Uses the Ollama model as a judge to score the generated answer."""
        judge_prompt = f"""
        You are an impartial judge evaluating a math answer.
        Question: {question}
        Expected correct numerical answer: {expected_answer}
        Generated answer: {generated_answer}
        
        Does the Generated answer arrive at the exact expected numerical answer?
        Respond ONLY with a single number: 1 if it is completely correct, 0.5 if it is partially correct, or 0 if it is incorrect.
        Do not add any explanation. Just the number.
        """
        
        judge_response = self.run_model(judge_prompt, max_tokens=10)
        
        try:
            match = re.search(r'\b(1(\.0+)?|0\.5|0(\.0+)?)\b', judge_response)
            if match:
                return float(match.group(0))
                
            match = re.search(r'\d+(\.\d+)?', judge_response)
            if match:
                score = float(match.group(0))
                return max(0.0, min(1.0, score))
                
            return 0.0
        except ValueError:
            return 0.0
