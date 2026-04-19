import json
import google.generativeai as genai

class PromptGenerator:
    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash", 
            generation_config={"response_mime_type": "application/json"}
        )

    def generate_variations(self, base_task: str, num_variations: int = 5) -> list[str]:
        """Generates prompt variations based on academic prompt engineering strategies."""
        print(f"Generating {num_variations} prompt variations using Gemini...")
        
        system_instruction = f"""
        You are an expert Prompt Engineer. The user will provide a base task.
        Your goal is to generate exactly {num_variations} unique prompt instructions.
        
        CRITICAL: To test our evaluation system against academic benchmarks, you MUST generate prompts that follow these 5 specific styles:
        1. Chain-of-Thought (CoT): explicitly instruct the model to think step-by-step before answering.
        2. Persona-based: give the model an expert persona (e.g., world-class mathematician).
        3. Zero-shot Standard: direct, professional instructions without extra reasoning steps.
        4. Concise: extremely brief and to the point.
        5. Baseline/Poor: intentionally vague and unhelpful (e.g., "do the math thing") to serve as a negative control.
        
        Return the response as a JSON object containing a list of strings under the key "prompts".
        Example Output format:
        {{"prompts": ["prompt 1", "prompt 2", ...]}}
        """
        
        try:
            response = self.model.generate_content(system_instruction + "\n\nBase Task: " + base_task)
            result = json.loads(response.text)
            return result.get("prompts", [])
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return []
