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

    def generate_variations(self, base_task: str, num_variations: int = 3) -> list[str]:
        """Generates prompt variations of purposefully DIFFERENT qualities."""
        print(f"Generating {num_variations} prompt variations using Gemini...")
        
        system_instruction = f"""
        You are an expert Prompt Engineer. The user will provide a base task.
        Your goal is to generate exactly {num_variations} unique prompt instructions.
        
        CRITICAL: To test our evaluation system, you MUST generate prompts of varying quality:
        - 1 extremely HIGH quality, detailed, step-by-step prompt.
        - 1 AVERAGE quality prompt with basic instructions.
        - 1 extremely POOR quality, vague, and confusing prompt (e.g. "do the math thing").
        
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
