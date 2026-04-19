from generator import PromptGenerator
from evaluator import OllamaEvaluator

class PromptOptimizer:
    def __init__(self, generator: PromptGenerator, evaluator: OllamaEvaluator, dataset: list[dict]):
        self.generator = generator
        self.evaluator = evaluator
        self.dataset = dataset

    def run(self, base_task: str, num_variations: int = 3):
        print("=" * 50)
        print("PromptOptima: Automated Prompt Engineering")
        print("=" * 50)
        print(f"\nBase Task: {base_task}\n")
        
        variations = self.generator.generate_variations(base_task, num_variations=num_variations)
        if not variations:
            print("Failed to generate variations.")
            return

        print("\nGenerated Prompt Variations (Mixed Qualities):")
        for i, p in enumerate(variations):
            print(f"[{i+1}] {p}")
            
        print("\n" + "-" * 50)
        print("Evaluating Prompts using Ollama...")
        
        prompt_scores = []
        
        for i, prompt_instruction in enumerate(variations):
            print(f"\nEvaluating Prompt [{i+1}]...")
            total_score = 0.0
            
            for j, item in enumerate(self.dataset):
                question = item["input"]
                expected = item["expected"]
                
                full_prompt = f"{prompt_instruction}\n\nQuestion: {question}\nAnswer:"
                
                # Model attempts to answer the GSM8K question
                answer = self.evaluator.run_model(full_prompt, max_tokens=150)
                
                # Judge evaluates the answer against the correct expected number
                score = self.evaluator.evaluate_with_judge(question, answer, expected)
                total_score += score
                
                print(f"  Test {j+1}: Score = {score}")
                
            mean_score = total_score / len(self.dataset)
            prompt_scores.append({
                "prompt": prompt_instruction,
                "score": mean_score,
                "id": i+1
            })
            print(f"  => Mean Score for Prompt [{i+1}]: {mean_score:.2f}")

        self._print_results(prompt_scores)

    def _print_results(self, prompt_scores: list[dict]):
        print("\n" + "=" * 50)
        print("Final Results (Tie-breaker: Shortest prompt wins)")
        print("=" * 50)
        
        # Sort by score descending, then by prompt length ascending
        prompt_scores.sort(key=lambda x: (-x["score"], len(x["prompt"])))
        
        for rank, item in enumerate(prompt_scores):
            print(f"Rank {rank+1} | Score: {item['score']:.2f} | Length: {len(item['prompt'])} chars | Prompt [{item['id']}]")
            
        best_prompt = prompt_scores[0]
        
        print("\n🏆 BEST PROMPT 🏆")
        print(f"Score: {best_prompt['score']:.2f}")
        print(f"Length: {len(best_prompt['prompt'])} characters")
        print(best_prompt['prompt'])
