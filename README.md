# PromptOptima 🚀

PromptOptima is an automated, algorithmic prompt engineering framework inspired by the concept that Large Language Models (LLMs) can act as human-level prompt engineers. 

This tool uses the **Gemini API** to generate multiple unique prompt variations for a given task and then evaluates them using a local **Ollama** model (`phi3`) against standardized benchmarks (like GSM8K or logic puzzles). The prompt that guides the local model to the highest score is mathematically selected as the winner.

## 🏗️ Architecture

The codebase is highly modular to allow for easy contributions and extensions:

*   **`main.py`**: The entry point that wires all modules together and kicks off the optimization loop.
*   **`generator.py`**: Contains the `PromptGenerator` class. It interfaces with the Gemini API to intelligently generate prompt variations of differing styles and qualities.
*   **`evaluator.py`**: Contains the `OllamaEvaluator` class. It manages the local LLM execution (for answering questions) and the "LLM Judge" logic (for scoring answers from 0.0 to 1.0).
*   **`optimizer.py`**: Contains the `PromptOptimizer` class. It runs the evaluation loop, calculates mean scores, and applies tie-breaker logic (shortest prompt length wins ties).
*   **`dataset_loader.py`**: A dedicated module for loading benchmark datasets (currently supports hardcoded `gsm8k` and `logic` subsets for fast offline testing).

## 🚀 Getting Started

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) installed and running locally with the `phi3` model pulled (`ollama pull phi3`).
3. A Google Gemini API Key.

### Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an environment file:
   Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### Running the Optimizer

Make sure your Ollama server is running in the background, then simply execute:
```bash
python main.py
```

The script will:
1. Load a subset of benchmark questions.
2. Generate prompt variations.
3. Use Ollama to evaluate each prompt.
4. Output a ranked leaderboard crowning the best prompt!

## 🤝 Contributing

Contributions are welcome! If you want to help improve PromptOptima, consider contributing in the following areas:
*   **New Benchmarks**: Add support for more datasets in `dataset_loader.py`.
*   **More Evaluators**: Implement a class in `evaluator.py` to support cloud models (like OpenAI or Anthropic) instead of just local Ollama.
*   **Advanced Tie-breakers**: Improve the sorting algorithms in `optimizer.py` (e.g., punishing high token-count prompts).

Please open an issue or submit a Pull Request!
