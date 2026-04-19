def load_benchmark(name: str = "gsm8k", num_samples: int = 3) -> list[dict]:
    """Loads a hardcoded subset of a specific benchmark to avoid network issues."""
    
    benchmarks = {
        "gsm8k": [
            {
                "input": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "expected": "72"
            },
            {
                "input": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                "expected": "10"
            },
            {
                "input": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
                "expected": "5"
            },
            {
                "input": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
                "expected": "42"
            },
            {
                "input": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
                "expected": "624"
            }
        ],
        "logic": [
            {
                "input": "If all bloops are razzies and all razzies are lazzies, are all bloops definitively lazzies? Answer with 1 for yes, 0 for no.",
                "expected": "1"
            },
            {
                "input": "John is twice as old as his sister. In 5 years, John will be 15. How old is his sister now?",
                "expected": "5"
            },
            {
                "input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
                "expected": "5"
            }
        ]
    }
    
    dataset = benchmarks.get(name.lower(), benchmarks["gsm8k"])
    print(f"Loaded {min(num_samples, len(dataset))} samples from {name.upper()} offline benchmark...")
    
    return dataset[:num_samples]
