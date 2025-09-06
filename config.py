CONFIG = {
    'input_folder': "papers/30-papers",
    'output_folder': "our_reviews/30-papers/no-improvement",
    'metadata_file': "papers/paper_metadata.json",
    'max_workers': 10,
    'sections': ["Motivation", "Prior Work", "Approach", "Evidence", "Contribution", "Presentation"],
    "reflection_loops": 0,
    "improvements": False,
    # "model": "qwen/qwen-2.5-72b-instruct"
    # "model": "qwen/qwen3-32b"
    # "model": "qwen/qwen-turbo"
    # "model": "openai/gpt-4.1-nano"
    # "model": "deepseek/deepseek-chat"
    # "model": "nvidia/llama-3.1-nemotron-70b-instruct"
    # "model": "mistralai/ministral-3b"
    # "model": "claude-3-5-haiku-latest"
    "model": "gpt-4o-mini"
} 