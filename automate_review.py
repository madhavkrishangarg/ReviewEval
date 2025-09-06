import os
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from review_prompts_generator import (
    parse_guidelines,
    generate_review_prompts,
    getHTMLContent,
)
from gemini import get_gemini_response, prepare_pdf, model_name
from reflection import reflect
from evaluation.main import evaluate_review
from config import CONFIG
import traceback

def get_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def load_paper_metadata():
    with open(CONFIG['metadata_file']) as f:
        return json.load(f)

def get_conference_info(paper_number):
    """Get conference information for a specific paper."""
    metadata = load_paper_metadata()
    for paper in metadata['papers']:
        if paper['paper_number'] == paper_number:
            return paper
    raise ValueError(f"No metadata found for paper {paper_number}")

def get_prompt_cache_path(conference, model):
    """Get the path for cached prompts for a specific conference and model."""
    cache_dir = Path("prompt_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{conference}_{model.replace('/', '_')}_prompts.json"

def get_guidelines_cache_path(conference):
    """Get the path for cached guidelines for a specific conference."""
    cache_dir = Path("guidelines_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{conference}_guidelines.txt"

def prepare_document(file_path, api_key):
    """Prepare PDF document for processing."""
    print(f"Preparing document for: {file_path}")
    return prepare_pdf(api_key, file_path)

def parse_conference_guidelines(guidelines_url, api_key, conference):
    """Parse conference guidelines from URL with caching."""
    cache_path = get_guidelines_cache_path(conference)
    
    # Check if cached guidelines exist
    if cache_path.exists():
        print(f"Loading cached guidelines for {conference}...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Parse new guidelines if cache doesn't exist
    print("Parsing conference guidelines...")
    html_content = getHTMLContent(guidelines_url)
    guidelines = parse_guidelines(html_content, api_key)
    
    # Save to cache - ensure we're writing the content of the AIMessage
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(guidelines.content if hasattr(guidelines, 'content') else str(guidelines))
    
    return guidelines.content if hasattr(guidelines, 'content') else str(guidelines)

def generate_review_section(prompt, section_name, document, api_key):
    """Generate review for a specific section of the paper."""
    print(f"Generating review for section: {section_name}")
    init_response = get_gemini_response(
        str(prompt) + f"following these guidelines, you are supposed to review the {section_name} section of the provided paper. Your review must adhere to the given guidelines and should be given as constructive feedback. Note that you ARE the reviewer who has to adhere to the guidelines and you're not supposed to give guidelines to others. You are the one who will follow these guidelines and provide a good review for the specified section of the paper.",
        f"Give me the review of {section_name.lower()} of the research paper. You are supposed to give me the review of the paper",
        document,
        api_key)
    
    while init_response.content == "":
        init_response = get_gemini_response(
            str(prompt) + f"following these guidelines, you are supposed to review the {section_name} section of the provided paper. Your review must adhere to the given guidelines and should be given as constructive feedback. Note that you ARE the reviewer who has to adhere to the guidelines and you're not supposed to give guidelines to others. You are the one who will follow these guidelines and provide a good review for the specified section of the paper.",
            f"Give me the review of {section_name.lower()} of the research paper. You are supposed to give me the review of the paper",
            document,
            api_key)

    reflection = reflect(prompt, init_response, document, CONFIG['reflection_loops'], api_key)
    while reflection.content == "":
        reflection = reflect(prompt, init_response, document, CONFIG['reflection_loops'], api_key)
    return reflection

def generate_review_prompts_from_guidelines(guidelines, api_key, conference, model):
    """Generate review prompts from conference guidelines with caching."""
    cache_path = get_prompt_cache_path(conference, model)
    
    # Check if cached prompts exist
    if cache_path.exists():
        print(f"Loading cached prompts for {conference} using {model}...")
        with open(cache_path) as f:
            return json.load(f)
    
    # Generate new prompts if cache doesn't exist
    print(f"Generating new prompts for {conference} using {model}...")
    prompts = generate_review_prompts(guidelines, api_key)
    prompts_dict = {i: prompts[i].content for i in prompts}
    
    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(prompts_dict, f)
    
    return prompts_dict

def format_review(review, guidelines, paper, api_key):
    """Format the review according to conference requirements."""
    system_instruction = f"""You are an expert in writing reviews for research paper submissions to high impact conferences.
    You will be given reviews for various sections of a research paper, and the research paper itself and you are supposed **combine all the reviews and rewrite in the comprehensive format that is expected for reviewer comments** to the specified conference.
    You will be given the contents of the reviewer guidelines for the conference and you are supposed to adhere to it strictly. 
    **Do not change or edit the original review content at all; only reformat the structure and headings.**
    The output review should have consolidated main headings for each major criteria of the guidelines along with headings like
        - **Summary**
        - **Strengths**
        - **Weaknesses**
        - **Soundness**
        - **Presentation**
        - **Contribution**
        - **Questions**
        - **Limitations**
        - **Ethics Review**
        - **Rating & Confidence**
        - **Additional Comments/Recommendations** (if provided) (different from Final Comments)
    If some sections are not provided in the input, do not create them. The formatting should adjust dynamically based on the available content.
    The whole review should not be more than **500 words**."""

    prompt = f"""This is the conference guidelines for the conference : {str(guidelines)}
    This is the review to be formatted : {str(review)}"""

    new_review = get_gemini_response(system_instruction, prompt, paper, api_key)
    # print(new_review.content)
    if new_review and new_review.content == "":
        new_review = get_gemini_response(system_instruction, prompt, paper, api_key)
    while new_review.content == "":
        new_review = get_gemini_response(system_instruction, prompt, paper, api_key)
    return new_review

def improve_review(formatted_review, guidelines, review, paper, api_key, pdf_path):
    """Improve the review through multiple iterations based on evaluation metrics."""
    if not CONFIG['improvements'] or not formatted_review:
        return formatted_review
    
    new_review = formatted_review
    evaluation_results = []
    
    for i in range(1):
        print(f"Improvement reflection {i+1}...")
        try:
            eval_result = evaluate_review(guidelines, str(pdf_path), new_review.content)
            
            evaluation_results.append(new_review.content)
            evaluation_results.append(eval_result)

            # print(eval_result)
            
            improvement_prompt = f"""
            You are an expert reviewer. Here is your current review:
            
            ```{new_review.content}```
            
            ## Here are the evaluation results for this review:
            
            Depth of analysis in review: A good review provides a comprehensive, critical evaluation rather than a superficial commentary.
                Total Depth of Analysis Score (Higher is better): {eval_result["depth"]["depth_of_analysis_score"]} out of 1
                Sub-criteria Wise Score (out of 3) (Higher is better):
                    Existing Literature Comparison Score: {eval_result["depth"]["existing_literature_comparison_score"]}
                    Methodological Scrutiny Score: {eval_result["depth"]["methodological_scrutiny_score"]}
                    Results Interpretation Score: {eval_result["depth"]["results_interpretation_score"]}
                    Theoretical Contributions Score: {eval_result["depth"]["theoretical_contributions_score"]}
                    Logical Gaps Identification Score: {eval_result["depth"]["logical_gaps_identification_score"]}
                    
            Adherence to reviewer guidelines: A high-quality review complies with the established criteria and aligns with venue guidelines.
                Total Adherence Score (Higher is better): {eval_result["adherence"]["final_score"]} out of 1
                Criteria Wise Score (out of 3) (Higher is better):
                    {eval_result["adherence"]["criteria_scores"]}
                    
            Actionable Insights: A good review offers concrete guidance for improving the work.
                Total Actionable Insights Score (Higher is better): {eval_result["actionable"]["percentage_of_actionable_insights"]} out of 1
                Non Actionable Insights (Insights which lack specificity, feasibility, implementation details): {", ".join(eval_result["actionable"]["non_actionable_insights"])}

            Factual (Statements in the review which are not true or hallucinated): A good review should be factually correct and not contain any hallucinated orincorrect statements.
                Factually Incorrect Statements (Lower is better): {eval_result["factual"]["negative_incorrect"]}
                Factual Correctness Score (Higher is better): {eval_result["factual"]["factual_correctness_score"]} out of 1
            
            Please improve your current review based on these metrics while maintaining the same format and guidelines and word limit of **500 words**. Concentrate on enhancing sections that received lower scores. Use the knowledge base below to improve the review, but do not alter the original format:

            Here is the knowledge base for the paper:
            Detailed Review: ```{str(review)}```
            Paper Text: ```{str(paper)}```
            """
            
            improved_review = get_gemini_response(system_instruction=None, prompt=improvement_prompt, api_key=api_key)
            if improved_review and improved_review.content == "":
                improved_review = get_gemini_response(system_instruction=None, prompt=improvement_prompt, api_key=api_key)
            
            if improved_review:
                new_review = improved_review
                
        except Exception as e:
            print(f"Error during improvement iteration {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if new_review and hasattr(new_review, 'content'):
        evaluation_results.append(new_review.content)

    # with open("evaluation_results.txt", "w") as f:
    #     for result in evaluation_results:
    #         if isinstance(result, dict):
    #             f.write(str(result))
    #         else:
    #             f.write(str(result))
    #         f.write("\n")
    
    return new_review

def format_review_for_conference(review, guidelines, paper, api_key, pdf_path):
    """Format and improve the review according to conference requirements."""
    formatted_review = format_review(review, guidelines, paper, api_key)
    if formatted_review is None:
        return None
    
    return improve_review(formatted_review, guidelines, review, paper, api_key, pdf_path)

def generate_paper_review(pdf_path, api_key, conference_info):
    """Generate a complete review for a paper."""
    document = prepare_document(pdf_path, api_key)
    
    guidelines = parse_conference_guidelines(conference_info['reviewer_guidelines'], api_key, conference_info['conference'])
    prompts = generate_review_prompts_from_guidelines(
        guidelines, 
        api_key, 
        conference_info['conference'],
        CONFIG['model']
    )
    
    review_sections = {}
    for section in CONFIG['sections']:
        review_sections[section] = generate_review_section(prompts[section], section, document, api_key).content
        print(f"Completed review for section: {section}")

    final_review = "\n".join(review_sections.values())
    formatted_review = format_review_for_conference(final_review, guidelines, document, api_key, pdf_path)
    return formatted_review.content

def process_single_paper(pdf_path, api_key):
    """Process a single paper and generate its review."""
    try:
        pdf_path = Path(pdf_path)
        paper_name = pdf_path.stem
        paper_number = int(paper_name.split('_')[-1] if '_' in paper_name else paper_name)
        
        conference_info = get_conference_info(paper_number)
        
        # Determine output path first to check if review already exists
        output_dir = Path(CONFIG['output_folder'])
        # Combine model name and reflection loops in directory name
        model_dir_name = f"{CONFIG['model'].replace('/', '_')}_reflections_{CONFIG['reflection_loops']}"
        model_dir = output_dir / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        review_filename = f"{paper_number}.txt"
        review_path = model_dir / review_filename
        
        # Check if review file already exists
        if review_path.exists():
            # print(f"Review for paper {paper_number} already exists at {review_path}. Skipping.")
            return f"Skipped paper {paper_number} - review already exists"
        
        print(f"Started processing paper {paper_number} for {conference_info['conference']}")
        
        review = generate_paper_review(str(pdf_path), api_key, conference_info)

        if review is None:
            review = generate_paper_review(str(pdf_path), api_key, conference_info)
        
        with open(review_path, 'w', encoding='utf-8') as f:
            f.write(review)
        print(f"Written review at: {review_path}")
        return f"Successfully generated review for paper {paper_number}"
        
    except Exception as e:
        return f"Error processing paper {paper_number}: {str(e)}\nTraceback: {traceback.format_exc()}"

def process_folder_papers(folder_path, api_key, max_workers=CONFIG['max_workers']):
    """Process all papers in a folder in parallel."""
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(process_single_paper, pdf_path, api_key): pdf_path
            for pdf_path in pdf_files
        }
        
        for future in as_completed(future_to_paper):
            result = future.result()
            print(result)

    print("Finished processing all papers in the folder")

def main():
    """Main function to run the review system."""
    print(f"Current working directory: {os.getcwd()}")
    
    api_key = get_api_key()
    
    Path(CONFIG['output_folder']).mkdir(parents=True, exist_ok=True)
    
    process_folder_papers(CONFIG['input_folder'], api_key)

if __name__ == "__main__":
    main()