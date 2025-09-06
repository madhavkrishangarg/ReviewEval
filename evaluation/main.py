from evaluation.depth_of_analysis import depth_of_analysis
from evaluation.actionable_insights import actionable_insights
from evaluation.adherence import adherence_analysis
from evaluation.factual_correctness import factual_correctness
from evaluation.ai_human import ai_human_evaluation
from evaluation.model import model_name as model_name_for_evaluation

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import os
import pandas as pd
from pathlib import Path
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_review(guidelines: str, paper_path: str, review: str) -> Dict[str, Any]:
    """
    Evaluates a review across multiple dimensions in parallel.
    
    Args:
        guidelines (str): The conference guidelines text
        paper_path (str): Path to the research paper being reviewed
        review (str): The review text to evaluate
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation results for each dimension
    """
    results = {}
    paper_id = os.path.basename(paper_path).split('.')[0]
    print(f"Paper {paper_id}: Starting evaluation across all metrics")

    default_results = {
        'depth': {"depth_of_analysis_score": 0.5, "existing_literature_comparison_score": 1, 
                 "methodological_scrutiny_score": 1, "results_interpretation_score": 1,
                 "theoretical_contributions_score": 1, "logical_gaps_identification_score": 1},
        'actionable': {"percentage_of_actionable_insights": 0.5, "non_actionable_insights": []},
        'adherence': {"final_score": 0.5, "criteria_scores": {}},
        'factual': {"factual_correctness_score": 0.0, "negative_incorrect": 0}
    }
    
    tasks = [
        ('depth', depth_of_analysis, [review]),
        ('actionable', actionable_insights, [review]),
        ('adherence', adherence_analysis, [guidelines, review]),
        ('factual', factual_correctness, [paper_path, review])
    ]
    
    completed_tasks = 0
    total_tasks = len(tasks)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for key, func, args in tasks:
            if key == 'factual' and (not paper_path or not os.path.exists(paper_path)):
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"[DONE] {key.capitalize()}")
                continue
                
            try:
                futures[key] = executor.submit(func, *args)
            except Exception as e:
                print(f"Error setting up {key} evaluation: {str(e)}")
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE with error ({completed_tasks}/{total_tasks})")
        
        for key, future in futures.items():
            try:
                results[key] = future.result()
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE ({completed_tasks}/{total_tasks})")
            except Exception as e:
                print(f"Error in {key} evaluation: {str(e)}")
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE with error ({completed_tasks}/{total_tasks})")
    
    for key in default_results:
        if key not in results:
            print(f"Missing evaluation for {key}, using default values")
            results[key] = default_results[key]
    
    print(f"Paper {paper_id}: All metrics completed")
    return results

def complete_evaluation(guidelines: str, paper: str, ai_review: str, expert_review: str):
    """
    Evaluates a review across all dimensions including AI-human comparison in parallel.
    
    Args:
        guidelines (str): The conference guidelines text
        paper (str): Path to the research paper being reviewed
        ai_review (str): The AI-generated review text to evaluate
        expert_review (str): The expert review for comparison
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation results for all dimensions
    """
    results = {}
    paper_id = os.path.basename(paper).split('.')[0]
    print(f"Paper {paper_id}: Starting evaluation across all metrics")

    default_results = {
        # 'depth': {"depth_of_analysis_score": 0.5, "existing_literature_comparison_score": 1, 
                #  "methodological_scrutiny_score": 1, "results_interpretation_score": 1,
                #  "theoretical_contributions_score": 1, "logical_gaps_identification_score": 1},
        # 'actionable': {"percentage_of_actionable_insights": 0.5, "non_actionable_insights": []},
        # 'adherence': {"final_score": 0.5, "criteria_scores": {}},
        'factual': {"factual_correctness_score": 0.0, "negative_incorrect": 0},
        # 'ai_human': {"coverage": 0.5, "semantic_similarity": 0.5}
    }
    
    tasks = [
        # ('depth', depth_of_analysis, [ai_review]),
        # ('actionable', actionable_insights, [ai_review]),
        # ('adherence', adherence_analysis, [guidelines, ai_review]),
        ('factual', factual_correctness, [paper, ai_review]),
        # ('ai_human', ai_human_evaluation, [ai_review, expert_review])
    ]
    
    completed_tasks = 0
    total_tasks = len(tasks)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for key, func, args in tasks:
            if key == 'factual' and (not paper or not os.path.exists(paper)):
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE with error (missing paper) ({completed_tasks}/{total_tasks})")
                continue
                
            try:
                futures[key] = executor.submit(func, *args)
            except Exception as e:
                print(f"Error setting up {key} evaluation: {str(e)}")
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE with error ({completed_tasks}/{total_tasks})")
        
        for key, future in futures.items():
            try:
                results[key] = future.result()
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE ({completed_tasks}/{total_tasks})")
            except Exception as e:
                print(f"Error in {key} evaluation: {str(e)}")
                results[key] = default_results[key]
                completed_tasks += 1
                print(f"Paper {paper_id}: {key.capitalize()} metric DONE with error ({completed_tasks}/{total_tasks})")
    
    for key in default_results:
        if key not in results:
            print(f"Missing evaluation for {key}, using default values")
            results[key] = default_results[key]
    
    print(f"Paper {paper_id}: All metrics completed")
    return results
        
# Function to process a single paper
def process_paper(review_file, papers_dir, expert_reviews_dir, guidelines_dir, paper_conferences):
    paper_number = review_file.stem
    print(f"Started processing review for paper {paper_number}...")
    
    try:
        # Find corresponding paper
        paper_file = None
        for ext in ['.pdf', '.txt']:
            potential_paper = papers_dir / f"{paper_number}{ext}"
            if potential_paper.exists():
                paper_file = potential_paper
                break
        
        if not paper_file:
            print(f"  Warning: No paper file found for review {review_file}")
            return None
        
        # Find corresponding expert review if directory provided
        expert_review_file = None
        if expert_reviews_dir:
            for ext in ['.txt']:
                potential_expert = expert_reviews_dir / f"{paper_number}{ext}"
                if potential_expert.exists():
                    expert_review_file = potential_expert
                    break
        
        # Get conference guidelines
        conference = paper_conferences.get(paper_number)
        if not conference:
            print(f"  Warning: No conference information found for paper {paper_number}")
            return None
        
        guideline_file = guidelines_dir / f"{conference}_guidelines.txt"
        if not guideline_file.exists():
            print(f"  Warning: No guidelines found for conference {conference}")
            return None
        
        # Load content of files
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                review_text = f.read()
            
            with open(guideline_file, 'r', encoding='utf-8') as f:
                guidelines_text = f.read()
            
            # Don't read PDF files directly - use the file path for evaluate_review
            paper_path = str(paper_file)
            
            # Run evaluation
            if expert_review_file:
                try:
                    with open(expert_review_file, 'r', encoding='utf-8', errors='replace') as f:
                        expert_review_text = f.read()
                    
                    # Use complete_evaluation
                    evaluation_results = complete_evaluation(
                        guidelines_text, 
                        paper_path,  # Pass the path, not content
                        review_text, 
                        expert_review_text
                    )
                except Exception as e:
                    print(f"  Error using complete_evaluation for {paper_number}: {e}")
                    # Fallback to basic evaluation
                    evaluation_results = evaluate_review(
                        guidelines_text,
                        paper_path,
                        review_text
                    )
            else:
                # Use evaluate_review without AI-human comparison
                evaluation_results = evaluate_review(
                    guidelines_text,
                    paper_path,
                    review_text
                )
        except UnicodeDecodeError as e:
            print(f"  Error reading files with UTF-8 encoding for {paper_number}: {e}")
            print(f"  Trying with a different encoding...")
            
            # Try again with error handling for encoding issues
            try:
                with open(review_file, 'r', encoding='latin-1') as f:
                    review_text = f.read()
                
                with open(guideline_file, 'r', encoding='utf-8', errors='replace') as f:
                    guidelines_text = f.read()
                
                paper_path = str(paper_file)
                
                # Run evaluation with the file path
                evaluation_results = evaluate_review(
                    guidelines_text,
                    paper_path,
                    review_text
                )
            except Exception as e2:
                print(f"  Error after retrying with different encoding for {paper_number}: {e2}")
                return None
        
        # Create paper data for DataFrame
        paper_data = {
            'paper_number': paper_number,
            'conference': conference,
            # 'depth_score': evaluation_results['depth']['depth_of_analysis_score'],
            # 'actionable_insights': evaluation_results['actionable']['percentage_of_actionable_insights'],
            # 'adherence_score': evaluation_results['adherence']['final_score'],
            'factual_correctness': evaluation_results['factual']['factual_correctness_score'],
            # 'coverage': evaluation_results['ai_human']['coverage'],
            # 'semantic_similarity': evaluation_results['ai_human']['semantic_similarity']
        }
        
        print(f"Paper {paper_number}: COMPLETED")
        return (paper_number, evaluation_results, paper_data)
        
    except Exception as e:
        print(f"Error processing review {review_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Attempt to set start method to 'spawn' for macOS compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Successfully set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or could not be set to 'spawn'.")

    import json
    import os
    from pathlib import Path
    import pandas as pd
    import time
    
    dirs = [
        "reviews/120-reviews"
            ]
    # =====================================
    # CHANGE THIS VALUE to your exact reviews directory path
    # =====================================
    for reviews_dir in dirs:
        # =====================================
        
        papers_dir = "papers/120-papers"
        expert_reviews_dir = "reviews/120-reviews"
        guidelines_dir = "guidelines_cache"
        metadata_file = "papers/paper_metadata.json"
        
        # Convert to Path objects
        reviews_dir_path = Path(reviews_dir)
        papers_dir_path = Path(papers_dir)
        expert_reviews_dir_path = Path(expert_reviews_dir) if expert_reviews_dir else None
        guidelines_dir_path = Path(guidelines_dir)
        
        # Get model name for output filename
        model_name = reviews_dir_path.name
        model_name_for_evaluation = model_name_for_evaluation.split("/")[-1]
        df_output_file = f"{model_name_for_evaluation}_{model_name}.csv"
        
        print(f"\n===== Processing directory: {model_name} =====")
        print(f"Processing reviews from: {reviews_dir_path}")
        print(f"Using papers from: {papers_dir_path}")
        
        num_workers = 10
        print(f"Using {num_workers} parallel workers (ProcessPoolExecutor with spawn)")
        print(f"Output will be saved to: {df_output_file}")
        
        if expert_reviews_dir:
            print(f"Using expert reviews from: {expert_reviews_dir_path}")
        
        # Load paper metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata for {len(metadata.get('papers', []))} papers")
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            metadata = {"papers": []}
        
        # Create a dictionary mapping paper numbers to conference names
        paper_conferences = {}
        for paper in metadata.get("papers", []):
            paper_number = paper.get("paper_number")
            conference = paper.get("conference")
            if paper_number and conference:
                paper_conferences[str(paper_number)] = conference
        
        # Collect all review files
        review_files = list(reviews_dir_path.glob('*.txt'))
        total_papers = len(review_files)
        print(f"Found {total_papers} papers to process")
        
        # Initialize data structures for results
        results = {}
        df_data = []
        start_time = time.time()
        completed_papers = 0
        
        # Using ProcessPoolExecutor now
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            
            # Submit all tasks
            for review_file in review_files:
                futures[executor.submit(
                    process_paper, 
                    review_file, 
                    papers_dir_path, 
                    expert_reviews_dir_path, 
                    guidelines_dir_path, 
                    paper_conferences
                )] = review_file
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    paper_number, evaluation_results, paper_data = result
                    
                    # Store results
                    results[paper_number] = evaluation_results
                    df_data.append(paper_data)
                    
                    # Update progress
                    completed_papers += 1
                    progress_pct = (completed_papers / total_papers) * 100
                    print(f"Overall progress: {completed_papers}/{total_papers} papers ({progress_pct:.1f}%)")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print summary
        print(f"\nProcessed {len(results)} papers in {elapsed_time:.2f} seconds")
        
        if results:
            print(f"\n===== EVALUATION SUMMARY FOR {model_name} =====")
            # avg_depth = sum(r['depth']['depth_of_analysis_score'] for r in results.values()) / len(results)
            # avg_actionable = sum(r['actionable']['percentage_of_actionable_insights'] for r in results.values()) / len(results)
            # avg_adherence = sum(r['adherence']['final_score'] for r in results.values()) / len(results)
            avg_factual = sum(r['factual']['factual_correctness_score'] for r in results.values()) / len(results)
            # avg_coverage = sum(r['ai_human']['coverage'] for r in results.values()) / len(results)
            # avg_semantic_similarity = sum(r['ai_human']['semantic_similarity'] for r in results.values()) / len(results)
            
            print(f"Average scores across {len(results)} papers:")
            # print(f"- Depth of analysis: {avg_depth:.2f}")
            # print(f"- Actionable insights: {avg_actionable:.2f}")
            # print(f"- Adherence to guidelines: {avg_adherence:.2f}")
            print(f"- Factual correctness: {avg_factual:.2f}")
            # print(f"- Coverage: {avg_coverage:.2f}")
            # print(f"- Semantic similarity: {avg_semantic_similarity:.2f}")
            
            # Create DataFrame from collected data
            if df_data:
                df = pd.DataFrame(df_data)
                
                # Calculate average scores and add as a new row
                avg_row = {'paper_number': 'AVERAGE', 'conference': ''}
                std_row = {'paper_number': 'STD_DEV', 'conference': ''}
                
                for col in df.columns:
                    if col not in ['paper_number', 'conference'] and df[col].dtype in ['float64', 'int64']:
                        avg_row[col] = df[col].mean()
                        std_row[col] = df[col].std()
                
                # Append average and std dev rows to DataFrame
                df = pd.concat([df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
                
                # Save DataFrame
                try:
                    # Create results directory if it doesn't exist
                    results_dir = Path("results")
                    results_dir.mkdir(exist_ok=True)
                    
                    # Save file in results directory
                    output_path = results_dir / df_output_file
                    if str(output_path).endswith('.xlsx'):
                        df.to_excel(output_path, index=False)
                    else:  # Default to CSV
                        df.to_csv(output_path, index=False)
                    print(f"\nDataFrame saved to {output_path}")
                except Exception as e:
                    print(f"Error saving DataFrame: {e}")
