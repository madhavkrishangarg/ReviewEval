import pandas as pd
import os
import numpy as np

# Create processed directory if not exists
os.makedirs('processed', exist_ok=True)

# Read reference file with paper numbers to isolate
reference_file = 'results/merged/deepseek_deepseek-chat_ablation-adherence.csv'
reference_df = pd.read_csv(reference_file)
# Filter out any rows that are "AVERAGE" or "STD_DEV"
reference_df = reference_df[~reference_df['paper_number'].isin(['AVERAGE', 'STD_DEV'])]
reference_paper_numbers = set(reference_df['paper_number'].tolist())

print(f"Reference paper numbers: {sorted(reference_paper_numbers)}")

# List of files to process
files_to_process = [
    'results/merged/qwen3-32b-marg-v1.csv',
    'results/merged/qwen3-32b-marg-v2.csv',
    'results/merged/qwen3_reviews_sakana.csv',
    'results/merged/deepseek-chat-marg-v2.csv',
    'results/merged/deepseek-chat-marg-v1.csv',
    'results/merged/deepseek_sakana_reviews.csv',
    'results/merged/expert.csv'
]

# Process each file
for file_path in files_to_process:
    print(f"Processing {file_path}...")
    
    # Read input file
    df = pd.read_csv(file_path)
    
    # Filter out any rows that are "AVERAGE" or "STD_DEV"
    df = df[~df['paper_number'].isin(['AVERAGE', 'STD_DEV'])]
    
    # Filter rows based on paper numbers in reference file
    filtered_df = df[df['paper_number'].isin(reference_paper_numbers)]
    
    # Calculate statistics (average and std dev) for each metric across filtered papers
    metrics = ['depth_score', 'actionable_insights', 'adherence_score', 'coverage', 
               'semantic_similarity', 'factual_correctness']
    
    # Create average row
    avg_row = {'paper_number': 'AVERAGE', 'conference': ''}
    for metric in metrics:
        if metric in filtered_df.columns:
            avg_row[metric] = filtered_df[metric].mean()
    
    # Create std dev row
    std_row = {'paper_number': 'STD_DEV', 'conference': ''}
    for metric in metrics:
        if metric in filtered_df.columns:
            std_row[metric] = filtered_df[metric].std()
    
    # Create output filename
    base_filename = os.path.basename(file_path)
    output_filename = os.path.join('processed', f'processed_{base_filename}')
    
    # Sort by paper_number for consistent output
    filtered_df = filtered_df.sort_values('paper_number')
    
    # Add statistics rows
    stats_df = pd.DataFrame([avg_row, std_row])
    result_df = pd.concat([filtered_df, stats_df], ignore_index=True)
    
    # Save to file
    result_df.to_csv(output_filename, index=False)
    
    print(f"  Saved to {output_filename}")
    print(f"  Found {len(filtered_df)} matching papers out of {len(df)} total")

print("Processing complete!") 