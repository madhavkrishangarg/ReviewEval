import os
import pandas as pd
import csv

# Directories
source_dir = 'results'
done_dir = 'results/done'
output_dir = 'results/merged'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of CSV files in both directories
source_files = [f for f in os.listdir(source_dir) if f.endswith('.csv') and not f.startswith('.')]
done_files = [f for f in os.listdir(done_dir) if f.endswith('.csv')]

# Process each file in done_dir
for done_file in done_files:
    # Check if the same file exists in source_dir
    if done_file in source_files:
        # Read the CSV files
        source_path = os.path.join(source_dir, done_file)
        done_path = os.path.join(done_dir, done_file)
        
        try:
            # Read CSV files
            source_df = pd.read_csv(source_path)
            done_df = pd.read_csv(done_path)
            
            # Merge the dataframes on 'paper_number' and 'conference'
            # Keep all columns from done_df and add factual_correctness from source_df
            if 'factual_correctness' in source_df.columns:
                # Create a mapping from (paper_number, conference) to factual_correctness
                factual_dict = {}
                for _, row in source_df.iterrows():
                    key = (row['paper_number'], row['conference'])
                    factual_dict[key] = row.get('factual_correctness', None)
                
                # Add factual_correctness to done_df
                if 'factual_correctness' not in done_df.columns:
                    done_df['factual_correctness'] = done_df.apply(
                        lambda row: factual_dict.get((row['paper_number'], row['conference']), None),
                        axis=1
                    )
            
            # Save the merged file
            output_path = os.path.join(output_dir, done_file)
            done_df.to_csv(output_path, index=False)
            print(f"Merged {done_file}")
        
        except Exception as e:
            print(f"Error processing {done_file}: {e}")
    else:
        print(f"No matching source file for {done_file}")

print("Merge completed!") 