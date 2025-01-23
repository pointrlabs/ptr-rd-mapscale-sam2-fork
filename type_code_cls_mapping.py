import os
import json
from collections import Counter
import pandas as pd
from pathlib import Path
import yaml

def analyze_type_codes(root_dir):
    """
    Recursively analyze all GeoJSON files under root_dir and extract type-code statistics
    """
    type_code_counter = Counter()
    file_count = 0
    type_code_examples = {}  # Store example properties for each type code
    
    # Recursively find all GeoJSON files
    for path in Path(root_dir).rglob('*.geojson'):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            if 'features' not in data:
                continue
                
            # Process each feature in the GeoJSON
            for feature in data['features']:
                properties = feature.get('properties', {})
                type_code = properties.get('typeCode')
                
                if type_code:
                    type_code_counter[type_code] += 1
                    
                    # Store first example of properties for each type code
                    if type_code not in type_code_examples:
                        type_code_examples[type_code] = properties
                        
            file_count += 1
            
            if file_count % 100 == 0:
                print(f"Processed {file_count} files...")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return {
        'total_files': file_count,
        'unique_type_codes': len(type_code_counter),
        'type_code_counts': dict(type_code_counter),
        'type_code_examples': type_code_examples
    }

def create_class_mapping(stats, output_dir):
    """
    Create and save class mapping files with pre-filled class names
    """
    # Get counts
    counts = stats['type_code_counts']
    
    # Create DataFrame with pre-filled class information
    df = pd.DataFrame([
        {
            'type_code': tc,
            'count': count,
            'class_id': idx + 1,  # 1-based index
            'class_name': tc  # Use type_code as class_name
        }
        for idx, (tc, count) in enumerate(counts.items())
    ])
    
    # Sort by frequency
    df = df.sort_values('count', ascending=False)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'type_code_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nAnalysis exported to {csv_path}")
    print("CSV file created with pre-filled class IDs and names")
    print("You can now run with --generate-mapping flag")
    
    return df

def generate_mapping_files(csv_path, output_dir):
    """
    Generate mapping files from edited CSV
    """
    # Read edited CSV
    df = pd.read_csv(csv_path)
    
    # Debug info
    print("\nDEBUG INFO:")
    print("CSV Columns:", df.columns.tolist())
    print("\nFirst few rows of the CSV:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    
    # Create mappings
    class_mapping = {}
    class_names = {}
    
    # Check for required columns
    required_columns = ['type_code', 'class_id', 'class_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nError: Missing required columns: {missing_columns}")
        return None, None
    
    # Convert class_id to numeric, handling any non-numeric values
    df['class_id'] = pd.to_numeric(df['class_id'], errors='coerce')
    
    # Filter out rows where class_id is not NaN
    valid_rows = df[df['class_id'].notna()]
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Valid rows (with class_id): {len(valid_rows)}")
    
    if len(valid_rows) == 0:
        print("\nNo valid class_id values found. Please make sure class_id column contains numeric values.")
        return None, None
    
    for _, row in valid_rows.iterrows():
        try:
            type_code = str(row['type_code'])
            class_id = int(row['class_id'])
            class_name = str(row['class_name'])
            
            if pd.notna(class_name):  # Check if class_name is valid
                class_mapping[type_code] = class_id
                class_names[class_id] = class_name
            else:
                print(f"Skipping row due to missing class_name: {row.to_dict()}")
        except Exception as e:
            print(f"Error processing row: {row.to_dict()}")
            print(f"Error message: {str(e)}")
            continue
    
    if not class_mapping or not class_names:
        print("\nNo valid mappings created. Please check that:")
        print("1. class_id contains valid integer values")
        print("2. class_name contains valid string values")
        print("3. type_code contains valid string values")
        return None, None
    
    # Save as YAML
    mapping_path = os.path.join(output_dir, 'class_mapping.yaml')
    mapping_data = {
        'class_mapping': class_mapping,
        'class_names': class_names
    }
    
    with open(mapping_path, 'w') as f:
        yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)
    
    # Save as Python
    py_path = os.path.join(output_dir, 'class_mapping.py')
    with open(py_path, 'w') as f:
        f.write('# Auto-generated class mapping\n\n')
        
        # Write CLASS_MAPPING
        f.write('CLASS_MAPPING = {\n')
        for type_code, class_id in sorted(class_mapping.items()):
            f.write(f"    '{type_code}': {class_id},  # {class_names[class_id]}\n")
        f.write('}\n\n')
        
        # Write CLASS_NAMES
        f.write('CLASS_NAMES = {\n')
        for class_id, name in sorted(class_names.items()):
            f.write(f"    {class_id}: '{name}',\n")
        f.write('}\n')
    
    print(f"\nMapping files generated:")
    print(f"- YAML: {mapping_path}")
    print(f"- Python: {py_path}")
    print(f"\nFound {len(class_mapping)} type codes mapped to {len(set(class_mapping.values()))} unique classes")
    
    # Print first few mappings as example
    print("\nExample mappings:")
    for i, (type_code, class_id) in enumerate(list(class_mapping.items())[:5]):
        print(f"  {type_code} -> {class_id} ({class_names[class_id]})")
    
    return class_mapping, class_names

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Root directory containing GeoJSON files')
    parser.add_argument('--output-dir', default='mapping', help='Output directory for mapping files')
    parser.add_argument('--generate-mapping', action='store_true', 
                       help='Generate mapping from edited CSV')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.generate_mapping:
        # Generate mapping files from edited CSV
        csv_path = os.path.join(args.output_dir, 'type_code_analysis.csv')
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Run without --generate-mapping first.")
            return
        generate_mapping_files(csv_path, args.output_dir)
    else:
        # Analyze files and create initial CSV
        print("\nAnalyzing GeoJSON files...")
        stats = analyze_type_codes(args.root_dir)
        
        print("\nAnalysis Summary:")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Unique type codes found: {stats['unique_type_codes']}")
        
        print("\nTop 10 most common type codes:")
        sorted_counts = sorted(stats['type_code_counts'].items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        for type_code, count in sorted_counts[:10]:
            print(f"{type_code}: {count}")
        
        create_class_mapping(stats, args.output_dir)

if __name__ == "__main__":
    main()