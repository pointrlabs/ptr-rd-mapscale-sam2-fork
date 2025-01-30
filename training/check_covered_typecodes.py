import yaml
import json
import os
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple

def load_class_mapping(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as f:
        mapping = yaml.safe_load(f)
    return mapping['class_mapping']

def analyze_type_codes(geojson_dir: str, class_mapping: Dict) -> Tuple[Counter, Counter, int]:
    # Get all type codes from the mapping
    covered_type_codes = set(class_mapping.keys())
    uncovered_counter = Counter()
    covered_counter = Counter()
    total_features = 0
    
    # Recursively walk through all files
    for root, dirs, files in os.walk(geojson_dir):
        for file in files:
            if file.endswith('.geojson'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r') as f:
                        geojson = json.load(f)
                        
                    # Check features
                    if 'features' in geojson:
                        for feature in geojson['features']:
                            total_features += 1
                            if 'properties' in feature and 'typeCode' in feature['properties']:
                                type_code = feature['properties']['typeCode']
                                if type_code not in covered_type_codes:
                                    uncovered_counter[type_code] += 1
                                else:
                                    covered_counter[type_code] += 1

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

    return uncovered_counter, covered_counter, total_features

def print_frequency_report(counter: Counter, total_features: int, title: str):
    if not counter:
        return
        
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Type Code':<30} {'Count':<10} {'Percentage':>10}")
    print("-" * 80)
    
    # Sort by frequency, highest first
    for type_code, count in counter.most_common():
        percentage = (count / total_features) * 100
        print(f"{type_code:<30} {count:<10} {percentage:>10.2f}%")

def main():
    # Configure paths
    # Configure paths
    yaml_path = '/home/jovyan/ml-data-ptr/ptr-rd-mapscale-sam2-fork-exp1/mapping/class_mapping_v2.yaml'  # Update this
    geojson_dir = '/home/jovyan/ml-data-ptr/sam2_unit_geojson_dataset_jan3/geojsons'  # Update this
    
    # Load class mapping
    class_mapping = load_class_mapping(yaml_path)
    
    # Find and analyze type codes
    print("Analyzing typeCode frequencies...")
    uncovered_counter, covered_counter, total_features = analyze_type_codes(geojson_dir, class_mapping)
    
    # Report results
    print(f"\nTotal number of features analyzed: {total_features}")
    
    # Print uncovered type codes with frequencies
    print_frequency_report(uncovered_counter, total_features, "UNCOVERED TYPE CODES")
    
    # Print covered type codes with frequencies
    print_frequency_report(covered_counter, total_features, "COVERED TYPE CODES")
    
    # Summary statistics
    total_uncovered = sum(uncovered_counter.values())
    total_covered = sum(covered_counter.values())
    
    print("\nSUMMARY")
    print("-" * 80)
    print(f"Total covered features: {total_covered} ({(total_covered/total_features)*100:.2f}%)")
    print(f"Total uncovered features: {total_uncovered} ({(total_uncovered/total_features)*100:.2f}%)")
    print(f"Number of unique uncovered type codes: {len(uncovered_counter)}")
    print(f"Number of unique covered type codes in use: {len(covered_counter)}")
    print(f"Number of type codes in mapping: {len(class_mapping)}")

if __name__ == "__main__":
    main()