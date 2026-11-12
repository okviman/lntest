"""Cell polygons and their graph-based cluster from the 10x segmentation GeoJSON, as GeoParquet.

    python lymphnode/extract_cluster_from_geojson.py <cells.geojson> <out.parquet> --keep-clusters Cluster-1,Cluster-3

Run by lymphnode/reproduce_vishd_cluster1_cluster3.sh.
"""

import geopandas as gpd
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract Cluster names from GeoJSON classification.")
    parser.add_argument("input_file", help="Path to input GeoJSON file")
    parser.add_argument("output_file", help="Path to output Parquet file")
    
    parser.add_argument("--keep-clusters", help="Comma-separated list of cluster names to keep (e.g., 'Cluster-6,Cluster-7')")
    
    args = parser.parse_args()

    try:
        # Read GeoJSON
        gdf = gpd.read_file(args.input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Function to parse classification string
    def extract_cluster_name(val):
        # Handle cases where value might be None, string, or already a dict
        if val is None:
            return None
        
        if isinstance(val, dict):
            return val.get("name")
            
        try:
            # Assume it's a JSON string
            data = json.loads(val)
            if isinstance(data, dict):
                return data.get("name")
        except (json.JSONDecodeError, TypeError):
            pass
            
        return None

    if 'classification' in gdf.columns:
        gdf['Cluster'] = gdf['classification'].apply(extract_cluster_name)
    else:
        print("Warning: 'classification' column not found.")
        gdf['Cluster'] = None

    # Keep only the Cluster column
    output_df = gdf.copy()

    # Filter if keep_clusters is provided
    if args.keep_clusters:
        clusters_to_keep = [c.strip() for c in args.keep_clusters.split(',')]
        output_df = output_df[output_df['Cluster'].isin(clusters_to_keep)]
        print(f"Filtered output to keep clusters: {clusters_to_keep}")

    # Write to Parquet
    try:
        output_df.rename(columns={'cell_id': 'ID'}).to_parquet(args.output_file)
        print(f"Successfully extracted Cluster column to {args.output_file}")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
