#!/usr/bin/env bash
#
# Reproduces data/vishd-cluster1-cluster3-2um*.h5ad from the public 10x
# Visium HD Human Lymph Node (FFPE, CytAssist, v4 / Space Ranger 4.0.1) dataset:
# https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-lymph-node-v4
#
# Pipeline
#   0. download + untar segmented_outputs and binned_outputs      (skipped if present)
#   1. extract_cluster_from_geojson.py : graph-based clusters from the annotated
#      cell segmentations, keeping only Cluster-1 and Cluster-3
#        -> data/cluster_1_and_3_vishd_lymph.parquet         (155,439 cells)
#   2. assign_visium_shapes.py --annotate-no-aggregate : assign every in-tissue
#      2um spot to the unique Cluster-1/Cluster-3 cell polygon containing it
#        -> data/vishd-cluster1-cluster3-2um.h5ad            (1,293,850 spots x 18,132 genes)
#   3. add_clusters_and_subsample.py : attach the parent cell's cluster label,
#      then crop to cells with ID < 40000 (a contiguous band of the section)
#        -> data/vishd-cluster1-cluster3-2um-with-clusters.h5ad             (1,293,850 spots, 155,435 cells)
#        -> data/vishd-cluster1-cluster3-2um-with-clusters-subsampled.h5ad  (  126,008 spots,  13,229 cells)
#
# Step 2 is the slow one (point-in-polygon over ~6M spots x 155k polygons,
# several hours, single threaded). Every step is skipped when its output
# already exists; delete the file or pass FORCE=1 to redo it.
#
# Cost: step 0 downloads 4.4 GB (segmented_outputs 1.8 GB + binned_outputs
# 2.6 GB) and the extracted tree plus the intermediates need well over that
# again on disk. Everything lands in lymphnode/data/, which is gitignored.
#
# Needs geopandas, shapely and pyarrow, which reproducibility/env.yaml installs.
#
# Usage:  bash lymphnode/reproduce_vishd_cluster1_cluster3.sh     # from reproducibility/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$HERE/data}"
# The python on PATH: activate de-ziln-reproducibility first, or set PY=.
PY="${PY:-python}"
FORCE="${FORCE:-0}"

BASE_URL="https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Lymph_Node_FFPE"
SHAPES="$DATA/cluster_1_and_3_vishd_lymph.parquet"
SPOTS="$DATA/vishd-cluster1-cluster3-2um.h5ad"
WITH_CLUSTERS="$DATA/vishd-cluster1-cluster3-2um-with-clusters.h5ad"
SUBSAMPLED="$DATA/vishd-cluster1-cluster3-2um-with-clusters-subsampled.h5ad"

need() { [[ "$FORCE" == "1" || ! -s "$1" ]]; }

mkdir -p "$DATA"
cd "$DATA"

# ---------------------------------------------------------------- 0. download
for tarball in segmented_outputs binned_outputs; do
	if [[ ! -d "$DATA/$tarball" ]]; then
		archive="Visium_HD_Human_Lymph_Node_FFPE_${tarball}.tar.gz"
		[[ -s "$DATA/$archive" ]] || curl -O "$BASE_URL/$archive"
		echo "Extracting $archive ..."
		tar -xzf "$DATA/$archive" -C "$DATA"
	fi
done

# ------------------------------------------------- 1. clusters from the GeoJSON
if need "$SHAPES"; then
	echo "== [1/3] extracting Cluster-1 / Cluster-3 cell polygons =="
	"$PY" "$HERE/extract_cluster_from_geojson.py" \
		"$DATA/segmented_outputs/graphclust_annotated_cell_segmentations.geojson" \
		"$SHAPES" \
		--keep-clusters "Cluster-1,Cluster-3"
else
	echo "== [1/3] $SHAPES exists, skipping =="
fi

# ------------------------------------------------- 2. assign 2um spots to cells
# Coordinate convention: shapes are in full-resolution image pixels, so x is
# pxl_col_in_fullres and y is pxl_row_in_fullres (the script defaults, no --swap-xy).
if need "$SPOTS"; then
	echo "== [2/3] assigning 2um spots to cell polygons (slow: hours) =="
	"$PY" "$HERE/assign_visium_shapes.py" \
		--shapes-parquet "$SHAPES" \
		--tissue16-parquet "$DATA/binned_outputs/square_016um/spatial/tissue_positions.parquet" \
		--tissue2-parquet "$DATA/binned_outputs/square_002um/spatial/tissue_positions.parquet" \
		--matrix-h5 "$DATA/binned_outputs/square_002um/filtered_feature_bc_matrix.h5" \
		--annotate-no-aggregate \
		--assignments-out "$DATA/vishd-cluster1-cluster3-2um-assignments.parquet" \
		--sanity-plot "$DATA/vishd-cluster1-cluster3-16um-sanity.png" \
		--agg-h5ad "$SPOTS"
else
	echo "== [2/3] $SPOTS exists, skipping =="
fi

# ------------------------------------------ 3. cluster labels + spatial crop
if need "$WITH_CLUSTERS" || need "$SUBSAMPLED"; then
	echo "== [3/3] adding cluster labels and subsampling (cell_id < 40000) =="
	"$PY" "$HERE/add_clusters_and_subsample.py" \
		--in-h5ad "$SPOTS" \
		--shapes-parquet "$SHAPES" \
		--out-annotated "$WITH_CLUSTERS" \
		--out-subsampled "$SUBSAMPLED" \
		--max-cell-id 40000
else
	echo "== [3/3] $WITH_CLUSTERS and $SUBSAMPLED exist, skipping =="
fi

echo "Done:"
ls -lh "$SHAPES" "$SPOTS" "$WITH_CLUSTERS" "$SUBSAMPLED"
