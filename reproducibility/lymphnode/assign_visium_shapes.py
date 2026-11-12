#!/usr/bin/env python3
"""Assign Visium HD 2 um spots to shapes and aggregate their counts.

Steps:
1) Load the shapes GeoParquet (with 'ID' and 'geometry'), compute the bounding box of each shape and sort by
   (minx, maxx, miny, maxy).
2) Load the 16 um tissue positions and plot them with the shapes as a sanity check.
3) Load the 2 um tissue positions. For each spot, filter the candidate shapes by bounding box, then test whether the
   spot is within each polygon. A spot in exactly one shape is assigned to it. Spots in none or in several are skipped
   and counted.
4) Load the 10x filtered_feature_bc_matrix.h5 of the 2 um bins and aggregate the counts per shape ID.
5) Save the aggregated h5ad and the optional plots.
"""
import argparse
from typing import List, Tuple, Optional, Dict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
import matplotlib.pyplot as plt


def read_shapes(shapes_parquet: str) -> gpd.GeoDataFrame:
	gdf = gpd.read_parquet(shapes_parquet)
	if "geometry" not in gdf.columns:
		raise ValueError("Shapes parquet must have a 'geometry' column.")
	if "ID" not in gdf.columns:
		raise ValueError("Shapes parquet must include an 'ID' column.")
	gdf = gdf.set_geometry("geometry")
	return gdf


def compute_sorted_bboxes(shapes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
	bounds = shapes_gdf.geometry.bounds.rename(columns={"minx": "x_min", "miny": "y_min", "maxx": "x_max", "maxy": "y_max"})
	df = pd.concat([shapes_gdf[["ID"]].reset_index(drop=True), bounds.reset_index(drop=True)], axis=1)
	df_sorted = df.sort_values(by=["x_min", "x_max", "y_min", "y_max"], ascending=[True, True, True, True], kind="mergesort").reset_index(drop=True)
	return df_sorted


def read_positions_parquet(path: str, x_col: str, y_col: str, barcode_col: Optional[str] = None, filter_in_tissue: bool = True) -> pd.DataFrame:
	df = pd.read_parquet(path)
	for col in (x_col, y_col):
		if col not in df.columns:
			raise ValueError(f"Column '{col}' not in {path}")
	if barcode_col and barcode_col not in df.columns:
		raise ValueError(f"Barcode column '{barcode_col}' not in {path}")
	# If available, filter to in_tissue == 1 for plotting/assignment
	if filter_in_tissue and "in_tissue" in df.columns:
		df = df[df["in_tissue"] == 1].copy()
	return df


def plot_shapes_and_points(shapes_gdf: gpd.GeoDataFrame, pts_df: pd.DataFrame, x_col: str, y_col: str, out_png: Optional[str], figsize_inches: Tuple[float, float] = (10.0, 10.0), dpi: int = 300) -> None:
	fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)
	shapes_gdf.boundary.plot(ax=ax, color="black", linewidth=0.6, label="shapes",aspect="equal")
	ax.scatter(pts_df[x_col].to_numpy(), pts_df[y_col].to_numpy(), s=1, c="red", alpha=0.7, label="points")
	ax.set_aspect("equal", adjustable="box")
	ax.legend(loc="best")
	ax.set_title("Sanity check: 16um points vs shapes")
	if out_png:
		fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


def build_prepared_shapes(shapes_gdf: gpd.GeoDataFrame) -> List:
	return [prep(geom) for geom in shapes_gdf.geometry.tolist()]


def assign_spots_to_shapes(shapes_gdf: gpd.GeoDataFrame, shapes_bbox_sorted: pd.DataFrame, prepared_shapes: List, spots_df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[pd.DataFrame, int, int, int]:
	ids = shapes_gdf["ID"].to_numpy()
	# For bbox filtering, keep arrays
	x_min = shapes_bbox_sorted["x_min"].to_numpy()
	x_max = shapes_bbox_sorted["x_max"].to_numpy()
	y_min = shapes_bbox_sorted["y_min"].to_numpy()
	y_max = shapes_bbox_sorted["y_max"].to_numpy()
	id_sorted = shapes_bbox_sorted["ID"].to_numpy()
	# Map ID to index in original shapes_gdf to locate geometry/prepared
	id_to_index: Dict = {shape_id: int(idx) for idx, shape_id in enumerate(ids)}

	assignments: List[Tuple[str, str]] = []  # (barcode, shape_id)
	total = 0
	skipped_multi = 0
	skipped_none = 0
	for _, row in spots_df.iterrows():
		total += 1
		x = float(row[x_col]); y = float(row[y_col])
		pt = Point(x, y)
		# bbox candidate mask
		cand_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
		if not cand_mask.any():
			skipped_none += 1
			continue
		cand_ids = id_sorted[cand_mask]
		matches: List[str] = []
		for sid in cand_ids:
			idx = id_to_index[sid]
			if prepared_shapes[idx].contains(pt) or prepared_shapes[idx].covers(pt):
				matches.append(sid)
				if len(matches) > 1:
					break
		if len(matches) == 1:
			barcode = row.get("barcode", row.get("Barcode", row.get("BC", None)))
			if barcode is None:
				# fallback: use index as pseudo-barcode
				barcode = str(_)
			assignments.append((barcode, matches[0]))
		elif len(matches) > 1:
			skipped_multi += 1
		else:
			skipped_none += 1
	assign_df = pd.DataFrame(assignments, columns=["barcode", "shape_id"])
	return assign_df, total, skipped_multi, skipped_none


def annotate_adata_with_shapes(h5_10x_path: str, assignments_df: pd.DataFrame, out_h5ad: str) -> None:
	"""Annotate adata with shape_id without aggregating counts."""
	try:
		import scanpy as sc
	except Exception as exc:
		raise ImportError("scanpy is required to read 10x h5 and write h5ad. pip install scanpy") from exc
	adata = sc.read_10x_h5(h5_10x_path)
	# Ensure barcodes index matches 'barcode' column
	adata.var_names_make_unique()
	barcodes = pd.Index(adata.obs_names)
	# Filter assignments to barcodes present
	assignments_df = assignments_df[assignments_df["barcode"].isin(barcodes)].copy()
	if assignments_df.empty:
		raise ValueError("No assigned barcodes matched the 10x h5 barcodes.")
	# Keep only assigned barcodes in adata
	adata = adata[assignments_df["barcode"].values, :].copy()
	# Add shape_id annotation to obs
	adata.obs["shape_id"] = assignments_df["shape_id"].values
	# Write annotated adata
	adata.write(out_h5ad)


def aggregate_counts_to_shapes(h5_10x_path: str, assignments_df: pd.DataFrame, out_h5ad: str) -> None:
	try:
		import scanpy as sc
	except Exception as exc:
		raise ImportError("scanpy is required to read 10x h5 and write h5ad. pip install scanpy") from exc
	adata = sc.read_10x_h5(h5_10x_path)
	# Ensure barcodes index matches 'barcode' column
	adata.var_names_make_unique()
	barcodes = pd.Index(adata.obs_names)
	# Filter assignments to barcodes present
	assignments_df = assignments_df[assignments_df["barcode"].isin(barcodes)].copy()
	if assignments_df.empty:
		raise ValueError("No assigned barcodes matched the 10x h5 barcodes.")
	# Sort assignments by shape_id so rows for each group are contiguous (improves sparse summation performance)
	assignments_df.sort_values(by="shape_id", inplace=True, kind="mergesort")
	# Keep only assigned barcodes in adata in the sorted order
	adata = adata[assignments_df["barcode"].values, :].copy()
	adata.obs["shape_id"] = assignments_df["shape_id"].values
	# Aggregate by shape_id into a sparse matrix
	import scipy.sparse as sp
	shape_ids = pd.Index(adata.obs["shape_id"].unique())  # already in sorted order due to mergesort above
	data_rows_sparse = []
	obs_shape = adata.obs["shape_id"].to_numpy()
	# Compute contiguous group boundaries
	group_counts = pd.Series(obs_shape).value_counts(sort=False)[shape_ids].to_numpy()
	start = 0
	for cnt in group_counts:
		end = start + int(cnt)
		# Sum a contiguous slice; stays sparse
		sum_row = adata.X[start:end, :].sum(axis=0)
		# Ensure CSR 1 x n_genes
		if not sp.issparse(sum_row):
			sum_row = sp.csr_matrix(sum_row)
		else:
			sum_row = sum_row.tocsr()
		data_rows_sparse.append(sum_row)
		start = end
	X_sparse = sp.vstack(data_rows_sparse, format="csr")
	# Build aggregated AnnData (sparse X)
	agg = sc.AnnData(X=X_sparse, obs=pd.DataFrame(index=shape_ids), var=adata.var.copy())
	agg.obs.index.name = "shape_id"
	agg.write(out_h5ad)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Assign Visium HD 2um spots to shapes and aggregate counts.")
	p.add_argument("--shapes-parquet", required=True, help="GeoParquet with columns ['ID', 'geometry'].")
	p.add_argument("--tissue16-parquet", required=True, help="16um tissue positions parquet.")
	p.add_argument("--tissue2-parquet", required=True, help="2um tissue positions parquet.")
	p.add_argument("--matrix-h5", required=True, help="10x filtered_feature_bc_matrix.h5 for 2um.")
	p.add_argument("--x16-col", default="pxl_col_in_fullres", help="X column in 16um positions (default: pxl_col_in_fullres).")
	p.add_argument("--y16-col", default="pxl_row_in_fullres", help="Y column in 16um positions (default: pxl_row_in_fullres).")
	p.add_argument("--x2-col", default="pxl_col_in_fullres", help="X column in 2um positions (default: pxl_col_in_fullres).")
	p.add_argument("--y2-col", default="pxl_row_in_fullres", help="Y column in 2um positions (default: pxl_row_in_fullres).")
	p.add_argument("--swap-xy", action="store_true", help="Swap X/Y interpretation (treat provided X as Y and Y as X).")
	p.add_argument("--barcode2-col", default="barcode", help="Barcode column in 2um parquet (default: barcode).")
	p.add_argument("--sanity-plot", default=None, help="Output PNG path for 16um sanity plot.")
	p.add_argument("--agg-h5ad", required=True, help="Output path for aggregated h5ad (or annotated h5ad if --annotate-no-aggregate is used).")
	p.add_argument("--assignments-out", default=None, help="Optional path to write spot-to-shape assignments (parquet or csv) before aggregation.")
	p.add_argument("--skip-aggregation", action="store_true", help="Write assignments and skip the expression aggregation step.")
	p.add_argument("--annotate-no-aggregate", action="store_true", help="Annotate 2um adata with shape_id without aggregating counts per shape.")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	# 1) Shapes and bbox
	shapes_gdf = read_shapes(args.shapes_parquet)
	bbox_sorted = compute_sorted_bboxes(shapes_gdf)
	print(f"Loaded {len(shapes_gdf)} shapes.")
	# Resolve column names with optional XY swap
	x16_name, y16_name = args.x16_col, args.y16_col
	x2_name, y2_name = args.x2_col, args.y2_col
	if args.swap_xy:
		x16_name, y16_name = y16_name, x16_name
		x2_name, y2_name = y2_name, x2_name
	# 2) 16um plot
	df16 = read_positions_parquet(args.tissue16_parquet, x16_name, y16_name, filter_in_tissue=True)
	plot_shapes_and_points(shapes_gdf, df16, x16_name, y16_name, args.sanity_plot, figsize_inches=(10, 10), dpi=300)
	# 3) 2um assignment
	df2 = read_positions_parquet(args.tissue2_parquet, x2_name, y2_name, barcode_col=args.barcode2_col, filter_in_tissue=True)
	prepared = build_prepared_shapes(shapes_gdf)
	assign_df, total, skipped_multi, skipped_none = assign_spots_to_shapes(shapes_gdf, bbox_sorted, prepared, df2, x2_name, y2_name)
	print(f"2um spots total: {total}, assigned: {len(assign_df)}, skipped_multi: {skipped_multi}, skipped_none: {skipped_none}")
	# Write assignments if requested
	if args.assignments_out:
		out_path = args.assignments_out
		lower = out_path.lower()
		if lower.endswith(".parquet"):
			assign_df.to_parquet(out_path, index=False)
		elif lower.endswith(".csv"):
			assign_df.to_csv(out_path, index=False)
		else:
			# default to parquet
			assign_df.to_parquet(out_path, index=False)
		print(f"Wrote assignments: {out_path}")
	# Optionally stop before heavy aggregation
	if args.skip_aggregation:
		print("Skipping aggregation (--skip-aggregation set).")
		return
	# 4) Either annotate without aggregation or aggregate counts to shapes
	if args.annotate_no_aggregate:
		annotate_adata_with_shapes(args.matrix_h5, assign_df, args.agg_h5ad)
		print(f"Wrote annotated h5ad: {args.agg_h5ad}")
	else:
		aggregate_counts_to_shapes(args.matrix_h5, assign_df, args.agg_h5ad)
		print(f"Wrote aggregated h5ad: {args.agg_h5ad}")


if __name__ == "__main__":
	main()


