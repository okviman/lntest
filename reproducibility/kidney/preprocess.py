#!/usr/bin/env python3
"""Build the kidney arm's two inputs from the public Visium HD download.

    python -m kidney.preprocess        # from reproducibility/

Writes into kidney/data/, where umi_null, umi_de and spot_split look:

    podocytes_2um.h5ad             one row per 2um spot, with shape_id
    merged_blobs_in_cluster_5.h5ad one row per capsule, counts summed

Script form of preprocess.ipynb. Display-only cells are dropped; the
commented-out QuPath and overlay plotting blocks are kept at the bottom,
because cell 32 is the only record of how Podocytes_outline.png was drawn.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from paths import data_dir  # noqa: E402

DATA = data_dir(__file__)
DATA.mkdir(parents=True, exist_ok=True)


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def main() -> None:
    print(f"DATA = {DATA}", flush=True)

    step("[0/6] fetching inputs (~6.5 GB, skips what is already verified)")
    subprocess.run(
        [sys.executable, "-m", "fetch_data", "--arm", "kidney"],
        cwd=REPRO,
        check=True,
    )

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.optimize import minimize
    from shapely import Polygon

    step("[1/6] reading the 16um matrix and its graph clusters")
    adata_16um = sc.read_10x_h5(DATA / "binned_outputs/square_016um/filtered_feature_bc_matrix.h5")
    clusters = pd.read_csv(
        DATA / "binned_outputs/square_016um/analysis/clustering/gene_expression_graphclust/clusters.csv"
    )
    adata_16um.obs = (
        adata_16um.obs.reset_index()
        .rename(columns={"index": "Barcode"})
        .merge(clusters, on="Barcode", how="left")
    )
    adata_16um.obs = adata_16um.obs.assign(Cluster=adata_16um.obs.Cluster.astype(str))

    # cluster 5 is the glomerular capsules
    adata_16um = adata_16um[adata_16um.obs.Cluster == "5"].copy()
    print(f"cluster 5: {adata_16um.n_obs} spots x {adata_16um.n_vars} genes", flush=True)

    obs = adata_16um.obs.rename(columns={"Barcode": "barcode"})

    step("[2/6] fitting the spot grid and dissolving capsules")
    tissue_pos_parquet_path = DATA / "binned_outputs/square_016um/spatial/tissue_positions.parquet"

    def convert_obs_to_gdf(obs_orig, tissue_pos_parquet_path, initial_guess=[0, 0, 10, 0], correlation_sign=+1):
        if tissue_pos_parquet_path is not None:
            tissue_pos = pd.read_parquet(tissue_pos_parquet_path)
            obs = obs_orig.merge(tissue_pos, on="barcode")
        else:
            obs = obs_orig.copy()
        if correlation_sign * (
            abs(np.corrcoef(obs.pxl_col_in_fullres, obs.array_row)[0, 1])
            - abs(np.corrcoef(obs.pxl_col_in_fullres, obs.array_col)[0, 1])
        ) < 0:
            obs = obs.assign(grid_col=obs.array_row)
            obs = obs.assign(grid_row=obs.array_col)
        else:
            obs = obs.assign(grid_col=obs.array_col)
            obs = obs.assign(grid_row=obs.array_row)
        x = obs.grid_col
        y = obs.grid_row
        X = obs.pxl_col_in_fullres
        Y = obs.pxl_row_in_fullres
        print(np.corrcoef(obs.pxl_col_in_fullres, obs.grid_col))

        def objective(params):
            X0, Y0, c, theta = params
            X_pred = X0 + c * x * np.cos(theta) - c * y * np.sin(theta)
            Y_pred = Y0 + c * y * np.cos(theta) + c * x * np.sin(theta)
            residuals_X = X - X_pred
            residuals_Y = Y - Y_pred
            return np.sum(residuals_X**2 + residuals_Y**2)

        result = minimize(objective, initial_guess, method="BFGS")
        X0, Y0, c, theta = result.x

        print(f"X0: {X0}")
        print(f"Y0: {Y0}")
        print(f"c: {c}")
        print(f"Theta (radians): {theta}")

        def transform(x, y):
            return (
                X0 + c * x * np.cos(theta) - c * y * np.sin(theta),
                Y0 + c * y * np.cos(theta) + c * x * np.sin(theta),
            )

        def compute_polygon(row):
            v1 = transform(row["grid_col"] - 0.5, row["grid_row"] - 0.5)
            v2 = transform(row["grid_col"] + 0.5, row["grid_row"] - 0.5)
            v3 = transform(row["grid_col"] + 0.5, row["grid_row"] + 0.5)
            v4 = transform(row["grid_col"] - 0.5, row["grid_row"] + 0.5)
            return Polygon([v1, v2, v3, v4])

        obs = obs.assign(geometry=obs.apply(compute_polygon, axis=1))
        return gpd.GeoDataFrame(obs, geometry="geometry")

    gobs = convert_obs_to_gdf(obs, tissue_pos_parquet_path)

    # merge neighbouring spots into capsules
    gdf = (
        gobs.loc[:, "Cluster geometry".split()]
        .dissolve(by="Cluster")
        .reset_index()
        .explode()
        .reset_index()
        .rename(columns={"index": "ID"})
    )
    gdf = gdf.assign(ID=[i for i in range(len(gdf))])
    print(f"{len(gdf)} capsules", flush=True)

    def assign_shape_id(obs, gdf):
        """Assign each obs shape the gdf shape it shares the most area with."""
        intersections = gpd.overlay(obs, gdf, how="intersection")
        intersections["area"] = intersections.geometry.area
        max_area_idx = intersections.groupby(obs.index)["area"].idxmax()
        obs["shape_id"] = intersections.loc[max_area_idx, "ID"].values
        return obs

    def assign_shape_id_contain(obs, gdf):
        """Assign each obs shape the gdf shape that contains it."""
        if obs.crs != gdf.crs:
            gdf = gdf.to_crs(obs.crs)
        joined = gpd.sjoin(obs, gdf[["ID", "geometry"]], how="left", predicate="within")
        joined = joined.rename(columns={"ID": "shape_id"})
        joined = joined.drop(
            columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore"
        )
        return joined

    step("[3/6] reading the 2um matrix (large: several GB)")
    adata_2um = sc.read_10x_h5(DATA / "binned_outputs/square_002um/filtered_feature_bc_matrix.h5")
    obs_2um = pd.read_parquet(DATA / "binned_outputs/square_002um/spatial/tissue_positions.parquet")
    print(f"2um: {adata_2um.n_obs} spots x {adata_2um.n_vars} genes", flush=True)

    step("[4/6] keeping the 2um spots whose centre falls inside a capsule")
    obs_2um = gpd.GeoDataFrame(
        obs_2um,
        geometry=gpd.points_from_xy(obs_2um["pxl_col_in_fullres"], obs_2um["pxl_row_in_fullres"]),
        crs=None,
    )
    obs_2um_filtered = assign_shape_id_contain(obs_2um, gdf)
    obs_2um_filtered = obs_2um_filtered[~pd.isna(obs_2um_filtered.shape_id)]
    print(f"{len(obs_2um_filtered)} spots inside a capsule", flush=True)

    adata_2um = adata_2um[adata_2um.obs.index.isin(set(obs_2um_filtered.barcode))].copy()
    adata_2um.obs = (
        adata_2um.obs.reset_index()
        .rename(columns={"index": "barcode"})
        .merge(obs_2um_filtered, on="barcode", how="left")
    )
    adata_2um.obs = adata_2um.obs.drop(columns="geometry")

    out_2um = DATA / "podocytes_2um.h5ad"
    adata_2um.write(out_2um)
    print(f"wrote {out_2um}  ({adata_2um.n_obs} x {adata_2um.n_vars})", flush=True)

    step("[5/6] assigning 16um spots to capsules")
    gobs = assign_shape_id(gobs, gdf)
    adata = adata_16um.copy()
    adata.obs = (
        adata.obs.rename(columns={"Barcode": "barcode"})
        .merge(gobs, on="barcode Cluster".split(), how="left")
    )

    step("[6/6] summing counts within each capsule")

    import anndata

    def aggregate_by_shape_id(adata):
        """Sum observations sharing a shape_id into one row."""
        adata.obs["shape_id"] = adata.obs["shape_id"].astype(str)
        X_df = pd.DataFrame(adata.X.todense(), index=adata.obs.index)
        grouped_X = X_df.groupby(adata.obs["shape_id"]).sum()
        ndata = anndata.AnnData(X=grouped_X.values)
        ndata.obs = pd.DataFrame(index=grouped_X.index)
        ndata.var = adata.var
        return ndata

    ndata = aggregate_by_shape_id(adata)
    out_blobs = DATA / "merged_blobs_in_cluster_5.h5ad"
    ndata.write_h5ad(out_blobs)
    print(f"wrote {out_blobs}  ({ndata.n_obs} x {ndata.n_vars})", flush=True)

    print("\nDone.", flush=True)


# ---------------------------------------------------------------------------
# Kept from the notebook, never executed there either. write_qp_geojson_AB
# splits capsules into A/B groups for QuPath; plot_scaled_gdf_on_image drew
# Podocytes_outline.png, which has no other generator in the repository.
#
# def write_qp_geojson_AB(gdf, output_geojson_path, p=0.5, label_column="group"):
#     """Randomly assign shapes to 'A'/'B' via Binomial(n, p), write QuPath GeoJSON."""
#     if not isinstance(gdf, gpd.GeoDataFrame):
#         raise TypeError("gdf must be a GeoDataFrame")
#     if not (0.0 <= p <= 1.0):
#         raise ValueError("p must be between 0 and 1")
#     gdf_out = gdf.copy()
#     n = len(gdf_out)
#     if n == 0:
#         raise ValueError("gdf has no rows")
#     n_A = np.random.binomial(n, p)
#     labels = np.array(["A"] * n_A + ["B"] * (n - n_A))
#     perm = np.random.permutation(n)
#     gdf_out[label_column] = None
#     gdf_out.loc[gdf_out.index[perm], label_column] = labels
#     gdf_out["classification"] = gdf_out[label_column]
#     gdf_out["object_type"] = "annotation"
#     gdf_out.to_file(output_geojson_path, driver="GeoJSON")
#     return gdf_out
#
# gdf_out = write_qp_geojson_AB(tst, DATA / 'sample_capsule_2um_split.geojson')
# write_qp_geojson_AB(tst, DATA / 'sample_capsule_2um_split_0poit3.geojson', p=0.3)
# write_qp_geojson_AB(tst, DATA / 'sample_capsule_2um_split_0poit1.geojson', p=0.1)
#
# def plot_scaled_gdf_on_image(gdf, image, scale_factor, output_png_path,
#                              line_color="green", line_width=1.0):
#     """Scale gdf about the origin, overlay on image, save PNG."""
#     from matplotlib.image import imread
#     from shapely.affinity import scale as shp_scale
#     import matplotlib.pyplot as plt
#     gdf_scaled = gdf.copy()
#     gdf_scaled["geometry"] = gdf_scaled["geometry"].apply(
#         lambda geom: shp_scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
#     )
#     img = imread(image) if isinstance(image, str) else image
#     fig, ax = plt.subplots(figsize=(20, 20))
#     ax.imshow(img, origin="upper")
#     gdf_scaled.boundary.plot(ax=ax, color=line_color, linewidth=line_width)
#     ax.set_aspect("equal")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title(f"Scaled geometries (factor={scale_factor}) over image")
#     plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#
# plot_scaled_gdf_on_image(gdf=gdf, scale_factor=0.13636674,
#                          image=DATA / 'spatial/tissue_hires_image.png',
#                          output_png_path='Podocytes_outline_new.png',
#                          line_width=0.5, line_color='yellow')
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
