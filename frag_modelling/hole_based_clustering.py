import math
import pandas as pd
import numpy as np

import geopandas as gpd
import pygeoda

import shapely.wkt as wkt
from shapely.ops import unary_union

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from plotly.colors import sample_colorscale
import plotly.io as pio

pio.renderers.default = "notebook"

pd.options.display.max_columns = None

df_psd = pd.read_csv("Data/hole_psd_summary.csv")
df_voxel = pd.read_csv("Data/voxel_data.csv")

# Convert back from string to polygon
df_psd.cell_polygon = df_psd.cell_polygon.apply(wkt.loads)

pattern_identifier = "pattern_number"
hole_identifier = "hole_id"

spatial_clustering_columns_to_tune = [
    "hole_height_m",
    "hole_diameter_mm",
    "actual_stemming_m",
    "p50_mm",
    "p80_mm",
    "p95_mm",
    "p98_mm",
]

IMPACT_COLOURSCALE = "RdYlGn_r"
HOLE_LABEL_SIZE = 9


def show_metric_cards(df_metrics, cols=None, decimals=2):
    """Display a small rounded summary table for key metrics."""
    if cols is None:
        cols = df_metrics.columns.tolist()
    styled = (
        df_metrics[cols]
        .style.format(precision=decimals, na_rep="–")
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1f2937"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {"selector": "td", "props": [("padding", "8px 10px")]},
                {
                    "selector": "",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("width", "100%"),
                        ("font-size", "13px"),
                    ],
                },
            ]
        )
    )
    display(styled)


def add_polygon_trace(
    fig,
    polygon,
    value,
    cmin,
    cmax,
    colorscale=IMPACT_COLOURSCALE,
    name="Voronoi cell",
    row=None,
    col=None,
):

    if polygon.is_empty:
        return

    geometries = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)

    for geom in geometries:
        x, y = geom.exterior.xy
        norm = 0.5 if cmax == cmin else (value - cmin) / (cmax - cmin)
        colour = sample_colorscale(colorscale, [min(max(norm, 0), 1)])[0]

        plot = go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            fill="toself",
            line=dict(color="rgba(255,255,255,0.85)", width=1.1),
            fillcolor=colour,
            # hovertemplate=(
            #     f"{metric_display_name(METRIC)}: {metric_hover_value(METRIC, value)}"
            #     "<extra>Voronoi cell</extra>"
            # ),
            showlegend=False,
        )
        if row is not None and col is not None:
            fig.add_trace(plot, row=row, col=col)
        else:
            fig.add_trace(plot)


def build_display_boundary_from_cells(cell_geometries, simplify_tol=0.0):
    """
    Build the displayed pattern boundary from the union of final clipped cells.
    """
    valid_cells = [
        geom.buffer(0)
        for geom in cell_geometries
        if geom is not None and not geom.is_empty
    ]

    if not valid_cells:
        return None

    merged = unary_union(valid_cells).buffer(0)

    if simplify_tol and simplify_tol > 0:
        merged = merged.simplify(simplify_tol, preserve_topology=True)

    return merged


def add_boundary_trace(
    fig,
    boundary_geom,
    name="Pattern boundary",
    colour="black",
    width=2.4,
    row=None,
    col=None,
):
    """
    Plot a Polygon or MultiPolygon boundary cleanly.
    """
    if boundary_geom is None or boundary_geom.is_empty:
        return

    geometries = (
        [boundary_geom]
        if boundary_geom.geom_type == "Polygon"
        else list(boundary_geom.geoms)
    )

    first = True
    for geom in geometries:
        x, y = geom.exterior.xy
        plot = go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            name=name if first else name,
            line=dict(color=colour, width=width),
            hoverinfo="skip",
            showlegend=first,
            legendgroup=name,
        )
        if row is not None and col is not None:
            if col > 1:
                plot.showlegend = False
            fig.add_trace(plot, row=row, col=col)
        else:
            fig.add_trace(plot)
        first = False


def plot_cluster_holes(
    psd_df: pd.DataFrame,
    cluster_series: pd.Series,
    cluster_color_scale=IMPACT_COLOURSCALE,
    hole_label_size=HOLE_LABEL_SIZE,
):

    vmin = cluster_series.min()
    vmax = cluster_series.max()

    fig = go.Figure()

    # ---------------------------------------------------------------------
    # Voronoi cells
    # ---------------------------------------------------------------------
    for _, row in psd_df.iterrows():
        add_polygon_trace(
            fig,
            row["cell_polygon"],
            cluster_series.iloc[_],
            vmin,
            vmax,
            colorscale=cluster_color_scale,
        )

    # ---------------------------------------------------------------------
    # Pattern boundary
    # ---------------------------------------------------------------------
    display_boundary = build_display_boundary_from_cells(psd_df["cell_polygon"])

    add_boundary_trace(
        fig,
        display_boundary,
        name="Pattern boundary",
        colour="black",
        width=2.4,
    )

    # ---------------------------------------------------------------------
    # Hole collars trace
    # ---------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=psd_df["actual_x"],
            y=psd_df["actual_y"],
            mode="markers",
            name="Holes",
            marker=dict(
                size=8.5,
                color=cluster_series,
                colorscale=cluster_color_scale,
                cmin=vmin,
                cmax=vmax,
                line=dict(color="black", width=0.9),
                colorbar=dict(
                    title="Cluster",
                    thickness=18,
                    len=0.78,
                    y=0.5,
                ),
                showscale=True,
            ),
            showlegend=True,
            customdata=np.stack(
                [
                    psd_df["p50_mm"],
                    psd_df["p80_mm"],
                    psd_df["p95_mm"],
                    psd_df["effective_volume_m3"],
                    psd_df["effective_tonnes"],
                    psd_df["actual_explosive_kg"],
                    psd_df["kco_specific_charge_q"],
                    psd_df["actual_stemming_m"],
                    psd_df["hole_height_m"],
                    cluster_series,
                ],
                axis=1,
            ),
            hovertemplate=(
                "Easting: %{x:.0f}<br>"
                "Northing: %{y:.0f}<br>"
                "Hole Depth: %{customdata[8]:.1f} m<br>"
                "P50: %{customdata[0]:.0f} mm<br>"
                "P80: %{customdata[1]:.0f} mm<br>"
                "P95: %{customdata[2]:.0f} mm<br>"
                "Effective volume: %{customdata[3]:.1f} m³<br>"
                "Effective tonnes: %{customdata[4]:.1f} t<br>"
                "Charge mass: %{customdata[5]:.1f} kg<br>"
                "Actual specific charge: %{customdata[6]:.3f} kg/m³<br>"
                "Stemming used: %{customdata[7]:.2f} m<br>"
                + "<extra>Cluster: %{customdata[9]}</extra>"
            ),
        )
    )

    # ---------------------------------------------------------------------
    # Hole label trace
    # ---------------------------------------------------------------------
    label_offset = 1.5

    fig.add_trace(
        go.Scatter(
            x=psd_df["actual_x"],
            y=psd_df["actual_y"] + label_offset,
            mode="text",
            name="Hole labels",
            text=psd_df["hole_number"].astype("Int64").astype(str),
            textposition="top center",
            textfont=dict(
                size=hole_label_size,
                color="#111827",
            ),
            texttemplate="%{text}",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # ---------------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------------

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=75, b=50),
        title=dict(x=0.5),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )

    fig.update_xaxes(
        tickformat=".0f",
        showline=True,
        mirror=True,
        zeroline=False,
    )

    fig.update_yaxes(
        tickformat=".0f",
        showline=True,
        mirror=True,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    return fig.show()


CLUSTER_COUNT = 10


def remap_clusters_by_p80(
    cluster_series,
    df,
    method_name,
    metric_col="p80_mm",
    addln_aggr_columns=["p50_mm", "p95_mm", "p98_mm"],
    agg_funcs=["mean", "min", "max", "std"],
):

    agg_funcs_dict = {metric_col: agg_funcs}
    for col in addln_aggr_columns:
        agg_funcs_dict[col] = agg_funcs

    grouped_df = (
        df.groupby(pd.Series(cluster_series, name="cluster"))
        .agg(agg_funcs_dict)
        .sort_values(by=(metric_col, "mean"), ascending=True)
        .reset_index(drop=False)
    )
    if method_name:
        grouped_df.columns = [
            "_".join(col).strip() + f"_{method_name}" if col[1] != "" else col[0]
            for col in grouped_df.columns
        ]
    else:
        grouped_df.columns = [
            "_".join(col).strip() if col[1] != "" else col[0]
            for col in grouped_df.columns
        ]

    cluster_mapping_dict = (
        grouped_df.reset_index(drop=False, names=["NewCluster"])
        .set_index("cluster")
        .to_dict(orient="dict")["NewCluster"]
    )

    grouped_df["cluster"] = grouped_df["cluster"].replace(cluster_mapping_dict)

    return pd.Series(cluster_series).replace(cluster_mapping_dict), grouped_df


cols_for_clustering = spatial_clustering_columns_to_tune.copy()

# Instantiate the geodataframe from geopandas
geo_df = gpd.GeoDataFrame(
    df_psd[cols_for_clustering],
    geometry=gpd.points_from_xy(df_psd.actual_x, df_psd.actual_y),
    crs=None,
)
geo_df = pygeoda.open(geo_df)
queen_w = pygeoda.queen_weights(
    geo_df, order=1, include_lower_order=False, precision_threshold=0
)

cluster_results = pygeoda.redcap(
    CLUSTER_COUNT, queen_w, geo_df[cols_for_clustering], "fullorder-averagelinkage"
)  # Options: "firstorder-singlelinkage", "fullorder-singlelinkage", "fullorder-completelinkage", "fullorder-average-linkage", "fullorder-wardlinkage"

clusters_remastered, cluster_stats = remap_clusters_by_p80(
    cluster_results["Clusters"],
    df_psd,
    method_name=None,
    metric_col="p80_mm",
    addln_aggr_columns=["p50_mm", "p95_mm", "p98_mm"],
)

metrics_df = cluster_stats.set_index("cluster")
show_metric_cards(metrics_df)

plot_cluster_holes(df_psd, clusters_remastered, IMPACT_COLOURSCALE, HOLE_LABEL_SIZE)
