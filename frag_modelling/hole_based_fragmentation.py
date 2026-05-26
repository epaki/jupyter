############################################################
# 🧰 1. Imports and model bridge
############################################################
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import sample_colorscale

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.spatial import Voronoi, cKDTree
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point, MultiPoint
from ipywidgets import Output
from IPython.display import display
import plotly.io as pio
import ipywidgets as widgets
from shapely import BufferJoinStyle
from shapely.ops import unary_union

pio.renderers.default = "notebook"

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dig.btd_workbook_model import (
    WorkbookInputs,
    build_psd_from_design,
)


############################################################
# 🎨 2. Notebook styling and plotting helpers
############################################################

PLOT_WIDTH = 1100
PLOT_HEIGHT = 720
PLOT_TEMPLATE = "plotly_white"
ACCENT_COLOUR = "#D97706"
IMPACT_COLOURSCALE = "RdYlGn_r"

# Presentation toggles
SHOW_HOLE_LABELS = True
HOLE_LABEL_SIZE = 10


def apply_standard_layout(
    fig, title, x_title=None, y_title=None, height=PLOT_HEIGHT, width=PLOT_WIDTH
):
    """Apply a consistent presentation style to Plotly figures."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title={"text": title, "x": 0.5, "xanchor": "center"},
        width=width,
        height=height,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=40, t=84, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        font=dict(size=13),
    )
    if x_title is not None:
        fig.update_xaxes(
            title=x_title,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        )
    if y_title is not None:
        fig.update_yaxes(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        )
    return fig


def format_mine_plan_axes(fig, x_title="Easting", y_title="Northing"):
    """Format plan-view axes for mine coordinates with integer tick labels."""
    fig.update_xaxes(title=x_title, tickformat=",.0f", separatethousands=True)
    fig.update_yaxes(title=y_title, tickformat=",.0f", separatethousands=True)
    return fig


def marker_text_mode(show_labels=True):
    """Return a Plotly mode string with optional hole labels."""
    return "markers+text" if show_labels else "markers"


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


############################################################
# 📥 3. Load and inspect drill actuals
############################################################

df = pd.read_csv("drill_actuals_spf.csv").copy()

# Standardise the key identifiers and derive drilled height from collar and toe RLs.
df["hole_height_m"] = df["actual_collar_rl"] - df["actual_toe_rl"]
df["hole_id"] = df["hole_id"].astype(str)
df["pattern_number"] = df["pattern_number"].astype(str)

print(f"Rows loaded: {len(df):,}")
print(f"Patterns found: {df['pattern_number'].nunique():,}")
print(
    f"Hole height range: {df['hole_height_m'].min():.2f} m to {df['hole_height_m'].max():.2f} m"
)

display(df.head())


############################################################
# ✅ 4. Quick QA summary
############################################################

qa_summary = pd.DataFrame(
    [
        {
            "metric": "Hole count",
            "value": len(df),
        },
        {
            "metric": "Pattern count",
            "value": df["pattern_number"].nunique(),
        },
        {
            "metric": "Mean drilled height (m)",
            "value": df["hole_height_m"].mean(),
        },
        {
            "metric": "Min drilled height (m)",
            "value": df["hole_height_m"].min(),
        },
        {
            "metric": "Max drilled height (m)",
            "value": df["hole_height_m"].max(),
        },
    ]
)

show_metric_cards(qa_summary, cols=["metric", "value"])


############################################################
# ⚙️ 5. Define engineering defaults and modelling assumptions
############################################################

DESIGN_DEFAULTS = {
    # Drill and blast geometry
    "hole_diameter_mm": 165.0,
    "stemming_m": 3.5,
    "subdrill_m": 1.0,
    "burden_m_nominal": 7.0,
    "spacing_m_nominal": 8.0,
    # Rock mass and geomechanical assumptions
    "rmd": 45.0,
    "jps": 1.5,
    "jpa": 1.0,
    "youngs_modulus_gpa": 35.0,
    "ucs_mpa": 120.0,
    "hf_override": None,
    # Explosive assumptions
    "explosive_density_kg_m3": 1250.0,
    "explosive_rws_pct": 100.0,
    "explosive_rbs_pct": None,
    "explosive_energy_mjkg": None,
    # Material handling and density assumptions
    "rock_density_t_m3": 2.8,
    "swell_frac": 0.25,
    "downtime_frac": 0.15,
    "tooth_k": 1.0,
    "calibration_scale": 0.3384,
    # Fragmentation model controls
    "drilling_deviation_w_m": 0.0,
    "insitu_block_size_cap_mm": 2000.0,
    "blastability_prefactor": 0.06,
    "bottom_charge_fraction": 0.20,
}

pd.DataFrame.from_dict(DESIGN_DEFAULTS, orient="index", columns=["value"])


############################################################
# 🧭 6. Select pattern and create buffered boundary
############################################################


def make_pattern_boundary(df_pattern, buffer_dist):
    """Create a buffered pattern outline from actual hole coordinates."""
    pts = [Point(xy) for xy in zip(df_pattern["actual_x"], df_pattern["actual_y"])]
    hull = MultiPoint(pts).convex_hull
    boundary = hull.buffer(buffer_dist, join_style=BufferJoinStyle.mitre)
    return boundary


pattern_id = df["pattern_number"].iloc[0]
dfp = df[df["pattern_number"] == pattern_id].copy().reset_index(drop=True)

pattern_boundary = make_pattern_boundary(
    dfp,
    buffer_dist=DESIGN_DEFAULTS["spacing_m_nominal"] * 0.5,
)

print(f"Working pattern: {pattern_id}")
print(f"Holes in pattern: {len(dfp)}")
print(f"Boundary area: {pattern_boundary.area:,.1f} m²")


############################################################
# 🗺️ 7. Preview pattern footprint
############################################################

boundary_x, boundary_y = pattern_boundary.exterior.xy

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(boundary_x),
        y=list(boundary_y),
        mode="lines",
        name="Buffered pattern boundary",
        line=dict(color="black", width=2),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=dfp["actual_x"],
        y=dfp["actual_y"],
        mode=marker_text_mode(SHOW_HOLE_LABELS),
        text=dfp["hole_number"].astype("Int64").astype(str),
        textposition="top center",
        textfont=dict(size=HOLE_LABEL_SIZE, color="#1f2937"),
        name="Actual holes",
        marker=dict(size=10, color=ACCENT_COLOUR, line=dict(color="white", width=1.2)),
        hovertemplate="Hole %{text}<br>Easting=%{x:,.0f}<br>Northing=%{y:,.0f}<extra></extra>",
    )
)
apply_standard_layout(
    fig, "Selected pattern and buffered clipping boundary", "Easting", "Northing"
)
format_mine_plan_axes(fig)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()


############################################################
# 🔷 8. Build clipped Voronoi cells
############################################################


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct finite Voronoi regions in 2D from a SciPy Voronoi object."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    centre = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centre, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def build_clipped_voronoi_cells(df_pattern, clip_polygon):
    """Generate a clipped Voronoi cell for each hole and return cell area with geometry."""
    pts = df_pattern[["actual_x", "actual_y"]].to_numpy()
    vor = Voronoi(pts)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    cells = []
    for region in regions:
        poly = Polygon(vertices[region])
        clipped = poly.intersection(clip_polygon)
        cells.append(clipped)

    out = df_pattern.copy().reset_index(drop=True)
    out["cell_polygon"] = cells
    out["cell_area_m2"] = [
        geom.area if geom and not geom.is_empty else 0.0 for geom in cells
    ]
    return out


def build_display_boundary_from_cells(cell_geometries, simplify_tol=0.0):
    """
    Build the displayed pattern boundary from the union of final clipped cells.
    This ensures the black outline matches the cells actually shown.
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
    fig, boundary_geom, name="Pattern boundary", colour="black", width=2.4
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
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                name=name if first else name,
                line=dict(color=colour, width=width),
                hoverinfo="skip",
                showlegend=first,
            )
        )
        first = False


# Build clipped 2D cells that will later be converted into effective hole volumes.
df_cells = build_clipped_voronoi_cells(dfp, pattern_boundary)
display_boundary = build_display_boundary_from_cells(df_cells["cell_polygon"])

display(df_cells[["hole_number", "cell_area_m2", "hole_height_m"]].head())

############################################################
# 📦 9. Convert influence area to effective volume and tonnes
############################################################

df_cells["effective_volume_m3"] = df_cells["cell_area_m2"] * df_cells["hole_height_m"]
df_cells["effective_tonnes"] = (
    df_cells["effective_volume_m3"] * DESIGN_DEFAULTS["rock_density_t_m3"]
)

volume_summary = pd.DataFrame(
    [
        {
            "metric": "Mean cell area (m²)",
            "value": df_cells["cell_area_m2"].mean(),
        },
        {
            "metric": "Mean effective volume (m³)",
            "value": df_cells["effective_volume_m3"].mean(),
        },
        {
            "metric": "Mean effective tonnes (t)",
            "value": df_cells["effective_tonnes"].mean(),
        },
        {
            "metric": "Total effective tonnes (t)",
            "value": df_cells["effective_tonnes"].sum(),
        },
    ]
)

show_metric_cards(volume_summary, cols=["metric", "value"])


############################################################
# 📐 10. Estimate local burden and spacing
############################################################


def estimate_local_geometry_row_aware(df_pattern, burden_nominal, spacing_nominal):
    """
    Estimate burden and spacing using pattern orientation and neighbour projection.
    This replaces the area-based approximation with a physically meaningful method.
    """

    pts = df_pattern[["actual_x", "actual_y"]].to_numpy()

    # Identify pattern orientation via PCA
    pca = PCA(n_components=2).fit(pts)
    spacing_dir = pca.components_[0]
    burden_dir = np.array([-spacing_dir[1], spacing_dir[0]])

    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=8)

    spacing_vals = []
    burden_vals = []

    for i in range(len(df_pattern)):
        neighbours = pts[idxs[i][1:]]
        vectors = neighbours - pts[i]

        spacing_proj = np.abs(vectors @ spacing_dir)
        burden_proj = np.abs(vectors @ burden_dir)

        spacing_candidates = spacing_proj[burden_proj < burden_nominal * 0.5]
        burden_candidates = burden_proj[spacing_proj < spacing_nominal * 0.5]

        spacing_vals.append(
            np.median(spacing_candidates)
            if len(spacing_candidates)
            else spacing_nominal
        )
        burden_vals.append(
            np.median(burden_candidates) if len(burden_candidates) else burden_nominal
        )

    df_out = df_pattern.copy()
    df_out["spacing_local_m"] = np.clip(
        spacing_vals, 0.5 * spacing_nominal, 1.5 * spacing_nominal
    )
    df_out["burden_local_m"] = np.clip(
        burden_vals, 0.5 * burden_nominal, 1.5 * burden_nominal
    )

    return df_out


df_cells = estimate_local_geometry_row_aware(
    df_cells,
    burden_nominal=DESIGN_DEFAULTS["burden_m_nominal"],
    spacing_nominal=DESIGN_DEFAULTS["spacing_m_nominal"],
)

display(df_cells[["hole_number", "burden_local_m", "spacing_local_m"]].head())

############################################################
# 📊 11. Review local geometry variation
############################################################

burden = df_cells["burden_local_m"].dropna()
spacing = df_cells["spacing_local_m"].dropna()

# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------
b_mean, b_med = burden.mean(), burden.median()
s_mean, s_med = spacing.mean(), spacing.median()

xmin = min(burden.min(), spacing.min())
xmax = max(burden.max(), spacing.max())
xpad = (xmax - xmin) * 0.05

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=burden,
        name="Local burden",
        nbinsx=18,
        opacity=0.58,
        marker=dict(
            color="rgba(79, 70, 229, 0.75)",
            line=dict(color="white", width=0.8),
        ),
        hovertemplate=(
            "<b>Local burden</b><br>"
            "Range: %{x}<br>"
            "Hole count: %{y}"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Histogram(
        x=spacing,
        name="Local spacing",
        nbinsx=18,
        opacity=0.50,
        marker=dict(
            color="rgba(234, 88, 12, 0.72)",
            line=dict(color="white", width=0.8),
        ),
        hovertemplate=(
            "<b>Local spacing</b><br>"
            "Range: %{x}<br>"
            "Hole count: %{y}"
            "<extra></extra>"
        ),
    )
)

# ---------------------------------------------------------------------
# Mean / median reference lines
# ---------------------------------------------------------------------
for xval, label, colour, dash in [
    (b_mean, "Burden mean", "rgba(79, 70, 229, 0.95)", "solid"),
    (b_med, "Burden median", "rgba(79, 70, 229, 0.95)", "dot"),
    (s_mean, "Spacing mean", "rgba(234, 88, 12, 0.95)", "solid"),
    (s_med, "Spacing median", "rgba(234, 88, 12, 0.95)", "dot"),
]:
    fig.add_vline(
        x=xval,
        line=dict(color=colour, width=2, dash=dash),
        opacity=0.95,
    )

# ---------------------------------------------------------------------
# Summary annotation
# ---------------------------------------------------------------------
fig.add_annotation(
    x=0.99,
    y=0.98,
    xref="paper",
    yref="paper",
    xanchor="right",
    yanchor="top",
    align="left",
    text=(
        "<b>Summary statistics</b><br>"
        f"Burden mean: {b_mean:.2f} m<br>"
        f"Burden median: {b_med:.2f} m<br>"
        f"Spacing mean: {s_mean:.2f} m<br>"
        f"Spacing median: {s_med:.2f} m"
    ),
    showarrow=False,
    bordercolor="rgba(0,0,0,0.10)",
    borderwidth=1,
    borderpad=8,
    bgcolor="rgba(255,255,255,0.88)",
    font=dict(size=12, color="#1f2937"),
)

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
apply_standard_layout(
    fig,
    "Distribution of Estimated Local Burden and Spacing",
    "Distance (m)",
    "Hole count",
    height=560,
)

fig.update_layout(
    barmode="overlay",
    width=950,
    title=dict(
        text="<b>Distribution of Estimated Local Burden and Spacing</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=22),
    ),
    font=dict(size=14, color="#1f2937"),
    margin=dict(l=70, r=40, t=80, b=65),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(0,0,0,0.08)",
        borderwidth=1,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    bargap=0.06,
)

fig.update_xaxes(
    range=[xmin - xpad, xmax + xpad],
    showline=True,
    linewidth=1.2,
    linecolor="black",
    ticks="outside",
    ticklen=6,
    tickwidth=1,
    gridcolor="rgba(180,180,180,0.22)",
    zeroline=False,
    title_font=dict(size=16),
)

fig.update_yaxes(
    showline=True,
    linewidth=1.2,
    linecolor="black",
    ticks="outside",
    ticklen=6,
    tickwidth=1,
    gridcolor="rgba(180,180,180,0.22)",
    zeroline=False,
    title_font=dict(size=16),
)

fig.show()

fig = go.Figure()

fig.add_trace(
    go.Violin(
        x=["Local burden"] * len(burden),
        y=burden,
        name="Local burden",
        box_visible=True,
        meanline_visible=True,
        line_color="rgba(79, 70, 229, 1.0)",
        fillcolor="rgba(79, 70, 229, 0.35)",
        opacity=0.85,
    )
)

fig.add_trace(
    go.Violin(
        x=["Local spacing"] * len(spacing),
        y=spacing,
        name="Local spacing",
        box_visible=True,
        meanline_visible=True,
        line_color="rgba(234, 88, 12, 1.0)",
        fillcolor="rgba(234, 88, 12, 0.35)",
        opacity=0.85,
    )
)

apply_standard_layout(
    fig,
    "Distribution of Estimated Local Burden and Spacing",
    "Distance (m)",
    "Hole count",
    height=560,
)
fig.show()

############################################################
# 🧠 12. Run hole-level PSD modelling
############################################################

CFG = {
    "fallback_a": 1.0,
}


def run_psd_for_hole(row, defaults, cfg):
    """Run the workbook PSD model for a single hole-level input set."""
    inp = WorkbookInputs(
        burden_m=float(row["burden_local_m"]),
        spacing_m=float(row["spacing_local_m"]),
        hole_diameter_mm=float(defaults["hole_diameter_mm"]),
        stemming_m=float(defaults["stemming_m"]),
        subdrill_m=float(defaults["subdrill_m"]),
        bench_height_m=float(row["hole_height_m"]),
        rmd=float(defaults["rmd"]),
        jps=float(defaults["jps"]),
        jpa=float(defaults["jpa"]),
        youngs_modulus_gpa=defaults["youngs_modulus_gpa"],
        ucs_mpa=defaults["ucs_mpa"],
        hf_override=defaults["hf_override"],
        explosive_density_kg_m3=float(defaults["explosive_density_kg_m3"]),
        explosive_rws_pct=float(defaults["explosive_rws_pct"]),
        explosive_rbs_pct=defaults["explosive_rbs_pct"],
        explosive_energy_mjkg=defaults["explosive_energy_mjkg"],
        rock_density_t_m3=float(defaults["rock_density_t_m3"]),
        swell_frac=float(defaults["swell_frac"]),
        downtime_frac=float(defaults["downtime_frac"]),
        tooth_k=float(defaults["tooth_k"]),
        calibration_scale=float(defaults["calibration_scale"]),
        drilling_deviation_w_m=float(defaults["drilling_deviation_w_m"]),
        insitu_block_size_cap_mm=float(defaults["insitu_block_size_cap_mm"]),
        blastability_prefactor=float(defaults["blastability_prefactor"]),
        bottom_charge_fraction=float(defaults["bottom_charge_fraction"]),
    )

    psd = build_psd_from_design(inp, cfg)
    return {
        "x50_mm": psd.x50_mm,
        "xmax_mm": psd.xmax_mm,
        "p50_mm": psd.p50_mm,
        "p80_mm": psd.p80_mm,
        "p95_mm": psd.p95_mm,
        "p98_mm": psd.p98_mm,
        "sample_x_mm": psd.sample_x_mm,
        "sample_y_pct": psd.sample_y_pct,
        "kco_specific_charge_q": psd.details.get("kco_specific_charge_q"),
        "kco_uniformity_n": psd.details.get("kco_uniformity_n"),
        "kco_b": psd.details.get("kco_b"),
    }


psd_rows = [
    run_psd_for_hole(row, DESIGN_DEFAULTS, CFG) for _, row in df_cells.iterrows()
]

df_psd = pd.concat([df_cells.reset_index(drop=True), pd.DataFrame(psd_rows)], axis=1)

display(
    df_psd[["hole_number", "p50_mm", "p80_mm", "p95_mm", "effective_tonnes"]].head()
)


############################################################
# 🧱 13. Build 3D blast volume voxel grid
############################################################

# ---------------------------------------------------------------------
# Build XY candidate grid first
# ---------------------------------------------------------------------
dx = DESIGN_DEFAULTS["spacing_m_nominal"] / 4
dy = DESIGN_DEFAULTS["burden_m_nominal"] / 4
dz = np.mean(dfp["hole_height_m"]) / 6

minx, miny, maxx, maxy = pattern_boundary.bounds

xs = np.arange(minx, maxx + dx, dx)
ys = np.arange(miny, maxy + dy, dy)

xy_candidates = np.array([(x, y) for x in xs for y in ys], dtype=float)

# ---------------------------------------------------------------------
# Keep only XY points inside the pattern boundary
# ---------------------------------------------------------------------
inside_mask = np.array(
    [pattern_boundary.contains(Point(x, y)) for x, y in xy_candidates]
)
xy_inside = xy_candidates[inside_mask]

print(f"XY candidates inside boundary: {len(xy_inside):,}")

# ---------------------------------------------------------------------
# Interpolate top and bottom surfaces in one batch
# ---------------------------------------------------------------------
points_xy = dfp[["actual_x", "actual_y"]].to_numpy()
top_z_vals = dfp["actual_collar_rl"].to_numpy()
bot_z_vals = dfp["actual_toe_rl"].to_numpy()

interp_top = griddata(points_xy, top_z_vals, xy_inside, method="linear")
interp_bot = griddata(points_xy, bot_z_vals, xy_inside, method="linear")

# ---------------------------------------------------------------------
# Build valid vertical columns
# ---------------------------------------------------------------------
valid_xy = []
valid_top = []
valid_bot = []

for (x, y), z_top, z_bot in zip(xy_inside, interp_top, interp_bot):
    if np.isnan(z_top) or np.isnan(z_bot):
        continue
    if z_top <= z_bot:
        continue

    height = z_top - z_bot
    if height < dz * 0.5:
        continue

    valid_xy.append((x, y))
    valid_top.append(z_top)
    valid_bot.append(z_bot)

valid_xy = np.array(valid_xy, dtype=float)
valid_top = np.array(valid_top, dtype=float)
valid_bot = np.array(valid_bot, dtype=float)

print(f"Valid XY columns: {len(valid_xy):,}")

# ---------------------------------------------------------------------
# Expand valid columns into voxels
# ---------------------------------------------------------------------
voxel_centres = []

for (x, y), z_top, z_bot in zip(valid_xy, valid_top, valid_bot):
    z_vals = np.arange(z_bot, z_top, dz)
    if len(z_vals) == 0:
        continue
    voxel_centres.extend([[x, y, z] for z in z_vals])

voxel_centres = np.array(voxel_centres, dtype=float)

print(f"Voxel count: {len(voxel_centres):,}")


############################################################
# 🧮 14. Assign voxels to nearest drillholes
############################################################


def point_to_segment_distance(points, a, b):
    ab = b - a
    ab2 = np.dot(ab, ab)

    if ab2 == 0:
        return np.linalg.norm(points - a, axis=1)

    ap = points - a
    t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab

    return np.linalg.norm(points - proj, axis=1)


# ---------------------------------------------------------------------
# Build hole segments and XY search tree
# ---------------------------------------------------------------------
hole_ids = []
hole_collars = []
hole_toes = []

for _, row in dfp.iterrows():
    hole_ids.append(str(row["hole_number"]))
    hole_collars.append(
        np.array(
            [row["actual_x"], row["actual_y"], row["actual_collar_rl"]], dtype=float
        )
    )
    hole_toes.append(
        np.array([row["actual_x"], row["actual_y"], row["actual_toe_rl"]], dtype=float)
    )

hole_collars = np.array(hole_collars, dtype=float)
hole_toes = np.array(hole_toes, dtype=float)
hole_xy = hole_collars[:, :2]

hole_tree = cKDTree(hole_xy)

# ---------------------------------------------------------------------
# Candidate shortlist per voxel from XY only
# ---------------------------------------------------------------------
k_candidates = min(8, len(hole_ids))
_, candidate_idx = hole_tree.query(voxel_centres[:, :2], k=k_candidates)

if k_candidates == 1:
    candidate_idx = candidate_idx[:, None]

# ---------------------------------------------------------------------
# Exact segment-distance assignment using only shortlist candidates
# ---------------------------------------------------------------------
assigned_index = np.empty(len(voxel_centres), dtype=int)

for i, voxel in enumerate(voxel_centres):
    candidates = candidate_idx[i]
    point_arr = voxel[None, :]

    best_idx = None
    best_dist = np.inf

    for cand in np.atleast_1d(candidates):
        d = point_to_segment_distance(point_arr, hole_collars[cand], hole_toes[cand])[0]
        if d < best_dist:
            best_dist = d
            best_idx = cand

    assigned_index[i] = best_idx

assigned_holes = [hole_ids[i] for i in assigned_index]

print(f"Assigned voxels: {len(assigned_holes):,}")


############################################################
# 📦 15. Compare 2D and 3D hole ownership volumes
############################################################

# ---------------------------------------------------------------------
# Build volumes
# ---------------------------------------------------------------------
voxel_volume = dx * dy * dz

volume_series = (
    pd.Series(assigned_holes, name="hole_number").value_counts().sort_index()
    * voxel_volume
)

df_psd["hole_number_str"] = df_psd["hole_number"].astype(str)
df_psd["voxel_volume_m3"] = df_psd["hole_number_str"].map(volume_series).fillna(0.0)
df_psd["voxel_tonnes"] = (
    df_psd["voxel_volume_m3"] * DESIGN_DEFAULTS["rock_density_t_m3"]
)
df_psd["volume_ratio_3d_to_2d"] = np.where(
    df_psd["effective_volume_m3"] > 0,
    df_psd["voxel_volume_m3"] / df_psd["effective_volume_m3"],
    np.nan,
)

plot_df = df_psd.copy()

# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------
x = plot_df["effective_volume_m3"]
y = plot_df["voxel_volume_m3"]

valid = plot_df[["effective_volume_m3", "voxel_volume_m3"]].dropna()
r2 = r2_score(valid["voxel_volume_m3"], valid["effective_volume_m3"])
corr = valid["effective_volume_m3"].corr(valid["voxel_volume_m3"])
mean_ratio = plot_df["volume_ratio_3d_to_2d"].mean()
median_ratio = plot_df["volume_ratio_3d_to_2d"].median()

# Axis range for 1:1 line
xy_min = min(x.min(), y.min())
xy_max = max(x.max(), y.max())
pad = (xy_max - xy_min) * 0.06
axis_min = max(0, xy_min - pad)
axis_max = xy_max + pad

# ---------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------
fig = px.scatter(
    plot_df,
    x="effective_volume_m3",
    y="voxel_volume_m3",
    trendline="ols",
    color="volume_ratio_3d_to_2d",
    color_continuous_scale="RdYlGn",
    labels={
        "effective_volume_m3": "2D Volume (m³)",
        "voxel_volume_m3": "3D Volume (m³)",
        "volume_ratio_3d_to_2d": "3D / 2D Ratio",
    },
    hover_data={
        "hole_number": True,
        "effective_volume_m3": ":.1f",
        "voxel_volume_m3": ":.1f",
        "voxel_tonnes": ":.1f",
        "volume_ratio_3d_to_2d": ":.2f",
    },
    title="2D vs 3D Volume Comparison",
)

# Make markers look cleaner
fig.update_traces(
    selector=dict(mode="markers"),
    marker=dict(
        size=8,
        line=dict(width=0.8, color="white"),
        opacity=0.88,
    ),
)

# Make trendline cleaner
fig.update_traces(
    selector=dict(mode="lines"),
    line=dict(width=2.5, dash="solid"),
    opacity=0.85,
)

# ---------------------------------------------------------------------
# Add 1:1 reference line
# ---------------------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="1:1 line",
        line=dict(color="rgba(60,60,60,0.85)", width=2, dash="dash"),
        hoverinfo="skip",
    )
)

# ---------------------------------------------------------------------
# Annotation box
# ---------------------------------------------------------------------
fig.add_annotation(
    x=0.02,
    y=0.98,
    xref="paper",
    yref="paper",
    xanchor="left",
    yanchor="top",
    align="left",
    text=(
        f"<b>Summary</b><br>"
        f"Points: {len(valid):,}<br>"
        f"Correlation: {corr:.2f}<br>"
        f"R² vs 1:1: {r2:.2f}<br>"
        f"Mean 3D/2D ratio: {mean_ratio:.2f}<br>"
        f"Median 3D/2D ratio: {median_ratio:.2f}"
    ),
    showarrow=False,
    bordercolor="rgba(80,80,80,0.25)",
    borderwidth=1,
    borderpad=8,
    bgcolor="rgba(255,255,255,0.88)",
    font=dict(size=12),
)

# ---------------------------------------------------------------------
# Layout polish
# ---------------------------------------------------------------------
fig.update_layout(
    template="plotly_white",
    width=980,
    height=650,
    title=dict(
        text="<b>2D vs 3D Volume Comparison</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=22),
    ),
    font=dict(size=14),
    margin=dict(l=70, r=40, t=80, b=70),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(255,255,255,0.7)",
    ),
    coloraxis_colorbar=dict(
        title="3D / 2D",
        thickness=16,
        len=0.75,
    ),
)

fig.update_xaxes(
    title_text="2D Volume (m³)",
    range=[axis_min, axis_max],
    showline=True,
    linewidth=1.2,
    linecolor="black",
    mirror=False,
    ticks="outside",
    gridcolor="rgba(180,180,180,0.25)",
    zeroline=False,
)

fig.update_yaxes(
    title_text="3D Volume (m³)",
    range=[axis_min, axis_max],
    showline=True,
    linewidth=1.2,
    linecolor="black",
    mirror=False,
    ticks="outside",
    gridcolor="rgba(180,180,180,0.25)",
    zeroline=False,
    scaleanchor=None,
)

fig.show()


############################################################
# 🧾 16. Prepare voxel dataset
############################################################

voxel_df = pd.DataFrame(voxel_centres, columns=["x", "y", "z"])
voxel_df["assigned_hole"] = assigned_holes
voxel_df["voxel_volume_m3"] = voxel_volume
voxel_df["assigned_hole"] = voxel_df["assigned_hole"].astype(str)

print(f"Voxel rows: {len(voxel_df):,}")
display(voxel_df.head())

############################################################
# 🎨 17. Build hole colour mapping
############################################################

hole_ids = dfp["hole_number"].astype(str).tolist()
hole_ids_sorted = sorted(
    hole_ids, key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else str(x)
)

palette = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
)
hole_colour_map = {
    hole_id: palette[i % len(palette)] for i, hole_id in enumerate(hole_ids_sorted)
}

hole_metric_lookup = df_psd.assign(
    hole_number_str=df_psd["hole_number"].astype(str)
).set_index("hole_number_str")

display(
    pd.DataFrame(
        {
            "hole_id": list(hole_colour_map.keys())[:10],
            "colour": list(hole_colour_map.values())[:10],
        }
    )
)

############################################################
# 🔍 18. Sample voxels for display
############################################################


def sample_voxels_for_display(voxel_df, max_points=30000, random_state=42):
    if len(voxel_df) <= max_points:
        return voxel_df.copy()
    return voxel_df.sample(n=max_points, random_state=random_state).copy()


VOXEL_DISPLAY_MAX = 30000
voxel_plot_df = sample_voxels_for_display(voxel_df, max_points=VOXEL_DISPLAY_MAX)

print(f"Displaying {len(voxel_plot_df):,} of {len(voxel_df):,} voxels")

############################################################
# 🛠️ 19. Shared 3D plotting helpers
############################################################


def build_selected_hole_3d_view(
    voxel_df_in, holes_df_in, selected_holes, hole_colour_map
):
    subset_vox = voxel_df_in[voxel_df_in["assigned_hole"].isin(selected_holes)].copy()
    subset_holes = holes_df_in[
        holes_df_in["hole_number"].astype(str).isin(selected_holes)
    ].copy()

    fig = go.Figure()

    for hole_id, sub in subset_vox.groupby("assigned_hole", sort=False):
        colour = hole_colour_map.get(str(hole_id), "#636EFA")
        fig.add_trace(
            go.Scatter3d(
                x=sub["x"],
                y=sub["y"],
                z=sub["z"],
                mode="markers",
                marker=dict(size=3, color=colour, opacity=0.6),
                name=f"Hole {hole_id}",
                showlegend=True,
            )
        )

    for _, row in subset_holes.iterrows():
        hole_id = str(row["hole_number"])
        colour = hole_colour_map.get(hole_id, "#111111")
        fig.add_trace(
            go.Scatter3d(
                x=[row["actual_x"], row["actual_x"]],
                y=[row["actual_y"], row["actual_y"]],
                z=[row["actual_collar_rl"], row["actual_toe_rl"]],
                mode="lines+markers",
                line=dict(color=colour, width=8),
                marker=dict(size=4, color=colour),
                name=f"Trace {hole_id}",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Selected-hole 3D ownership view",
        width=1100,
        height=850,
        scene=dict(
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="RL",
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


############################################################
# 🧊 20. Visualise 3D voxel ownership by hole
############################################################

plot_df = voxel_plot_df.copy()
plot_df["assigned_hole_str"] = plot_df["assigned_hole"].astype(str)
plot_df["p80_mm"] = plot_df["assigned_hole_str"].map(hole_metric_lookup["p80_mm"])
plot_df["voxel_volume_m3"] = plot_df["assigned_hole_str"].map(
    hole_metric_lookup["voxel_volume_m3"]
    if "voxel_volume_m3" in hole_metric_lookup.columns
    else {}
)
plot_df["colour"] = plot_df["assigned_hole_str"].map(
    lambda h: hole_colour_map.get(h, "#636EFA")
)

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=plot_df["x"],
        y=plot_df["y"],
        z=plot_df["z"],
        mode="markers",
        marker=dict(
            size=2.0,
            color=plot_df["colour"],
            opacity=0.72,
        ),
        customdata=np.column_stack(
            [
                plot_df["assigned_hole_str"],
                plot_df["p80_mm"],
                plot_df["voxel_volume_m3"],
            ]
        ),
        hovertemplate=(
            "<b>Assigned hole %{customdata[0]}</b><br>"
            "Easting: %{x:.0f} m<br>"
            "Northing: %{y:.0f} m<br>"
            "RL: %{z:.1f} m<br>"
            "Hole P80: %{customdata[1]:.1f} mm<br>"
            "Hole 3D volume: %{customdata[2]:.1f} m³"
            "<extra></extra>"
        ),
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter3d(
        x=dfp["actual_x"],
        y=dfp["actual_y"],
        z=np.full(len(dfp), plot_df["z"].max() + dz),
        mode="markers",
        name="Hole collars",
        marker=dict(
            size=4.5,
            color="black",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "<b>Hole collar</b><br>"
            "Easting: %{x:.0f} m<br>"
            "Northing: %{y:.0f} m"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title=dict(
        text="<b>3D Blast Volume Ownership</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=24),
    ),
    width=1180,
    height=860,
    margin=dict(l=10, r=10, t=70, b=10),
    paper_bgcolor="white",
    scene=dict(
        xaxis=dict(
            title="Easting", tickformat=".0f", backgroundcolor="rgb(245,247,250)"
        ),
        yaxis=dict(
            title="Northing", tickformat=".0f", backgroundcolor="rgb(245,247,250)"
        ),
        zaxis=dict(title="RL", tickformat=".1f", backgroundcolor="rgb(250,250,252)"),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=0.82)),
    ),
)

fig.show()


############################################################
# 🎯 21. Inspect selected holes in 3D
############################################################


def get_neighbouring_holes(
    df_holes: pd.DataFrame,
    seed_hole: str,
    n_holes: int = 5,
) -> list[str]:
    """
    Return the seed hole plus its nearest neighbouring holes in plan view.
    """
    holes = df_holes.copy()
    holes["hole_number_str"] = holes["hole_number"].astype(str)

    if seed_hole not in holes["hole_number_str"].values:
        raise ValueError(f"Seed hole {seed_hole} not found.")

    seed_row = holes.loc[holes["hole_number_str"] == seed_hole].iloc[0]

    holes["dist_to_seed"] = np.sqrt(
        (holes["actual_x"] - seed_row["actual_x"]) ** 2
        + (holes["actual_y"] - seed_row["actual_y"]) ** 2
    )

    selected = (
        holes.sort_values(["dist_to_seed", "hole_number_str"])
        .head(n_holes)["hole_number_str"]
        .tolist()
    )

    return selected


def make_selected_hole_colour_map(selected_holes: list[str]) -> dict[str, str]:
    palette = [
        "#1f4e79",
        "#4f81bd",
        "#76a5af",
        "#93c47d",
        "#b7b7b7",
        "#c27ba0",
        "#674ea7",
        "#a64d79",
    ]
    return {hole: palette[i % len(palette)] for i, hole in enumerate(selected_holes)}


hole_options = sorted(
    dfp["hole_number"].astype("Int64").astype(str).unique(), key=lambda x: int(x)
)

seed_hole_widget = widgets.Dropdown(
    options=hole_options,
    value=hole_options[0],
    description="Seed hole:",
    layout=widgets.Layout(width="220px"),
)

n_holes_widget = widgets.IntSlider(
    value=5,
    min=2,
    max=min(10, len(hole_options)),
    step=1,
    description="Cluster size:",
    continuous_update=False,
    layout=widgets.Layout(width="320px"),
)

show_labels_widget = widgets.Checkbox(
    value=False,
    description="Show hole labels",
)

output = widgets.Output()


def render_selected_hole_cluster(
    seed_hole: str, n_holes: int, show_labels: bool = False
):
    selected_holes = get_neighbouring_holes(dfp, seed_hole=seed_hole, n_holes=n_holes)
    selected_colour_map = make_selected_hole_colour_map(selected_holes)

    fig = build_selected_hole_3d_view(
        voxel_plot_df,
        dfp,
        selected_holes,
        selected_colour_map,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>3D View of Selected Neighbouring Holes</b><br><sup>Seed hole: {seed_hole} | Selected holes: {', '.join(selected_holes)}</sup>",
            x=0.5,
            xanchor="center",
        ),
        width=1150,
        height=850,
        paper_bgcolor="white",
        font=dict(size=14, color="#1f2937"),
    )

    fig.update_scenes(
        xaxis=dict(title="Easting", tickformat=".0f"),
        yaxis=dict(title="Northing", tickformat=".0f"),
        zaxis=dict(title="RL", tickformat=".1f"),
    )

    fig.show()


def update_plot(_=None):
    with output:
        output.clear_output(wait=True)
        render_selected_hole_cluster(
            seed_hole=seed_hole_widget.value,
            n_holes=n_holes_widget.value,
            show_labels=show_labels_widget.value,
        )


seed_hole_widget.observe(update_plot, names="value")
n_holes_widget.observe(update_plot, names="value")
show_labels_widget.observe(update_plot, names="value")

controls = widgets.HBox([seed_hole_widget, n_holes_widget, show_labels_widget])

display(controls, output)
update_plot()


def estimate_nominal_spacing(df_holes: pd.DataFrame) -> float:
    holes = df_holes.copy()
    coords = holes[["actual_x", "actual_y"]].to_numpy()

    dists = []
    for i in range(len(coords)):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = dist[dist > 0]
        if len(dist) > 0:
            dists.append(dist.min())

    return float(np.median(dists))


def get_neighbouring_holes_with_threshold(
    df_holes: pd.DataFrame,
    seed_hole: str,
    n_holes: int = 5,
    max_spacing_factor: float = 1.6,
) -> list[str]:
    holes = df_holes.copy()
    holes["hole_number_str"] = holes["hole_number"].astype(str)

    seed_row = holes.loc[holes["hole_number_str"] == seed_hole].iloc[0]
    nominal_spacing = estimate_nominal_spacing(holes)
    max_dist = nominal_spacing * max_spacing_factor

    holes["dist_to_seed"] = np.sqrt(
        (holes["actual_x"] - seed_row["actual_x"]) ** 2
        + (holes["actual_y"] - seed_row["actual_y"]) ** 2
    )

    candidate_holes = holes.loc[holes["dist_to_seed"] <= max_dist].copy()

    if len(candidate_holes) < n_holes:
        candidate_holes = holes.copy()

    selected = (
        candidate_holes.sort_values(["dist_to_seed", "hole_number_str"])
        .head(n_holes)["hole_number_str"]
        .tolist()
    )
    return selected


def build_selected_hole_3d_view_with_context(
    voxel_plot_df,
    dfp,
    selected_holes,
    selected_colour_map,
):
    fig = go.Figure()

    selected_set = set(map(str, selected_holes))
    plot_df = voxel_plot_df.copy()
    plot_df["assigned_hole_str"] = plot_df["assigned_hole"].astype(str)

    bg = plot_df.loc[~plot_df["assigned_hole_str"].isin(selected_set)]
    fg = plot_df.loc[plot_df["assigned_hole_str"].isin(selected_set)]

    # Background context
    fig.add_trace(
        go.Scatter3d(
            x=bg["x"],
            y=bg["y"],
            z=bg["z"],
            mode="markers",
            marker=dict(
                size=1.6,
                color="rgba(160,160,160,0.18)",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Selected foreground by hole
    for hole_id in selected_holes:
        sub = fg.loc[fg["assigned_hole_str"] == str(hole_id)]
        fig.add_trace(
            go.Scatter3d(
                x=sub["x"],
                y=sub["y"],
                z=sub["z"],
                mode="markers",
                name=f"Hole {hole_id}",
                marker=dict(
                    size=2.8,
                    color=selected_colour_map[str(hole_id)],
                    opacity=0.82,
                ),
                hovertemplate=(
                    f"<b>Hole {hole_id}</b><br>"
                    "Easting: %{x:.0f} m<br>"
                    "Northing: %{y:.0f} m<br>"
                    "RL: %{z:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    # Collars for selected holes
    collars = dfp.copy()
    collars["hole_number_str"] = collars["hole_number"].astype(str)
    collars = collars.loc[collars["hole_number_str"].isin(selected_set)]

    collar_z = plot_df["z"].max() + 1.0

    fig.add_trace(
        go.Scatter3d(
            x=collars["actual_x"],
            y=collars["actual_y"],
            z=np.full(len(collars), collar_z),
            mode="markers+text",
            text=collars["hole_number_str"],
            textposition="top center",
            marker=dict(
                size=5,
                color="black",
                line=dict(color="white", width=1),
            ),
            textfont=dict(size=10, color="black"),
            name="Selected collars",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Selected Neighbouring Hole Ownership in 3D</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=24),
        ),
        width=1150,
        height=860,
        paper_bgcolor="white",
        font=dict(size=14, color="#1f2937"),
        margin=dict(l=10, r=10, t=70, b=10),
        scene=dict(
            xaxis=dict(title="Easting", tickformat=".0f"),
            yaxis=dict(title="Northing", tickformat=".0f"),
            zaxis=dict(title="RL", tickformat=".1f"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.45, z=0.85)),
        ),
    )

    return fig


############################################################
# 🌈 22. Visualise 3D blast volume by elevation
############################################################

plot_df = voxel_plot_df.copy()

fig = px.scatter_3d(
    plot_df,
    x="x",
    y="y",
    z="z",
    color="z",
    color_continuous_scale="Earth",
    opacity=0.65,
)

# ---------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------
fig.update_traces(
    marker=dict(
        size=2.2,
        opacity=0.7,
    ),
    hovertemplate=(
        "<b>Voxel</b><br>"
        "Easting: %{x:.0f} m<br>"
        "Northing: %{y:.0f} m<br>"
        "RL: %{z:.1f} m"
        "<extra></extra>"
    ),
)

fig.add_trace(
    go.Scatter3d(
        x=dfp["actual_x"],
        y=dfp["actual_y"],
        z=[plot_df["z"].max()] * len(dfp),
        mode="markers",
        marker=dict(size=4, color="black"),
        name="Hole collars",
    )
)

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
fig.update_layout(
    title=dict(
        text="<b>3D Blast Volume Coloured by Elevation</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=24),
    ),
    width=1180,
    height=860,
    margin=dict(l=10, r=10, t=70, b=10),
    paper_bgcolor="white",
    font=dict(size=14, color="#1f2937"),
)

# ---------------------------------------------------------------------
# Scene styling
# ---------------------------------------------------------------------
fig.update_scenes(
    xaxis=dict(
        title="Easting",
        tickformat=".0f",
        showbackground=True,
        backgroundcolor="rgb(245,247,250)",
        gridcolor="rgba(150,150,150,0.25)",
        zeroline=False,
    ),
    yaxis=dict(
        title="Northing",
        tickformat=".0f",
        showbackground=True,
        backgroundcolor="rgb(245,247,250)",
        gridcolor="rgba(150,150,150,0.25)",
        zeroline=False,
    ),
    zaxis=dict(
        title="RL",
        tickformat=".1f",
        showbackground=True,
        backgroundcolor="rgb(250,250,252)",
        gridcolor="rgba(150,150,150,0.20)",
        zeroline=False,
    ),
    aspectmode="data",
    aspectratio=dict(x=1, y=1, z=0.6),
    camera=dict(eye=dict(x=1.45, y=1.55, z=0.85)),
)

# ---------------------------------------------------------------------
# Colourbar
# ---------------------------------------------------------------------
fig.update_layout(
    coloraxis_colorbar=dict(
        title="RL (m)",
        thickness=16,
        len=0.75,
        y=0.5,
        yanchor="middle",
        outlinewidth=0,
        tickformat=".1f",
    )
)

zmin = plot_df["z"].quantile(0.02)
zmax = plot_df["z"].quantile(0.98)

fig.update_traces(marker=dict(cmin=zmin, cmax=zmax))

fig.show()

############################################################
# 📈 23. Visualise 3D blast volume by P80
############################################################

plot_df = voxel_plot_df.copy()
plot_df["assigned_hole_str"] = plot_df["assigned_hole"].astype(str)

plot_df["p80_mm"] = plot_df["assigned_hole_str"].map(
    hole_metric_lookup["p80_mm"].to_dict()
)

# Optional: clip colour range slightly to reduce the influence of extreme outliers
p80_min = plot_df["p80_mm"].quantile(0.02)
p80_max = plot_df["p80_mm"].quantile(0.98)

fig = px.scatter_3d(
    plot_df,
    x="x",
    y="y",
    z="z",
    color="p80_mm",
    color_continuous_scale=IMPACT_COLOURSCALE,
    opacity=0.68,
    range_color=[p80_min, p80_max],
    custom_data=["assigned_hole_str", "p80_mm"],
)

fig.update_traces(
    marker=dict(
        size=2.2,
        opacity=0.72,
    ),
    hovertemplate=(
        "<b>Assigned hole %{customdata[0]}</b><br>"
        "Easting: %{x:.0f} m<br>"
        "Northing: %{y:.0f} m<br>"
        "RL: %{z:.1f} m<br>"
        "Hole P80: %{customdata[1]:.0f} mm"
        "<extra></extra>"
    ),
)

fig.add_trace(
    go.Scatter3d(
        x=dfp["actual_x"],
        y=dfp["actual_y"],
        z=np.full(len(dfp), plot_df["z"].max() + dz),
        mode="markers",
        name="Hole collars",
        marker=dict(
            size=4.5,
            color="black",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "<b>Hole collar</b><br>"
            "Easting: %{x:.0f} m<br>"
            "Northing: %{y:.0f} m"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title=dict(
        text="<b>3D Blast Volume Coloured by Assigned-Hole P80</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=24),
    ),
    width=1180,
    height=860,
    margin=dict(l=10, r=10, t=70, b=10),
    paper_bgcolor="white",
    font=dict(size=14, color="#1f2937"),
    coloraxis_colorbar=dict(
        title="P80 (mm)",
        thickness=16,
        len=0.75,
        y=0.5,
        yanchor="middle",
        outlinewidth=0,
        tickformat=".0f",
    ),
    scene=dict(
        xaxis=dict(
            title="Easting",
            tickformat=".0f",
            showbackground=True,
            backgroundcolor="rgb(245,247,250)",
            gridcolor="rgba(150,150,150,0.25)",
            zeroline=False,
            showspikes=False,
        ),
        yaxis=dict(
            title="Northing",
            tickformat=".0f",
            showbackground=True,
            backgroundcolor="rgb(245,247,250)",
            gridcolor="rgba(150,150,150,0.25)",
            zeroline=False,
            showspikes=False,
        ),
        zaxis=dict(
            title="RL",
            tickformat=".1f",
            showbackground=True,
            backgroundcolor="rgb(250,250,252)",
            gridcolor="rgba(150,150,150,0.20)",
            zeroline=False,
            showspikes=False,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=0.85)),
    ),
)

fig.show()

############################################################
# 📋 24. Summarise hole-level fragmentation results
############################################################

frag_summary = pd.DataFrame(
    [
        {"metric": "Mean P50 (mm)", "value": df_psd["p50_mm"].mean()},
        {"metric": "Mean P80 (mm)", "value": df_psd["p80_mm"].mean()},
        {"metric": "Mean P95 (mm)", "value": df_psd["p95_mm"].mean()},
        {"metric": "P80 min (mm)", "value": df_psd["p80_mm"].min()},
        {"metric": "P80 max (mm)", "value": df_psd["p80_mm"].max()},
        {"metric": "P80 std dev (mm)", "value": df_psd["p80_mm"].std()},
    ]
)

show_metric_cards(frag_summary, cols=["metric", "value"])


############################################################
# 🧪 25. Build diagnostic metrics
############################################################

nominal_area = (
    DESIGN_DEFAULTS["burden_m_nominal"] * DESIGN_DEFAULTS["spacing_m_nominal"]
)

df_psd["nominal_volume_m3"] = nominal_area * df_psd["hole_height_m"]

df_psd["volume_ratio"] = df_psd["effective_volume_m3"] / df_psd["nominal_volume_m3"]

# Proxy: lower volume → higher PF
df_psd["pf_proxy"] = 1.0 / df_psd["volume_ratio"]

baseline_p80 = df_psd["p80_mm"].median()

df_psd["delta_p80"] = df_psd["p80_mm"] - baseline_p80
df_psd["delta_p80_pct"] = df_psd["delta_p80"] / baseline_p80 * 100

display(df_psd[["hole_number", "volume_ratio", "pf_proxy", "delta_p80"]].head())


############################################################
# ⚖️ 26. Compare 2D-weighted vs 3D-weighted PSD
############################################################


def aggregate_psd_with_weight(df_subset, weight_col):
    size = np.array(df_subset.iloc[0]["sample_x_mm"], dtype=float)
    total_w = df_subset[weight_col].sum()

    agg = np.zeros_like(size, dtype=float)

    for _, row in df_subset.iterrows():
        agg += row[weight_col] * (np.array(row["sample_y_pct"], dtype=float) / 100.0)

    return size, agg / max(total_w, 1e-9)


s2d, y2d = aggregate_psd_with_weight(df_psd, "effective_tonnes")
s3d, y3d = aggregate_psd_with_weight(df_psd, "voxel_tonnes")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=s2d, y=y2d * 100, mode="lines", name="2D-weighted PSD", line=dict(width=3)
    )
)
fig.add_trace(
    go.Scatter(
        x=s3d,
        y=y3d * 100,
        mode="lines",
        name="3D-weighted PSD",
        line=dict(width=3, dash="dash"),
    )
)

apply_standard_layout(
    fig,
    "2D vs 3D Weighted PSD",
    "Particle Size (mm)",
    "Passing (%)",
    height=560,
    width=980,
)
fig.update_xaxes(type="log")
fig.update_yaxes(range=[0, 100], ticksuffix="%")
fig.show()

############################################################
# 🚩 27. Flag high-impact ownership shifts
############################################################

df_psd["volume_shift_m3"] = df_psd["voxel_volume_m3"] - df_psd["effective_volume_m3"]
df_psd["volume_shift_pct"] = df_psd["volume_shift_m3"] / df_psd[
    "effective_volume_m3"
].replace(0, np.nan)

high_shift = df_psd.reindex(
    df_psd["volume_shift_pct"].abs().sort_values(ascending=False).index
).head(15)

display(
    high_shift[
        [
            "hole_number",
            "effective_volume_m3",
            "voxel_volume_m3",
            "volume_shift_m3",
            "volume_shift_pct",
            "p80_mm",
        ]
    ]
)


############################################################
# 🗺️ 28. Create Voronoi impact map
############################################################

df_psd["volume_ratio_3d"] = df_psd["voxel_volume_m3"] / df_psd["effective_volume_m3"]

METRIC = "volume_ratio_3d"
# METRIC = "p80_mm"
# Options:
# "p80_mm"
# "delta_p80"
# "volume_ratio"
# "pf_proxy"


def metric_display_name(metric):
    mapping = {
        "p80_mm": "P80",
        "delta_p80": "ΔP80",
        "volume_ratio": "Volume Ratio",
        "volume_ratio_3d": "Volume Ratio (3D)",
        "pf_proxy": "Powder Factor Proxy",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def metric_unit(metric):
    mapping = {
        "p80_mm": "mm",
        "delta_p80": "mm",
        "volume_ratio": "",
        "volume_ratio_3d": "",
        "pf_proxy": "",
    }
    return mapping.get(metric, "")


def metric_colourbar_title(metric):
    unit = metric_unit(metric)
    return f"{metric_display_name(metric)}{f' ({unit})' if unit else ''}"


def metric_hover_value(metric, value):
    unit = metric_unit(metric)
    if unit:
        return f"{value:.1f} {unit}"
    return f"{value:.2f}"


def add_polygon_trace(
    fig, polygon, value, cmin, cmax, colorscale=IMPACT_COLOURSCALE, name="Voronoi cell"
):
    if polygon.is_empty:
        return

    geometries = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)

    for geom in geometries:
        x, y = geom.exterior.xy
        norm = 0.5 if cmax == cmin else (value - cmin) / (cmax - cmin)
        colour = sample_colorscale(colorscale, [min(max(norm, 0), 1)])[0]

        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself",
                line=dict(color="rgba(255,255,255,0.85)", width=1.1),
                fillcolor=colour,
                hovertemplate=(
                    f"{metric_display_name(METRIC)}: {metric_hover_value(METRIC, value)}"
                    "<extra>Voronoi cell</extra>"
                ),
                showlegend=False,
            )
        )


vmin = df_psd[METRIC].min()
vmax = df_psd[METRIC].max()

fig = go.Figure()

# ---------------------------------------------------------------------
# Voronoi cells
# ---------------------------------------------------------------------
for _, row in df_psd.iterrows():
    add_polygon_trace(
        fig,
        row["cell_polygon"],
        row[METRIC],
        vmin,
        vmax,
        colorscale=IMPACT_COLOURSCALE,
    )

# ---------------------------------------------------------------------
# Pattern boundary
# ---------------------------------------------------------------------
display_boundary = build_display_boundary_from_cells(df_psd["cell_polygon"])

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
        x=df_psd["actual_x"],
        y=df_psd["actual_y"],
        mode="markers",
        name="Holes",
        marker=dict(
            size=8.5,
            color=df_psd[METRIC],
            colorscale=IMPACT_COLOURSCALE,
            cmin=vmin,
            cmax=vmax,
            line=dict(color="black", width=0.9),
            colorbar=dict(
                title=metric_colourbar_title(METRIC),
                thickness=18,
                len=0.78,
                y=0.5,
            ),
            showscale=True,
        ),
        customdata=np.stack(
            [
                df_psd["hole_number"].astype("Int64").astype(str),
                df_psd["p50_mm"],
                df_psd["p80_mm"],
                df_psd["p95_mm"],
                df_psd["effective_volume_m3"],
                df_psd["effective_tonnes"],
                df_psd[METRIC],
            ],
            axis=1,
        ),
        hovertemplate=(
            "Hole %{customdata[0]}<br>"
            "Easting: %{x:.0f}<br>"
            "Northing: %{y:.0f}<br>"
            "P50: %{customdata[1]:.1f} mm<br>"
            "P80: %{customdata[2]:.1f} mm<br>"
            "P95: %{customdata[3]:.1f} mm<br>"
            "Effective volume: %{customdata[4]:.1f} m³<br>"
            "Effective tonnes: %{customdata[5]:.1f} t<br>"
            f"{metric_display_name(METRIC)}: "
            + "%{customdata[6]:.2f}"
            + (f" {metric_unit(METRIC)}" if metric_unit(METRIC) else "")
            + "<extra></extra>"
        ),
        showlegend=True,
    )
)

# ---------------------------------------------------------------------
# Hole label trace
# ---------------------------------------------------------------------
label_offset = 1.5

fig.add_trace(
    go.Scatter(
        x=df_psd["actual_x"],
        y=df_psd["actual_y"] + label_offset,
        mode="text",
        name="Hole labels",
        text=df_psd["hole_number"].astype("Int64").astype(str),
        textposition="top center",
        textfont=dict(
            size=HOLE_LABEL_SIZE,
            color="#111827",
        ),
        texttemplate="%{text}",
        hoverinfo="skip",
        showlegend=True,
        visible=True if SHOW_HOLE_LABELS else "legendonly",
    )
)

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
apply_standard_layout(
    fig,
    f"Voronoi-Based Hole-Level Impact Map | {metric_display_name(METRIC)}",
    "Easting",
    "Northing",
    height=820,
)

format_mine_plan_axes(fig)

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

fig.show()

############################################################
# 📍 30. Create reference scatter view
############################################################

fig = px.scatter(
    df_psd,
    x="actual_x",
    y="actual_y",
    color=METRIC,
    size="effective_tonnes",
    text="hole_number",
    color_continuous_scale=IMPACT_COLOURSCALE,
    custom_data=[
        "hole_number",
        "p50_mm",
        "p80_mm",
        "p95_mm",
        "effective_volume_m3",
        "effective_tonnes",
    ],
)

fig.update_traces(
    mode=marker_text_mode(SHOW_HOLE_LABELS),
    text=df_psd["hole_number"].astype("Int64").astype(str),
    textposition="top center",
    textfont=dict(
        size=HOLE_LABEL_SIZE,
        color="#1f2937",
    ),
    marker=dict(
        line=dict(color="white", width=1.2),
        sizemode="area",
        opacity=0.92,
    ),
    hovertemplate=(
        "Hole %{customdata[0]}<br>"
        "Easting: %{x:.0f}<br>"
        "Northing: %{y:.0f}<br>"
        "P50: %{customdata[1]:.0f} mm<br>"
        "P80: %{customdata[2]:.0f} mm<br>"
        "P95: %{customdata[3]:.0f} mm<br>"
        "Effective volume: %{customdata[4]:.1f} m³<br>"
        "Effective tonnes: %{customdata[5]:.1f} t"
        "<extra></extra>"
    ),
)

apply_standard_layout(
    fig,
    f"Hole-Level Predicted {METRIC.replace('_', ' ').title()}",
    "Easting",
    "Northing",
)

format_mine_plan_axes(fig)

fig.update_layout(
    height=720,
    width=980,
    margin=dict(l=40, r=40, t=70, b=50),
    title=dict(x=0.5),
)

fig.update_coloraxes(
    colorbar_title=f"{METRIC.replace('_', ' ').title()} (mm)",
    colorbar=dict(
        thickness=18,
        len=0.78,
        y=0.5,
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

fig.show()


############################################################
# 🧮 31. Interactive PSD aggregation and comparison
############################################################
# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------


def prepare_voxel_plot_df_for_slicing(df, hole_metric_lookup, metric_col="p80_mm"):
    """
    Return a copy of the voxel dataframe with the requested hole-level metric
    mapped onto each voxel row via assigned_hole.

    This makes the viewer cell robust to notebook execution order.
    """
    out = df.copy()

    if "assigned_hole" not in out.columns:
        raise ValueError("voxel dataframe must contain an 'assigned_hole' column.")

    lookup = hole_metric_lookup.copy()
    lookup.index = lookup.index.astype(str)

    if metric_col not in lookup.columns:
        raise ValueError(
            f"'{metric_col}' was not found in hole_metric_lookup. "
            f"Available columns: {list(lookup.columns)}"
        )

    out["assigned_hole_str"] = out["assigned_hole"].astype(str)
    out[metric_col] = out["assigned_hole_str"].map(lookup[metric_col])

    missing_metric = out[metric_col].isna().sum()
    if missing_metric > 0:
        print(
            f"Warning: {missing_metric} voxel rows could not be mapped to '{metric_col}'. "
            "Those rows will still plot, but with missing colour values."
        )

    return out


# Build a viewer-specific dataframe
voxel_slice_df = prepare_voxel_plot_df_for_slicing(
    voxel_plot_df,
    hole_metric_lookup,
    metric_col="p80_mm",
)

# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------


def _fmt_coord(value):
    """Format Easting/Northing/RL as whole number with no commas."""
    return f"{int(round(value))}"


def _format_interval_label(vmin, vmax, axis_name):
    return f"{axis_name}: {_fmt_coord(vmin)} to {_fmt_coord(vmax)}"


def _build_slice_ranges(values, mode="step", step=None, thickness=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return []

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    if mode == "unique":
        uniq = np.sort(np.unique(vals))
        if len(uniq) == 1:
            return [(uniq[0], uniq[0])]
        return [(float(uniq[i]), float(uniq[i + 1])) for i in range(len(uniq) - 1)]

    if step is None:
        uniq = np.sort(np.unique(np.round(vals, 6)))
        diffs = np.diff(uniq)
        diffs = diffs[diffs > 0]
        step = float(np.median(diffs)) if len(diffs) else float(vmax - vmin)

    if thickness is None:
        thickness = step

    if step <= 0:
        step = vmax - vmin if vmax > vmin else 1.0
    if thickness <= 0:
        thickness = step

    starts = np.arange(vmin, vmax + 1e-9, step)
    ranges = [(float(s), float(min(s + thickness, vmax))) for s in starts]
    return [(a, b) for (a, b) in ranges if b > a] or [(vmin, vmax)]


def _make_dropdown_from_ranges(ranges, axis_name, description):
    options = [(_format_interval_label(a, b, axis_name), (a, b)) for a, b in ranges]
    return widgets.Dropdown(
        options=options,
        description=description,
        layout=widgets.Layout(width="420px"),
        style={"description_width": "initial"},
    )


# ---------------------------------------------------------------------
# Common plot styling
# ---------------------------------------------------------------------


def _common_scatter_layout(fig, x_title, y_title, colour_title="P80 (mm)"):
    fig.update_traces(
        marker=dict(size=8, opacity=0.9, line=dict(width=0)),
        hovertemplate=(
            "Easting: %{customdata[0]:.0f}<br>"
            "Northing: %{customdata[1]:.0f}<br>"
            "RL: %{customdata[2]:.0f}<br>"
            "P80: %{marker.color:.1f} mm"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=40, r=40, t=70, b=50),
        title=dict(x=0.5),
        coloraxis_colorbar=dict(
            title=colour_title,
            len=0.8,
            thickness=18,
        ),
    )

    fig.update_xaxes(
        title=x_title,
        showgrid=True,
        zeroline=False,
        showline=True,
        mirror=True,
        tickformat=".0f",
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        zeroline=False,
        showline=True,
        mirror=True,
        tickformat=".0f",
    )

    return fig


# ---------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------


def build_horizontal_slice_figure(df, z_min, z_max, colour_col="p80_mm"):
    subset = df[(df["z"] >= z_min) & (df["z"] <= z_max)].copy()

    if subset.empty:
        return (
            None,
            f"No data found for RL slice {_fmt_coord(z_min)} to {_fmt_coord(z_max)}",
        )

    if colour_col not in subset.columns:
        return None, f"Column '{colour_col}' not found in dataframe."

    fig = px.scatter(
        subset,
        x="x",
        y="y",
        color=colour_col,
        color_continuous_scale=IMPACT_COLOURSCALE,
        title=f"Horizontal Slice | RL {_fmt_coord(z_min)} to {_fmt_coord(z_max)}",
        custom_data=["x", "y", "z"],
    )

    _common_scatter_layout(fig, x_title="Easting", y_title="Northing")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, None


def build_vertical_section_figure(
    df, section_axis, section_min, section_max, colour_col="p80_mm"
):
    subset = df[
        (df[section_axis] >= section_min) & (df[section_axis] <= section_max)
    ].copy()

    if subset.empty:
        axis_label = "Easting" if section_axis == "x" else "Northing"
        return (
            None,
            f"No data found for {axis_label} section {_fmt_coord(section_min)} to {_fmt_coord(section_max)}",
        )

    if colour_col not in subset.columns:
        return None, f"Column '{colour_col}' not found in dataframe."

    if section_axis == "x":
        plot_x = "y"
        section_name = "Easting"
        section_value_label = f"{_fmt_coord(section_min)} to {_fmt_coord(section_max)}"
        x_title = "Northing"
    else:
        plot_x = "x"
        section_name = "Northing"
        section_value_label = f"{_fmt_coord(section_min)} to {_fmt_coord(section_max)}"
        x_title = "Easting"

    fig = px.scatter(
        subset,
        x=plot_x,
        y="z",
        color=colour_col,
        color_continuous_scale=IMPACT_COLOURSCALE,
        title=f"Vertical Section | {section_name} {section_value_label}",
        custom_data=["x", "y", "z"],
    )

    _common_scatter_layout(fig, x_title=x_title, y_title="RL")

    return fig, None


# ---------------------------------------------------------------------
# Horizontal viewer
# ---------------------------------------------------------------------


def interactive_horizontal_slice_viewer(
    df, z_step=None, z_thickness=None, mode="step", colour_col="p80_mm"
):
    z_ranges = _build_slice_ranges(
        df["z"].values,
        mode=mode,
        step=z_step,
        thickness=z_thickness,
    )

    if not z_ranges:
        print("No valid RL ranges could be derived from the dataframe.")
        return

    dropdown = _make_dropdown_from_ranges(
        z_ranges,
        axis_name="RL",
        description="Horizontal slice:",
    )

    out = widgets.Output()

    def _update(change=None):
        if dropdown.value is None:
            return

        z_min, z_max = dropdown.value
        fig, msg = build_horizontal_slice_figure(
            df, z_min, z_max, colour_col=colour_col
        )

        out.clear_output(wait=True)
        with out:
            if msg is not None:
                print(msg)
            else:
                display(fig)

    dropdown.observe(_update, names="value")

    controls = widgets.HBox([dropdown])
    display(widgets.VBox([controls, out]))

    _update()


# ---------------------------------------------------------------------
# Vertical viewer
# ---------------------------------------------------------------------


def interactive_vertical_section_viewer(
    df,
    x_step=None,
    x_thickness=None,
    y_step=None,
    y_thickness=None,
    mode="step",
    colour_col="p80_mm",
):
    x_ranges = _build_slice_ranges(
        df["x"].values,
        mode=mode,
        step=x_step,
        thickness=x_thickness,
    )

    y_ranges = _build_slice_ranges(
        df["y"].values,
        mode=mode,
        step=y_step,
        thickness=y_thickness,
    )

    if not x_ranges and not y_ranges:
        print("No valid section ranges could be derived from the dataframe.")
        return

    direction_dropdown = widgets.Dropdown(
        options=[
            ("Easting section", "x"),
            ("Northing section", "y"),
        ],
        value="x",
        description="Section type:",
        layout=widgets.Layout(width="260px"),
        style={"description_width": "initial"},
    )

    range_dropdown = widgets.Dropdown(
        description="Section range:",
        layout=widgets.Layout(width="420px"),
        style={"description_width": "initial"},
    )

    out = widgets.Output()
    _state = {"suspend": False}

    def _update_range_options(change=None):
        section_axis = direction_dropdown.value
        ranges = x_ranges if section_axis == "x" else y_ranges
        axis_name = "Easting" if section_axis == "x" else "Northing"

        _state["suspend"] = True
        range_dropdown.options = [
            (_format_interval_label(a, b, axis_name), (a, b)) for a, b in ranges
        ]
        range_dropdown.value = ranges[0] if ranges else None
        _state["suspend"] = False

        _update_plot()

    def _update_plot(change=None):
        if _state["suspend"]:
            return
        if range_dropdown.value is None:
            return

        section_axis = direction_dropdown.value
        section_min, section_max = range_dropdown.value

        fig, msg = build_vertical_section_figure(
            df,
            section_axis,
            section_min,
            section_max,
            colour_col=colour_col,
        )

        out.clear_output(wait=True)
        with out:
            if msg is not None:
                print(msg)
            else:
                display(fig)

    direction_dropdown.observe(_update_range_options, names="value")
    range_dropdown.observe(_update_plot, names="value")

    controls = widgets.HBox([direction_dropdown, range_dropdown])
    display(widgets.VBox([controls, out]))

    _update_range_options()


# ---------------------------------------------------------------------
# Launch interactive slice viewers
# ---------------------------------------------------------------------

interactive_horizontal_slice_viewer(
    voxel_slice_df,
    z_step=5,
    z_thickness=5,
    mode="step",
    colour_col="p80_mm",
)

interactive_vertical_section_viewer(
    voxel_slice_df,
    x_step=10,
    x_thickness=10,
    y_step=10,
    y_thickness=10,
    mode="step",
    colour_col="p80_mm",
)
