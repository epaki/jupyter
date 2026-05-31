import os
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime
from pykrige.ok import OrdinaryKriging

import matplotlib.pyplot as plt
import pyvista as pv

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

ROOT_PATH = os.path.dirname(os.getcwd())
ROOT_DATA_PATH = os.path.join(ROOT_PATH, "data/")
DATA_PATH = os.path.join(ROOT_DATA_PATH, "Ravensworth_20260504/")
INPUT_PATH = os.path.join(DATA_PATH, "output/")
OUTPUT_PATH = INPUT_PATH

# Coal seam prediction parameters
input_cols = ['Hole_ID', 'EAST', 'NORTH', 'RelDepth', 'ROP', 'DEN(LS)', 'DEN(SS)', 'CoalPresence']
training_cols = ['RelDepth', 'ROP', 'DEN(LS)', 'DEN(SS)']
scoring_col = 'CoalPresence'
predicted_col = 'CoalPrediction'
probability_col = 'CoalPredictProbability'

# Seam merging parameters
GAP_THRESHOLD = 0.3

# Minimum percentage of hole population for a seam to be retained in kriging input dataset
MIN_POPULATION_PCT = 5
GRID_RESOLUTION = 100


# TODO - Change this to a manual bound based on the mine design
# xmin, xmax = pick_df.east.min(), pick_df.east.max()
# ymin, ymax = pick_df.north.min(), pick_df.north.max()

pred_seam_df = pd.read_csv(os.path.join(INPUT_PATH, 'predicted_seams.csv'))

pred_seam_df = pred_seam_df[['Hole_ID', 'EAST', 'NORTH', 'TopDepth', 'BottomDepth', 'Confidence']]
pred_seam_df = pred_seam_df.rename(columns={'Hole_ID': 'HoleID', 'Assigned_Seam': 'SeamID'})
pred_seam_df.sort_values(by=['TopDepth'], ascending=False, inplace=True, ignore_index=True)

pred_seam_df.sort_values(['HoleID', 'TopDepth'], ascending=[True, False], inplace=True)
pred_seam_df['GapToNextSeam'] = pred_seam_df['BottomDepth'].groupby(pred_seam_df['HoleID']).shift(1) - pred_seam_df['TopDepth']
pred_seam_df['NewGroup'] = (pred_seam_df['GapToNextSeam'] > GAP_THRESHOLD) | (pred_seam_df['GapToNextSeam'].isna())
pred_seam_df['SeamCluster'] = pred_seam_df.groupby('HoleID')['NewGroup'].cumsum()
pred_seam_df = pred_seam_df.groupby(['HoleID', 'SeamCluster']).agg({'TopDepth': 'max', 'BottomDepth': 'min', 'Confidence': 'mean', 'EAST': 'first', 'NORTH': 'first'}).reset_index()
pred_seam_df.drop(columns=['SeamCluster'], inplace=True)

pred_seam_df.head()

ads = pd.read_csv(os.path.join(INPUT_PATH, "master_adf.csv"))

# Dropping Nulls from DEN & ROP since they are the most critical columns especially since the number of coal seams from the missing Densities & ROPs are None
ads = ads.dropna(subset=['ROP', 'DEN(LS)', 'DEN(SS)'])[input_cols].reset_index(drop=True)

pick_df = pd.read_csv(os.path.join(DATA_PATH, "2020_current_rvn_stm_dhdb_picks_stu.csv"))
collar_df = pd.read_csv(os.path.join(DATA_PATH, "2020_current_rvn_stm_dhdb_collar_stu.csv"))
pick_df = pick_df.set_index('holeid').join(collar_df.set_index('holeid'), how='inner')
pick_df['from'] = pick_df['rl'] - pick_df['from']
pick_df['to'] = pick_df['rl'] - pick_df['to']

raw_pick_df = pick_df.copy()

pick_df['ply'] = pick_df['ply'].apply(lambda x: x[:-1] if x.endswith('U') else (x[:-1] if x.endswith('L') else x))
pick_df = pick_df.groupby(['holeid', 'ply']).agg({'from': 'max', 'to': 'min', 'east': 'first', 'north': 'first', 'rl': 'first', 'depth': 'first'}).reset_index()

# pick_df.reset_index(inplace=True)
pick_df.head()

# Step 1: Identify adjacent seams in the same hole and calculate the gap between them
pick_df = pick_df.sort_values(['holeid', 'from'], ascending=[True, False])
pick_df['seam_idx'] = pick_df.groupby('holeid').cumcount() + 1

next_pick_df = pick_df[['holeid', 'seam_idx', 'from', 'to', 'ply']].copy()

merged = pick_df.merge(next_pick_df, on='holeid', suffixes=('_curr', '_next'))
merged['gap'] = merged['to_curr'] - merged['from_next']
merged = merged[merged['seam_idx_curr'] == merged['seam_idx_next'] - 1]

# Step 2: Filter pairs of seams based on the gap threshold and minimum population percentage
merged_gaps = merged.sort_values(['ply_curr', 'ply_next']).groupby(['ply_curr', 'ply_next']).agg({'gap': 'mean', 'rl': 'count'}).reset_index()
filtered_merge = merged_gaps[(merged_gaps['gap'] < GAP_THRESHOLD) & (merged_gaps['rl'] >= MIN_POPULATION_PCT / 100 * merged['holeid'].nunique())]

# Step 3: Create a mapping of ply names that are likely the same seam based on the filtered pairs
unique_plys = {x: False for x in set(filtered_merge['ply_curr'].unique()).union(set(filtered_merge['ply_next'].unique()))}
rename_dict = {}
unique_counter = 1

for _, row in filtered_merge.iterrows():
    if unique_plys.get(row['ply_curr']):
        # Find where the current ply is being renamed to and rename the next ply to that name as well
        rename_dict[row['ply_next']] = rename_dict.get(row['ply_curr'])
        unique_plys[row['ply_next']] = True
    elif unique_plys.get(row['ply_next']):
        # Find where the next ply is being renamed to and rename the current ply to that name as well
        rename_dict[row['ply_curr']] = rename_dict.get(row['ply_next'])
        unique_plys[row['ply_curr']] = True
    else:
        rename_dict[row['ply_next']] = unique_counter
        rename_dict[row['ply_curr']] = unique_counter
        unique_plys[row['ply_next']] = True
        unique_plys[row['ply_curr']] = True
        unique_counter += 1

# Step 4: Generate the final mapping dictionary to rename the ply names in pick_df to the new unified seam names based on the rename_dict
rename_mapping = {}
for key, value in sorted(rename_dict.items()):
    if value in rename_mapping.keys():
        rename_mapping[value] += "," + str(key)
    else:
        rename_mapping[value] = str(key)    

rename_mapping = {key: rename_mapping[value] for key, value in sorted(rename_dict.items())}

# Step 5: Change the ply names in pick_df to the new unified seam names based on the rename_mapping
pick_df.ply = pick_df.ply.apply(lambda x: rename_mapping.get(x, x))

old_pick_df_len = len(pick_df)
pick_df = pick_df.groupby(['holeid', 'ply']).agg({'from': 'max', 'to': 'min', 'east': 'first', 'north': 'first', 'rl': 'first', 'depth': 'first'}).reset_index()
print(f"Total reduction in size due to seam clubbing: {len(pick_df) / old_pick_df_len :.2%}")

rop_df = pd.DataFrame()

for dir in Path(DATA_PATH).iterdir():
    if 'penrate' in dir.name.lower():
        penrate_df = pd.read_csv(dir)
        penrate_df['Hole_ID'] = penrate_df['Pattern'].apply(lambda x: x.replace('RN', 'BH').replace('_', '').replace('LOGHOLES', '').replace('BL', 'B').split('-')[0]) + penrate_df['HoleName']
        penrate_df.rename(columns={"PenetrationRate":'ROP', 'Pattern':'Pattern_ID', 'DrillName':'Drill'}, inplace=True)
        penrate_df.dropna(subset=['Depth'], inplace=True)
        rop_df = pd.concat([rop_df, penrate_df], ignore_index=True)
        
rop_df = rop_df[['Hole_ID', 'Depth', 'ROP']]

rop_df = rop_df.set_index('Hole_ID').join(collar_df[['holeid', 'rl']].set_index('holeid'), how='inner').reset_index()
rop_df['RelDepth'] = rop_df['rl'] - rop_df['Depth']
rop_df.drop(columns=['rl', 'Depth'], inplace=True)
rop_df = rop_df.sort_values(['Hole_ID', 'RelDepth']).reset_index(drop=True)
rop_df.head()

def show_metric_cards(df_metrics, cols=None, decimals=2):
	"""Display a small rounded summary table for key metrics."""
	if cols is None:
		cols = df_metrics.columns.tolist()
	styled = (
		df_metrics[cols]
		.style
		.format(precision=decimals, na_rep="–")
		.set_properties(**{"text-align": "center"})
		.set_table_styles([
			{"selector": "th", "props": [("background-color", "#1f2937"), ("color", "white"), ("text-align", "center")]},
			{"selector": "td", "props": [("padding", "8px 10px")]},
			{"selector": "", "props": [("border-collapse", "collapse"), ("width", "100%"), ("font-size", "13px")]},
		])
	)
	display(styled)
     

def _create_pyviz_plotter(title="PyVista Plotter", size=(1400, 700)):
    """Create a PyVista plotter with a dark theme and customized settings."""
    # pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=size)
    plotter.add_axes(interactive=True)
    plotter.set_background("white")
    plotter.view_isometric()
    plotter.add_title(title, font_size=14, color='black')
    return plotter

def _define_color_dict(categories, colormap=plt.cm.viridis):
    """Define a color dictionary for given categories using a specified colormap."""
    colors = colormap(np.linspace(0, 1, len(categories)))

    return {category: colors[i] for i, category in enumerate(categories)}


def _add_plane_and_points_to_plotter(
        plotter: pv.Plotter, seam_df: pd.DataFrame, top_plane: OrdinaryKriging, depth_plane: OrdinaryKriging, 
        x_bounds: tuple[float, float], y_bounds: tuple[float, float], grid_resolution: int, color: str='blue'
        ) -> pv.Plotter:
    '''
    Add kriging planes for top and bottom seam surfaces to the PyVista plotter.
    '''
    
    x = seam_df['east'].values
    y = seam_df['north'].values
    top = seam_df['from'].values
    depth = seam_df['from'].values - seam_df['to'].values

    top_points = pv.PolyData(np.array([x, y, top]).T)
    bottom_points = pv.PolyData(np.array([x, y, top - depth]).T)

    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, grid_resolution), np.linspace(ymin, ymax, grid_resolution))
    x_flat, y_flat = grid_x.ravel(), grid_y.ravel()

    z_top, z_depth = top_plane.execute('points', x_flat, y_flat)[0].data, depth_plane.execute('points', x_flat, y_flat)[0].data
    z_top, z_depth = z_top.reshape(grid_x.shape), z_depth.reshape(grid_x.shape)
    z_bottom = z_top - z_depth
    
    top_surface = pv.StructuredGrid(grid_x, grid_y, z_top)
    bottom_surface = pv.StructuredGrid(grid_x, grid_y, z_bottom)

    plotter.add_mesh(top_points, color=color, point_size=8, render_points_as_spheres=True, label='Top Picks')
    plotter.add_mesh(bottom_points, color=color, point_size=8, render_points_as_spheres=True, label='Bottom Picks')

    plotter.add_mesh(top_surface, color=color, opacity=0.7, show_edges=True, label='Top Surface', show_scalar_bar=False)
    plotter.add_mesh(bottom_surface, color=color, opacity=0.4, show_edges=True, label='Bottom Surface', show_scalar_bar=False)


def get_full_plotter_with_kriged_seams(seam_planes, x_bounds, y_bounds, grid_resolution, display=False):
    '''
    Similar to _add_plane_and_points_to_plotter but for entire seam planes at once and without the points of the seam
    '''
    plotter = _create_pyviz_plotter(title="Kriged Seam Surfaces")
    colors = _define_color_dict(seam_planes.keys())
    
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds

    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, grid_resolution), np.linspace(ymin, ymax, grid_resolution))
    x_flat, y_flat = grid_x.ravel(), grid_y.ravel()

    for seam in seam_planes.keys():
        top_plane, depth_plane = seam_planes[seam][0], seam_planes[seam][1]
        z_top, z_depth = top_plane.execute('points', x_flat, y_flat)[0].data, depth_plane.execute('points', x_flat, y_flat)[0].data
        z_top, z_depth = z_top.reshape(grid_x.shape), z_depth.reshape(grid_x.shape)
        z_bottom = z_top - z_depth

        top_surface = pv.StructuredGrid(grid_x, grid_y, z_top)
        bottom_surface = pv.StructuredGrid(grid_x, grid_y, z_bottom)

        plotter.add_mesh(top_surface, opacity=0.7, color=colors[seam], show_edges=True, label=f'{seam} Top Surface', show_scalar_bar=False)
        plotter.add_mesh(bottom_surface, opacity=0.4, color=colors[seam], show_edges=True, label=f'{seam} Bottom Surface', show_scalar_bar=False)

    if display:
        plotter.show()

    return plotter



# ---- Helper function to build figure ---- #
def build_figure(holeid: str, pick_df: pd.DataFrame, seam_mapping, collar_df: pd.DataFrame, kriging_input_df: pd.DataFrame, rop_df: pd.DataFrame, seam_mapped_pred_df: pd.DataFrame = None) -> go.Figure:
    '''
    This function builds a comprehensive figure for a given holeid that includes:
    - Actual picks from the pick_df
    - Gamma prediction mapping from seam_mapped_pred_df (if provided)
    - Kriging predictions based on the seam_mapping and kriging_input_df
    - A proximity plot showing all holes with color coding for holes used in kriging input and holes with seam mapping to gamma predictions (i.e. holes that are in our seam_mapped_pred_df)
    - ROP (Rate of Penetration) data from rop_df for the given holeid, plotted alongside the picks to provide additional context on drilling conditions which may have influenced the coal seam predictions and kriging results.

    Parameters:
    - holeid: The ID of the hole for which the figure is to be built.
    - pick_df: DataFrame containing the actual picks for all holes.
    - seam_mapping: Mapping of seams for kriging predictions.
    - collar_df: DataFrame containing collar information for all holes.
    - kriging_input_df: DataFrame containing kriging input data.
    - rop_df: DataFrame containing ROP (Rate of Penetration) data for all holes.
    - seam_mapped_pred_df: DataFrame containing gamma prediction mapping for seams (optional).

    Returns:
    - fig: A Plotly Figure object containing the comprehensive visualization for the given holeid.
    '''

    if seam_mapped_pred_df is not None:
        cols = 5
        subplot_titles = ("Actual Picks", "Gamma Pred Mapping", "Kriging Predictions", "ROP", "HoleProximity")
        col_widths = [0.15, 0.15, 0.15, 0.15, 0.4]
        gamma_pred_pick = seam_mapped_pred_df[seam_mapped_pred_df['HoleID'] == holeid]
    else:
        cols = 4
        subplot_titles = ("Actual Picks", "Kriging Predictions", "ROP", "HoleProximity")
        col_widths = [0.2, 0.2, 0.2, 0.4]

    hole_act_pick = pick_df[pick_df['holeid'] == holeid]
    krig_pred_pick = pd.DataFrame()
    for seam_info in get_coal_picks_for_hole(holeid, seam_mapping, collar_df):
        east = seam_info['east']
        north = seam_info['north']
        pred_top = seam_info['pred_top']
        pred_bottom = seam_info['pred_bottom']
        conf = seam_info['confidence']
        seam = seam_info['seam']

        krig_pred_pick = pd.concat([krig_pred_pick, pd.DataFrame({
            'HoleID': holeid,
            'EAST': east,
            'NORTH': north,
            'TopDepth': pred_top,
            'BottomDepth': pred_bottom,
            'confidence': conf,
            'seam': seam
        }, index=[0])], ignore_index=True)

    plys = set(hole_act_pick['ply'])
    if not krig_pred_pick.empty:
        plys = sorted(plys.union(set(krig_pred_pick['seam'])))
    if seam_mapped_pred_df is not None:
        plys = sorted(set(plys).union(set(gamma_pred_pick['seam'])))
    color_map = {ply: qualitative.Plotly[i % len(qualitative.Plotly)] for i, ply in enumerate(plys)}

    hole_collar = collar_df[collar_df['holeid'] == holeid].iloc[0]
    hole_rop = rop_df[rop_df['Hole_ID'] == holeid]
    collar = hole_collar['rl']
    toe = hole_collar['rl'] - hole_collar['depth']

    fig = make_subplots(
        rows=1, cols=cols,
        subplot_titles=subplot_titles,
        column_widths=col_widths
    )

    # ---- Actual Picks ---- #
    for _, row in hole_act_pick.iterrows():
        fig.add_trace(go.Scatter(
            y=[row['from'], row['from'], row['to'], row['to'], row['from']],
            x=[0, 1, 1, 0, 0],
            mode='lines',
            fill='toself',
            line=dict(color=color_map[row['ply']], width=2),
            name=row['ply'],
            hovertemplate=f"{row['ply']}<br>{row['from']:.2f} - {row['to']:.2f}",
            hoverinfo='text'
        ), row=1, col=1)

    if seam_mapped_pred_df is not None:
        for _, row in gamma_pred_pick.iterrows():
            fig.add_trace(go.Scatter(
                y=[row['TopDepth'], row['TopDepth'], row['BottomDepth'], row['BottomDepth'], row['TopDepth']],
                x=[0, 1, 1, 0, 0],
                mode='lines',
                fill='toself',
                line=dict(color=color_map[row['seam']], width=2, dash='dash'),
                name=row['seam'],
                hovertemplate=f"{row['seam']}<br>{row['TopDepth']:.2f} - {row['BottomDepth']:.2f}<br>XGB Conf: {row['Confidence']:.2f}"
            ), row=1, col=2)

    # ---- Kriging ---- #
    krig_col = 3 if seam_mapped_pred_df is not None else 2
    for _, row in krig_pred_pick.iterrows():
        fig.add_trace(go.Scatter(
            y=[row['TopDepth'], row['TopDepth'], row['BottomDepth'], row['BottomDepth'], row['TopDepth']],
            x=[0, 1, 1, 0, 0],
            mode='lines',
            fill='toself',
            line=dict(color=color_map[row['seam']], width=2),
            name=row['seam'],
            customdata=[row['confidence']] * 5,
            hovertemplate="Seam: %{name}<br>Top: %{y[0]:.2f}<br>Bottom: %{y[2]:.2f}<br>Kriging Conf: %{customdata[0]:.2f}<extra></extra>", 
            hoveron='fills'
        ), row=1, col=krig_col)



    # ---- ROP ---- #
    fig.add_trace(go.Scatter(
        x=hole_rop['ROP'],
        y=hole_rop['RelDepth'],
        mode='markers+lines',
        line=dict(color='orange', width=2),
        marker=dict(size=6, color='orange'),
        name='ROP',
        hovertemplate="ROP: %{x:.2f}<br>Depth: %{y:.2f}"
    ), row=1, col=krig_col + 1)

    # ---- Smoothened ROP ---- #
    if len(hole_rop) >= 5:  # Only smoothen if we have enough data points
        from scipy import signal

        fig.add_trace(go.Scatter(
            x=signal.savgol_filter(hole_rop['ROP'], 5, 2),
            y=hole_rop['RelDepth'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Smoothed ROP',
            hovertemplate="Smoothed ROP: %{x:.2f}<br>Depth: %{y:.2f}"
        ), row=1, col=krig_col + 1)

    # ---- All holes (clickable) ---- #
    prox_col = krig_col + 2
    color_1_bool = collar_df['holeid'].isin(kriging_input_df['holeid'].unique())
    color_1 = 'lightgreen' # Holes used in kriging input
    color_2 = 'lightblue' # Holes for which we have seam mapping to gamma predictions (i.e. holes that are in our seam_mapped_pred_df)
    color_3 = 'lightgray' # Holes that are neither in kriging input nor in seam_mapped_pred_df
    if seam_mapped_pred_df is not None:
        color_2_bool = collar_df['holeid'].isin(seam_mapped_pred_df['HoleID'].unique())
        colors = [color_1 if color_1_bool[i] else (color_2 if color_2_bool[i] else color_3) for i in range(len(collar_df))]
    else:
        colors = [color_1 if color_1_bool[i] else color_3 for i in range(len(collar_df))]

    fig.add_trace(go.Scatter(
        x=collar_df['east'],
        y=collar_df['north'],
        mode='markers',
        marker=dict(size=6, color=colors),
        customdata=collar_df['holeid'],
        name='All Holes'
    ), row=1, col=prox_col)

    # Highlight selected hole
    fig.add_trace(go.Scatter(
        x=[hole_collar['east']],
        y=[hole_collar['north']],
        customdata=[holeid],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Selected Hole'
    ), row=1, col=prox_col)

    fig.update_yaxes(range=[toe, collar], row=1, col=1, title="Depth")
    fig.update_yaxes(matches='y', row=1, col=krig_col)
    fig.update_yaxes(matches='y', row=1, col=krig_col + 1)

    if seam_mapped_pred_df is not None:
        fig.update_yaxes(matches='y', row=1, col=2)

    fig.update_layout(
        height=600,
        width=1500,
        showlegend=False,
        title=f"HoleID: {holeid}"
    )

    return fig



def _perform_kriging(input_df, x_bounds, y_bounds, grid_resolution, display=False):

    seam_planes = {}

    color_dict = _define_color_dict(input_df['ply'].unique(), colormap=plt.cm.viridis)
    plotter = _create_pyviz_plotter(title="Kriging-based Seam Surface Generation")

    for seam in input_df.ply.unique():
        seam_picks = input_df[input_df.ply == seam]
        if len(seam_picks) < 10:
            print(f"Warning: Seam {seam} has less than 10 picks ({len(seam_picks)}), which may lead to unreliable surface generation.")
            continue

        x = seam_picks['east'].values
        y = seam_picks['north'].values
        top = seam_picks['from'].values
        depth = seam_picks['from'].values - seam_picks['to'].values

        kriging_top = OrdinaryKriging(x, y, top, variogram_model='power', verbose=False, enable_plotting=False)
        kriging_depth = OrdinaryKriging(x, y, depth, variogram_model='power', verbose=False, enable_plotting=False)
        seam_planes[seam] = (kriging_top, kriging_depth)

        _add_plane_and_points_to_plotter(
            plotter, seam_picks, kriging_top, kriging_depth, x_bounds, y_bounds, grid_resolution, color=color_dict[seam]
        )
        
    # plotter.show_grid(color='lightgray')
    if display:
        plotter.show()
    return seam_planes, plotter


def get_coal_picks_for_hole(holeid: str, seam_planes: dict[str, OrdinaryKriging], collar_df: pd.DataFrame = collar_df):
    '''
    Generate the coal pick predictions for a given hole based on the kriging seam planes and the hole's collar information.

    Input:
    - holeid: The ID of the hole for which to generate predictions.
    - seam_planes: A dictionary mapping seam names to their corresponding kriging models (top and depth).
    - collar_df: A DataFrame containing collar information for the holes.

    Output:
    - A generator yielding dictionaries with predicted seam information for the given hole, including:
        - 'holeid': The ID of the hole.
        - 'seam': The name of the seam.
        - 'pred_top': The predicted top depth of the seam at the hole location.
        - 'pred_bottom': The predicted bottom depth of the seam at the hole location.
        - 'confidence': A confidence score for the prediction based on kriging variance.
    '''

    collar = collar_df[collar_df['holeid'] == holeid].iloc[0]
    east = collar['east']
    north = collar['north']
    top_d = collar['rl']
    bottom_d = collar['rl'] - collar['depth']

    for seam in seam_planes.keys():
        seam_top, seam_depth = seam_planes[seam]
        pred_top, top_var = seam_top.execute('points', np.array([east]), np.array([north]))
        pred_depth, depth_var = seam_depth.execute('points', np.array([east]), np.array([north]))

        pred_top = pred_top[0]
        pred_depth = pred_depth[0]
        top_conf = 1 / (1 + top_var[0])
        depth_conf = 1 / (1 + depth_var[0])

        pred_bottom = pred_top - pred_depth
        if pred_top >= top_d or pred_bottom <= bottom_d:
            continue

        yield {
            'holeid': holeid,
            'east': east,
            'north': north,
            'seam': seam,
            'pred_top': pred_top,
            'pred_bottom': pred_bottom,
            'confidence': (top_conf + depth_conf) / 2
        }


def align_seams_dynamic_prog(seam_df, pred_df):
    n, m = len(seam_df), len(pred_df)

    SKIP_SEAM_COST = 3
    MAX_DEPTH_DIFF = 10
    OFFSET_SMOOTH_WEIGHT = 3.0

    # -------------------------------
    # Cost functions
    # -------------------------------
    def match_cost(seam, pred):
        depth_diff = abs(seam["mid"] - pred["mid"])
        thickness_diff = abs(seam["thickness"] - pred["thickness"])

        # hard constraint
        if depth_diff > MAX_DEPTH_DIFF:
            return 1e6

        conf_weight = 1 + (1 - seam["confidence"])

        return conf_weight * (depth_diff + 2 * thickness_diff)

    def compute_offset(seam, pred):
        return seam["mid"] - pred["mid"]

    def offset_transition_cost(prev_offset, curr_offset):
        return OFFSET_SMOOTH_WEIGHT * (curr_offset - prev_offset) ** 2

    # -------------------------------
    # DP tables
    # -------------------------------
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    parent = {}
    offset_state = {}  # store last offset at each state

    offset_state[(0, 0)] = None

    # -------------------------------
    # DP loop
    # -------------------------------
    for i in range(n + 1):
        for j in range(m + 1):

            if dp[i, j] == np.inf:
                continue

            # -----------------------
            # 1️⃣ Assign pred[j] → seam[i]
            # -----------------------
            if i < n and j < m:
                seam = seam_df.iloc[i]
                pred = pred_df.iloc[j]

                curr_offset = compute_offset(seam, pred)

                cost = match_cost(seam, pred)

                prev_offset = offset_state.get((i, j))

                if prev_offset is not None:
                    cost += offset_transition_cost(prev_offset, curr_offset)

                if dp[i, j] + cost < dp[i, j + 1]:
                    dp[i, j + 1] = dp[i, j] + cost
                    parent[(i, j + 1)] = (i, j)
                    offset_state[(i, j + 1)] = curr_offset

            # -----------------------
            # 2️⃣ Skip seam (no preds mapped to it)
            # -----------------------
            if i < n:
                if dp[i, j] + SKIP_SEAM_COST < dp[i + 1, j]:
                    dp[i + 1, j] = dp[i, j] + SKIP_SEAM_COST
                    parent[(i + 1, j)] = (i, j)
                    offset_state[(i + 1, j)] = offset_state.get((i, j))

            # ⚠️ No skip-pred transition because every pred must be assigned

    # -------------------------------
    # Backtracking
    # -------------------------------
    mapping = []
    i, j = n, m

    while (i, j) in parent:
        pi, pj = parent[(i, j)]

        # pred assigned when j increases but i stays same
        if j == pj + 1 and i == pi:
            mapping.append((i, pj))  # seam i ← pred pj

        i, j = pi, pj

    mapping.reverse()

    # -------------------------------
    # Format output
    # -------------------------------
    result = []

    for seam_idx, pred_idx in mapping:
        result.append({
            "pred_idx": pred_idx,
            "seam_id": seam_df.iloc[seam_idx]["seam"],
            "krig_conf": seam_df.iloc[seam_idx]["confidence"]
        })

    return result



def predict_seams_for_hole(pred_df: pd.DataFrame, seam_planes: dict) -> pd.DataFrame:
    '''
    Predicts seams for each hole based on kriging predictions.

    Parameters:
    - pred_df: DataFrame containing predicted seam information.
    - seam_planes: Dictionary containing kriging models for each seam.

    Returns:
    - DataFrame with predicted seams and their confidence for each hole.
    '''

    assert 'EAST' in pred_df.columns and 'NORTH' in pred_df.columns and 'HoleID' in pred_df.columns, "pred_df must contain 'EAST', 'NORTH', and 'HoleID' columns."

    pred_seam_hole_df = pred_df[['EAST', 'NORTH', 'HoleID']].drop_duplicates().reset_index(drop=True)
    output_df = pd.DataFrame()

    for idx, holeid in enumerate(pred_seam_hole_df['HoleID'].unique()):
        # if holeid != 'BH13LLDB05LH2':
        #     continue
        print(f"Processing HoleID: {holeid} ({idx + 1}/{len(pred_seam_hole_df['HoleID'].unique())})", " "*20, end="\r")

        hole = pred_seam_hole_df[pred_seam_hole_df['HoleID'] == holeid].iloc[0]
        collar = collar_df[collar_df['holeid'] == holeid].iloc[0]

        top_d = collar['rl']
        bottom_d = collar['rl'] - collar['depth']
        east = hole['EAST']
        north = hole['NORTH']
        holeid = hole['HoleID']

        kriging_predictions_df = pd.DataFrame()

        # For each seam, predict top and bottom depth at the hole location and calculate confidence based on kriging variance. Then filter seams that are outside the hole's depth range.
        for seam, (kriging_top, kriging_depth) in seam_planes.items():
            pred_top, top_var = kriging_top.execute('points', np.array([east]), np.array([north]))
            pred_depth, depth_var = kriging_depth.execute('points', np.array([east]), np.array([north]))

            pred_top = pred_top[0]
            pred_depth = pred_depth[0]

            top_conf = 1 / (1 + top_var[0])
            depth_conf = 1 / (1 + depth_var[0])

            pred_bottom = pred_top - pred_depth
            if pred_top >= top_d or pred_bottom <= bottom_d:
                # We don't care about this, since this seam is outside our hole's depth range
                continue

            kriging_predictions_df = pd.concat([kriging_predictions_df, pd.DataFrame({
                'holeid': holeid,
                'pred_top': pred_top,
                'pred_bottom': pred_bottom,
                'confidence': (top_conf + depth_conf) / 2,
                'seam': seam
            }, index=[0])], ignore_index=True)

        hole_pred_df = pred_df[pred_df['HoleID'] == holeid].reset_index(drop=True).copy()

        def add_depth_and_thickness(df, top_col, bottom_col):
            df['thickness'] = df[bottom_col] - df[top_col]
            df['mid'] = (df[top_col] + df[bottom_col]) / 2
            return df
        
        def find_best_global_offset(gamma_seams, krig_seams, offset_range=(-10, 10), step=0.1):
            """
            Finds the best single vertical offset to align the entire dataset.
            This accounts for systematic shifts between prediction methods.
            """
            best_offset = 0
            min_total_dist = float('inf')

            for offset in np.arange(offset_range[0], offset_range[1], step):
                total_dist = 0
                # Temporarily apply offset to gamma midpoints
                gamma_midpoints_shifted = gamma_seams['mid'] + offset
                
                # For each shifted gamma seam, find the closest krig seam
                for g_mid in gamma_midpoints_shifted:
                    total_dist += np.min(np.abs(g_mid - krig_seams['mid']))
                    
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_offset = offset
                    
            # print(f"Found best global offset: {best_offset:.2f}")
            return best_offset

        if kriging_predictions_df.empty or hole_pred_df.empty:
            # print(f"Hole {holeid} - No predictions or no actual seams to align.")
            continue

        # print(f"\nHole {holeid} - Aligning {len(kriging_predictions_df)} kriging predictions with {len(hole_pred_df)} actual seams.", " "*20,)

        kriging_predictions_df.sort_values(by='pred_top', ascending=False, inplace=True, ignore_index=True)
        hole_pred_df.sort_values(by='TopDepth', ascending=False, inplace=True, ignore_index=True)

        kriging_predictions_df = add_depth_and_thickness(kriging_predictions_df, 'pred_top', 'pred_bottom')
        hole_pred_df = add_depth_and_thickness(hole_pred_df, 'TopDepth', 'BottomDepth')
        
        # 3. Find the best global offset to align the two datasets
        best_offset = find_best_global_offset(hole_pred_df, kriging_predictions_df)
        kriging_predictions_df['pred_top'] -= best_offset
        kriging_predictions_df['pred_bottom'] -= best_offset
        kriging_predictions_df['mid'] -= best_offset

        seam_alignment = align_seams_dynamic_prog(kriging_predictions_df, hole_pred_df)
        # return kriging_predictions_df, hole_pred_df
        if isinstance(seam_alignment, tuple):
            seam_alignment, global_offset = seam_alignment
            # print(f"Hole {holeid} - Estimated Global Depth Offset: {global_offset:.2f}")

        hole_pred_df['seam'] = {x['pred_idx']: x['seam_id'] for x in seam_alignment}
        hole_pred_df['krig_conf'] = {x['pred_idx']: x['krig_conf'] for x in seam_alignment}

        output_df = pd.concat([output_df, hole_pred_df], ignore_index=True)
    
    output_df = output_df.dropna(subset=['seam']).reset_index(drop=True)
    return output_df


non_gamma_picks = pick_df[~pick_df.holeid.isin(ads.Hole_ID.unique())].reset_index(drop=True)

# Stratified sampling of non-gamma picks to ensure we have a representative sample across the spatial extent of the data
shuffled_ads = ads[['Hole_ID', 'EAST', 'NORTH']].drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(shuffled_ads) * 0.7)

gamma_pred_holes = shuffled_ads.iloc[:split_idx]['Hole_ID'].unique()
gamma_train_holes = shuffled_ads.iloc[split_idx:]['Hole_ID'].unique()

pred_seam_df = pred_seam_df[pred_seam_df['HoleID'].isin(gamma_pred_holes)].reset_index(drop=True)

kriging_input_df = pick_df[~pick_df.holeid.isin(gamma_pred_holes)].reset_index(drop=True)

kriging_input_df = kriging_input_df[['holeid', 'ply', 'from', 'to', 'east', 'north']].copy()
kriging_input_df['Source'] = 'Engineer Picks'
kriging_input_df.head()

x_bounds = (pick_df.east.min(), pick_df.east.max())
y_bounds = (pick_df.north.min(), pick_df.north.max())

print(f"X bounds for kriging: {x_bounds}")
print(f"Y bounds for kriging: {y_bounds}")

seam_mapping, plotter = _perform_kriging(kriging_input_df, x_bounds, y_bounds, GRID_RESOLUTION, display=False)

# plotter.show()

shuffled_preds = pred_seam_df[['HoleID', 'EAST', 'NORTH']].drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(shuffled_preds) * 0.6)

pred_holes = shuffled_preds.iloc[:split_idx]['HoleID'].unique()
test_holes = shuffled_preds.iloc[split_idx:]['HoleID'].unique()

gamma_pred_seam_df = pred_seam_df[pred_seam_df['HoleID'].isin(pred_holes)].reset_index(drop=True)
non_gamma_df = pred_seam_df[pred_seam_df['HoleID'].isin(test_holes)].reset_index(drop=True)

app_1 = dash.Dash(__name__)

# ---- Layout ---- #
app_1.layout = html.Div([
    dcc.Graph(id='main-graph-1', figure=build_figure(collar_df.iloc[0]['holeid'], pick_df, seam_mapping, collar_df, kriging_input_df, rop_df, seam_mapped_pred_df=seam_mapped_pred_df))
])

# ---- Callback ---- #
@app_1.callback(
    Output('main-graph-1', 'figure'),
    Input('main-graph-1', 'clickData')
)
def update_on_click(clickData):

    if clickData is None:
        return dash.no_update
        # holeid = collar_df.iloc[0]['holeid']
    else:
        if clickData['points'][0].get('customdata') is None:
            # If customdata is not available, we can't determine the holeid, so we return the current figure without updating
            return dash.no_update
        
        holeid = clickData['points'][0]['customdata']

    return build_figure(holeid, pick_df, seam_mapping, collar_df, kriging_input_df, rop_df, seam_mapped_pred_df=seam_mapped_pred_df)


# ---- Run ---- #
if __name__ == '__main__':
    app_1.run(debug=True, port=8055)


new_krig_inputs = seam_mapped_pred_df.rename(columns={
    'HoleID': 'holeid',
    'EAST': 'east',
    'NORTH': 'north',
    'TopDepth': 'from',
    'BottomDepth': 'to',
    'seam': 'ply',
}).drop(columns=['mid', 'krig_conf', 'Confidence', 'thickness'])

new_krig_inputs['Source'] = f'Gamma Log Predictions {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

updated_kriging_input_df = pd.concat([kriging_input_df, new_krig_inputs], ignore_index=True)
updated_kriging_input_df = updated_kriging_input_df.groupby(['holeid', 'ply']).agg({'from': 'max', 'to': 'min', 'east': 'first', 'north': 'first', 'Source': 'first'}).reset_index()


# unmapped_gamma_pred_df = pred_seam_df[~pred_seam_df['HoleID'].isin(kriging_input_df['holeid'].unique())].reset_index(drop=True)
# krig_coal_pick_df = pd.DataFrame()

# for hole in unmapped_gamma_pred_df['HoleID'].unique():
#     for seam_info in get_coal_picks_for_hole(hole, seam_mapping, collar_df):
#         east = seam_info['east']
#         north = seam_info['north']
#         pred_top = seam_info['pred_top']
#         pred_bottom = seam_info['pred_bottom']
#         conf = seam_info['confidence']
#         seam = seam_info['seam']

#         krig_coal_pick_df = pd.concat([krig_coal_pick_df, pd.DataFrame({
#             'HoleID': hole,
#             'EAST': east,
#             'NORTH': north,
#             'TopDepth': pred_top,
#             'BottomDepth': pred_bottom,
#             'confidence': conf,
#             'seam': seam
#         }, index=[0])], ignore_index=True)



app_2 = dash.Dash(__name__)

# ---- Layout ---- #
app_2.layout = html.Div([
    dcc.Graph(id='main-graph-2', figure=build_figure(collar_df.iloc[0]['holeid'], pick_df, seam_mapping, collar_df, updated_kriging_input_df, rop_df))
])

# ---- Callback ---- #
@app_2.callback(
    Output('main-graph-2', 'figure'),
    Input('main-graph-2', 'clickData')
)
def update_on_click(clickData):

    if clickData is None:
        return dash.no_update
        # holeid = collar_df.iloc[0]['holeid']
    else:
        if clickData['points'][0].get('customdata') is None:
            # If customdata is not available, we can't determine the holeid, so we return the current figure without updating
            return dash.no_update
        
        holeid = clickData['points'][0]['customdata']

    return build_figure(holeid, pick_df, seam_mapping, collar_df, updated_kriging_input_df, rop_df)

# ---- Run ---- #
if __name__ == '__main__':
    app_2.run(debug=True, port=8052)


