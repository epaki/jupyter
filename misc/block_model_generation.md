🧱 BLOCK MODEL GENERATION

In this code, we procedurally construct a realistic 3D mining block model.
This isn't just a random grid of numbers — the block model is infused with geostatistical
textures, geological realism, and spatial logic that reflect how real orebody models are
built and interpreted in industry. Let’s walk through the process, layer by layer.

1. 📐 Grid Initialization and Geometry
We begin by unpacking the spatial configuration defined in Step 1, including:

origin = model_dimensions["origin"]
extent = model_dimensions["extent"]
block_size = model_dimensions["block_size"]

From this, we calculate the total number of blocks along each axis:

nx = int(extent[0] / block_size[0])
ny = int(extent[1] / block_size[1])
nz = int(extent[2] / block_size[2])

This discretizes the model space into regular voxel units (blocks), which will be iterated
through in a nested loop later.

2. 🏔️ Topographic Surface Generation
To simulate a realistic mining environment, we define a synthetic topographic surface:

topo_surface = np.zeros((nx, ny))
for j in range(ny):
	for i in range(nx):
		noise_val = pnoise3(i * 0.02, j * 0.02, 0, base=999)
		topo_surface[i, j] = topo_base_elevation + topo_variation_amplitude * noise_val

Using Perlin noise, we create a smooth, undulating surface typical of natural terrains.
This surface becomes essential for:
* Weathering zone classification (oxide, transition, fresh)
* Topographic filtering of near-surface ore
* Future drillhole simulation realism

3. 🎯 Domain Membership via Spatial Masks
Next, we define a helper function, is_in_domain, that checks whether a given (x, y, z)
block center lies inside a spatial domain. The function supports several domain shapes,
but of particular note is the use of a plunging ellipsoid to represent the orebody:

elif shape == "plunging_ellipsoid":
    ...
    # Tapered geometry, directional plunge, and Perlin noise distortion

This method integrates multiple geological features:
* Plunge direction and azimuth using directional cosines
* Tapering with depth, simulating narrowing ore shoots
* Organic edges via Perlin noise, mimicking irregular ore contacts

This moves us away from purely geometric primitives toward geometrically distorted but
geologically plausible orebodies.

4. ⚡ Fault Zone Computation
The compute_fault_distance and fault_membership_category functions calculate each block’s
proximity to a fault plane with adjustable curvature, thickness, and lateral offset:

distance, local_thickness = compute_fault_distance(x, y, z, fault_plane)

This uses a mathematical plane model with:
* Dip and azimuth
* Local curvature (sine wave modulation)
* Offset logic for faulted domains
* Localized thickness variation
* Roughness via 3D noise

The result is a powerful framework that distinguishes:
* Fault core
* Damage halo
* Unaffected zones

This enables selective enrichment or dilution logic during block value assignment.

5. 🔁 Block-by-Block Generation Loop
We then enter the 3D nested loop that spans the entire model volume. For each block, we compute:

a. Coordinates

x = origin[0] + (i + 0.5) * block_size[0]
y = origin[1] + (j + 0.5) * block_size[1]
z = origin[2] + (k + 0.5) * block_size[2]

Each block is centered in its voxel. The coordinates are stored in block_data.

b. Concentric Zone Assignment
To mimic zonation (e.g., core → outer halo), we calculate the normalized distance from
the orebody center:

norm_dist = (dx / a)**2 + (dy / b)**2 + (dz / c)**2

This gives us 5 zones (0 to 4) based on distance bands. These are later used for:
* Grade scaling
* Lithology assignment
* Processing route decisions

These zones are also stored in a categorical concentric_zone label, enabling dual
representations (numeric + semantic).

c. Zone-Based Grade Scaling
Before generating grades, we use the zone_grade_multipliers dictionary to scale Au/Cu
grades based on zonal richness:

raw_grade * zone_factors.get(mineral, 1.0)

This mimics industry behavior where cores are higher grade and halos are diluted.

We then apply Perlin noise to generate naturalistic grade textures. This noise is
scaled, normalized, and converted to a grade value:

noise_val = (pnoise3(...) + 1) / 2
grade = min_val + noise_val * (max_val - min_val)

Grades are smooth, continuous, and spatially correlated — a step up from random sampling.

d. Zone-Based Lithology and Processing Route
Instead of relying solely on depth or location for lithology or processing classification,
we now base these on concentric zones:

block_data["lith_code"] = lithology_by_zone[zone]
block_data["processing_route"] = processing_route_by_zone[zone]

This tightens the spatial logic and reflects real geological behavior where rock type
and processing style often correlate with zonation.

e. Fault Influence & Grade Enrichment
We use the earlier fault membership logic to conditionally enhance grades:

if fault_status == "core":
	block_data[key] *= fault_grade_multiplier[mineral]

This allows structural enrichment logic, commonly observed in hydrothermal systems,
shear zones, or contact zones.

f. Additional Property Simulation
We enrich the block model with additional attributes:
* Geophysical (e.g., density, resistivity, gamma)
* Geomechanical (e.g., UCS, RMR)
* Metallurgical (e.g., Axb, BWi, grindability)
* Processing indices (e.g., floatability, reagent demand)

Each follows the same pattern:
* 3D Perlin noise for spatial continuity
* Range scaling for realism
* Optional interdependencies (e.g., RMR = f(UCS))

This modular approach allows toggling each system via flags like enable_geomechanical_properties.

g. Depth Below Surface & Oxidation
We compute vertical distance from each block to the local surface:

depth_below_surface = local_surface_elevation - z

This is used to classify blocks into:
* Oxide
* Transition
* Fresh

Reflecting natural supergene profiles.

h. Other Geological Classifications
Using optional parameters and spatial logic, we assign:
* Alteration (based on 3D noise thresholding)
* Stratigraphy (based on x-position)
* Grain size (based on radial distance)

These can support geomet modeling, processing decisions, or drillhole domain tagging.

6. ✅ Block Inclusion Logic
At the end of each iteration, we determine whether a block should be added:

if in_orebody or include_margin_zone:
	block_list.append(block_data)

This filters out irrelevant blocks unless explicitly told to include marginal surroundings —
a useful toggle for downstream pit shell simulations.

7–8. 🧾 Final Output
We convert our block_list into a DataFrame:

block_model_df = pd.DataFrame(block_list)

Then add fault zone flags and preview the result:

block_model_df["fault_category"] = block_model_df["fault_zone_category"] == "core"
block_model_df.head()

🔁 Summary of Benefits
This approach blends:
* Real-world mining logic (zonation, fault enrichment, topographic weathering)
* Statistical realism (Perlin noise, parameter ranges)
* Modular flexibility (toggle systems, override rules)
* Efficient implementation (loop-based with helper functions and scalable structure)

The result is a highly customizable, geologically plausible synthetic block model that can
support downstream simulation, visualization, or machine learning experiments.