# PIT SHELL & SCHEDULING ENGINE ARCHITECTURE

This document provides a detailed technical description of the architecture, logic, and step-by-step workflow behind the **pit shell generation** and **mining schedule assignment** components of the Mining Simulator. It outlines how the synthetic block model is loaded and processed, the design logic underpinning Lerchs-Grossman pit shell creation, and the subsequent assignment of extraction periods. Each computational step is documented alongside its purpose, methodology, and reasoning, to support ongoing development, debugging, and future enhancements. This markdown file is part of a modular documentation series, with other simulator components (e.g., blasting, stockpiles, diagnostics) covered in separate documents.


## 🧱 STEP 0: BLOCK MODEL LOADING
Objective: Load the base synthetic block model for processing and planning.

### Implementation:
* File loaded from: data/synthetic_block_model.parquet
* Function: load_block_model() in api.py
* Format: Parquet for efficient columnar access and filtering
* Typical columns include:
    * x, y, z: spatial grid coordinates
    * Au_grade, value_smu: economic parameters
    * zone_code, fault_zone, oxidation_state: geological and processing domains

### Why:
* A standardized, efficient format (Parquet) allows scalable filtering and partitioning.
* This is the foundation for all downstream modeling—accurate geometry and grade data are essential for realism.


## ⚙️ STEP 1: ECONOMIC VALUATION
Objective: Calculate per-block economic value (value_smu) used for classification and scheduling.

### Implementation:
* Function: calculate_economic_value(chunks, cfg)
* Configurable via: config.yaml
* Parameters: cutoff grade, commodity prices, recovery factors, processing cost assumptions
* Output columns:
    * value_per_tonne, net_value, value_smu (used to distinguish ore vs waste)

### Why:
* Enables filtering blocks into economically mineable material vs waste.
* Used later in pit shell generation, strip ratio, and scheduling decisions.


## ⛏ STEP 2: BUFFER BLOCK ADDITION
Objective: Add margin blocks around the orebody for pit shell feasibility and ramp space.

### Implementation:
* Function: add_buffer_blocks(chunks, cfg)
* Margin controlled by config: buffer_margin, pit_shell_padding_x/y
* Classification: Buffer blocks are added as non-economic waste material to define slope geometry

### Why:
* Ensures that slopes can be calculated around the orebody
* Allows proper LG shell development without edge distortion


## 🧠 STEP 3: DEPENDENCY AND LG PIT SHELL GENERATION
Objective: Solve the pit using the Lerchs-Grossmann algorithm with geotechnical and value constraints.

### Implementation:
* Function: solve_lerchs_grossman(block_model, dependencies, cfg)
* Preceded by: build_dependencies() (enforces slope, inter-ramp angle, access direction)
* Output:
    * within_shell flag
    * Intermediate NPV calculations
    * Slope and directionally influenced precedence graph

### Why:
* LG is industry standard for optimal pit design
* Incorporates both geotechnical and economic inputs
* Provides a set of "included" blocks for scheduling


## 🔁 STEP 4: PHASE AND PUSHBACK ASSIGNMENT
Objective: Group pit shell into mining phases and pushbacks for operational sequencing.

### Implementation:
* Function: assign_phases(), assign_pushback_bins()
* Based on shape, elevation, and access logic
* Adds: phase, bin, pushback attributes

### Why:
* Facilitates staged development
* Allows ramp integration and access path planning
* Reflects common operational pit progression practices


## 📆 STEP 5: PERIOD SCHEDULING
Objective: Assign blocks to periods to meet throughput and access constraints.

### Implementation:
* Function: assign_periods(df, cfg)
* Strategy: Bench-by-bench, phase-aware with configurable throughput/capacity
* Result: Adds period and bench attributes
* Validates:
    * Period continuity
    * Access ramp feasibility
    * Ore and waste blend targets

### Why:
* Enables production planning
* Forms the basis of diagnostics like vertical spread, ore continuity, and strip ratio


## 🧪 STEP 6: DIAGNOSTICS
Objective: Provide insight into the quality and realism of the generated schedule.

### Implementation:
* API endpoint: /pit-shell/diagnostics
* Frontend integration: DiagnosticsPage.tsx
* Metrics:
    * Unscheduled blocks (due to access)
    * Period count and vertical spread
    * Strip ratio by period (waste vs ore tonnes)
    * Ore loss ratio
    * Phase-period heatmap

### Why:
* Identifies bottlenecks or schedule inefficiencies
* Provides real-time decision support
* Offers export to CSV for audit and review