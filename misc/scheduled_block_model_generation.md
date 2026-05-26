Scheduled Block Model – Logic Summary
=====================================

This document describes the logic used to convert a synthetic geological block model into a
scheduled block model suitable for mine design, value optimisation, and operational simulation.

The process includes:
- Economic filtering and valuation per block
- Pit shell generation and geotechnical access logic
- Mining phase and period scheduling (with ramp-up curve)
- Volume, tonnage, and metal estimation
- SMU-based dilution and classification
- Output and visualisation options

-------------------------------------
1. ECONOMIC CALCULATIONS
-------------------------------------
value_per_tonne       = Au + Cu value using grade × price × recovery (per metal)
processing_cost       = Applied per block, only for ore
net_value             = value_per_tonne – mining_cost – processing_cost
class                 = "Ore" if net_value ≥ 0, else "Waste"

NSR deductions and recovery factors are applied per metal before net value is calculated.
Optionally, `value_smu` (smoothed value) may be used to simulate production-scale selectivity.

-------------------------------------
2. VOLUME & TONNAGE CALCULATIONS
-------------------------------------
volume_m3             = X × Y × Z block dimensions
tonnes                = volume × density
domain_summary        = Volume, tonnes, and grade aggregated per lithology or domain

Includes logic for volume reconciliation and SMU classification, if enabled.

-------------------------------------
3. RESOURCE ESTIMATION
-------------------------------------
Contained metal       = tonnes × grade (per metal)
Metal units           = g/t converted to oz (Au), % to tonnes (Cu)
resource_df           = Summary of tonnes, grade, metal content by domain/class

Used to validate geostatistics and inform economic thresholds.

-------------------------------------
4. SMU ADJUSTMENT LOGIC
-------------------------------------
value_smu             = Smoothed using 2D spatial filter (XY) to simulate dilution
class_smu             = Reclassified ore/waste using smoothed value

Simulates loss of selectivity during production-scale blasting and loading.

-------------------------------------
5. BLOCK TRACKING & IDS
-------------------------------------
block_id              = Unique per block (e.g. "x_y_z")
Used for traceability, auditing, and re-linking during scheduling and reporting.

-------------------------------------
6. PIT SHELL & ACCESS LOGIC
-------------------------------------
inter_ramp_slope      = Used to define allowable slope geometry
catch_berms           = Bench-level flattening at vertical intervals
within_shell          = True if block lies inside generated elliptical shell
has_access            = True if reachable from surface via ramp access

Pit shell is computed using elliptical radius based on block depth, slope angle, and berm width.
Major fault zones apply slope penalties for geotechnical realism.

-------------------------------------
7. MINING PHASE ASSIGNMENT
-------------------------------------
pushback_x/y          = Horizontal binning by minimum mining width
bench                 = Discrete elevation level using bench height
phase                 = Assigned based on pushback and vertical order

Phases reflect a logical sequence of pushbacks, prioritising shallower and higher-value areas.
Phase 0 is reserved for prestrip of top waste benches.

-------------------------------------
8. PERIOD SCHEDULING (WITH RAMP-UP CURVE)
-------------------------------------
period                = Assigned based on target movement curve per period
schedule_curve        = Ramp-up to steady state, hold, then stepwise ramp-down
bin_sorting           = Within each phase/bench: prioritise logical progression, not only value
cumulative_tonnage    = Rolling sum used to manage period transitions

The period assignment honours fleet capacity while maintaining operational realism:
- Early ramp-up from 40–100% of steady capacity
- Sustained production over mid-life
- Controlled ramp-down toward final periods
- Optional cap on strip ratio or phase sequence

-------------------------------------
9. STRIP RATIO, CASHFLOW & VALUE METRICS
-------------------------------------
ore_tonnes            = Sum of ore blocks per period
waste_tonnes          = Sum of waste blocks per period
strip_ratio           = waste_tonnes / ore_tonnes
revenue               = Total metal value before costs
profit                = Revenue – mining – processing costs
npv                   = Discounted profit (8% default)
payback_period        = First period where cumulative NPV covers initial capex

Cashflow analysis reflects realistic early generation of value and support for sustaining capex.

-------------------------------------
10. EXPORT FORMATS
-------------------------------------
- CSV       → readable, archived format
- Parquet   → compressed high-performance storage
- JSON      → ready for API streaming
- Visuals   → Gantt charts, strip ratio plots, ramp curves, metal output

Output files are placed in `../data/` or configured directory for downstream workflows.
