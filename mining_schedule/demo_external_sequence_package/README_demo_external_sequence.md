# Demo external scheduler sequence package

This package contains a synthetic external scheduler sequence generated from the supplied demo data:

- `blast_master.csv`
- `block_model_copper.csv`
- `drill_charge_designs.zip`

It is not an output from a commercial scheduling package. It is an import-ready demo sequence designed to exercise the Scenario Simulation workflow.

## Files

- `demo_external_scheduler_sequence.csv` — pattern-level sequence for import into the Scenario Simulation workspace.
- `demo_external_scheduler_periods.csv` — day-level period summaries.
- `demo_external_scheduler_schedule_asset.json` — full schedule asset matching the current Blast Master prototype schema as closely as possible.
- `create_demo_external_sequence.ipynb` — notebook to regenerate or tune the sequence from the same source files.

## Sequence basis

- Horizon: 2 weeks / 14 daily periods.
- Bench: RL185-200.
- Patterns: 27 total.
- Scheduling logic: Ramp and slot first, production patterns next, trim/boundary patterns late.
- Objective: balanced feed and bench progression for demonstration.
- Start date: 2026-06-01T06:00:00Z.

## Demo summary

- Total scheduled tonnes: 4,390,290 t
- Ore tonnes: 4,087,815 t
- Waste tonnes: 302,475 t
- Average Cu grade: 1.043%
- Contained Cu: 42,651 t
- Recovered Cu proxy: 33,716 t
- Drill metres: 29,242 m
- Explosive mass: 672,081 kg

## Recommended use

Use `demo_external_scheduler_sequence.csv` if the workspace import expects a flat external scheduler file. Use `demo_external_scheduler_schedule_asset.json` if the import expects the local Blast Master schedule asset shape.

If the importer requires different column names, map these core columns:

- `pattern_id`
- `sequence_index`
- `scheduled_period`
- `scheduled_start`
- `scheduled_finish`
- `destination`
- `estimated_tonnes`
- `ore_tonnes`
- `waste_tonnes`
- `average_grade`
- `drill_metres_m`
- `explosive_kg`
- `risk_flags`
