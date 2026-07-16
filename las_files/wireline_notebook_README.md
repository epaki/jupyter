# MiningSim Wireline Hole Inspection Workbench

## Contents

- `MiningSim_wireline_hole_inspection_workbench.ipynb` — executed development notebook with parsers, QC, visualisations, engineering calculations and regression checks.
- `MiningSim_wireline_hole_inspection_workbench.html` — read-only review copy containing the executed outputs.
- `MiningSim_wireline_requirements.txt` — minimal Python package requirements.

The source measurement ZIP files are not duplicated in this package. Place the notebook and the supplied `Caliper*.zip` and `ACS03 Calibration runs*.zip` archives in the same working folder. The notebook reads the archives directly; they do not need to be extracted.

## Running the notebook

1. Create or select a Python 3.10+ Jupyter environment.
2. Install the requirements:

   ```bash
   python -m pip install -r MiningSim_wireline_requirements.txt
   ```

3. Open the notebook and run all cells.
4. Review the configuration cell before accepting engineering outputs. In particular, replace the example nominal diameter and tolerance with the hole plan values.
5. Set `DO_EXPORT = True` in the export section to write the canonical CSV and JSON hand-off tables.

## Engineering controls

The default caliper calculation treats `X1`, `X2`, `Y1` and `Y2` as independent radial arm distances, sums opposite arms into orthogonal diameters, approximates each cross-section as an ellipse and integrates area along depth using the trapezoidal rule. An alternative per-arm-diameter interpretation is included for comparison.

The four-arm channel convention, tool-body/arm-offset treatment, depth reference and directional-survey conventions must be confirmed with the equipment manufacturer or the applicable calibration procedure before production deployment. Calculation-profile versioning and retained source-file provenance are recommended for MiningSim.
