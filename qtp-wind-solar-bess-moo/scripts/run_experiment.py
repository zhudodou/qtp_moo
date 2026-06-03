"""Run the Wind-Solar-BESS spatial MOO experiments from a NetCDF input file.

Example
-------
python scripts/run_experiment.py --input data/MPDEV2018.nc --output results/demo --runs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from qtp_moo.optimization import ObjectiveData, moead, nsga2, spea2_with_de


SCENARIOS = {
    "Wind_LCOE": ("Wind_Power", "Wind_LCOE"),
    "Wind_BESS_LCOE": ("Wind_Power", "O_Wind_LCOE"),
    "Solar_LCOE": ("Solar_Power", "Solar_LCOE"),
    "Solar_BESS_LCOE": ("Solar_Power", "O_Solar_LCOE"),
    "Wind_Solar_LCOE": ("Wind_Solar_Power", "Wind_Solar_LCOE"),
    "Wind_Solar_BESS_LCOE": ("Wind_Solar_Power", "O_Wind_Solar_LCOE"),
}


def load_objective_data(ds: xr.Dataset, power_name: str, lcoe_name: str) -> ObjectiveData:
    if power_name == "Wind_Solar_Power" and power_name not in ds:
        ds = ds.assign(Wind_Solar_Power=ds["Wind_Power"] + ds["Solar_Power"])

    power = ds[power_name].values
    lcoe = ds[lcoe_name].values
    demand = ds["chec2019"].values
    distance = ds["chgriddis"].values
    valid = np.isfinite(power) & np.isfinite(lcoe) & np.isfinite(demand) & np.isfinite(distance)
    valid &= (power != 0) & (lcoe != 0)

    return ObjectiveData(
        power_kwh_per_hour=power[valid] * 1000.0,
        lcoe_cny_per_kwh=lcoe[valid],
        local_demand_kwh=demand[valid] * 100.0,
        grid_distance=distance[valid],
    )


def save_result(result, output_dir: Path, name: str, algorithm: str, run: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Generation": result.generations, "D_Indicator": result.d_indicator})
    df.to_csv(output_dir / f"{name}_{algorithm}_run{run:02d}_d_indicator.csv", index=False)
    np.savez_compressed(
        output_dir / f"{name}_{algorithm}_run{run:02d}_population_history.npz",
        **{f"gen_{gen}": pop for gen, pop in zip([1, 50, 100, 200, 500], result.history)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input NetCDF file, e.g. data/MPDEV2018.nc")
    parser.add_argument("--output", default=Path("results/run"), type=Path)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--pop-size", default=200, type=int)
    parser.add_argument("--generations", default=500, type=int)
    parser.add_argument("--crossover-rate", default=0.4, type=float)
    parser.add_argument("--mutation-factor", default=0.4, type=float)
    args = parser.parse_args()

    ds = xr.open_dataset(args.input)
    for scenario_name, (power_name, lcoe_name) in SCENARIOS.items():
        data = load_objective_data(ds, power_name, lcoe_name)
        for run in range(args.runs):
            seed = run
            results = {
                "NSGAII": nsga2(data, args.pop_size, args.generations, args.crossover_rate, args.mutation_factor, seed=seed),
                "MOEAD": moead(data, args.pop_size, args.generations, args.crossover_rate, args.mutation_factor, seed=seed),
                "SPEA2": spea2_with_de(
                    data,
                    args.pop_size,
                    args.pop_size,
                    args.generations,
                    args.crossover_rate,
                    args.mutation_factor,
                    seed=seed,
                ),
            }
            for algorithm, result in results.items():
                save_result(result, args.output / scenario_name, scenario_name, algorithm, run)


if __name__ == "__main__":
    main()
