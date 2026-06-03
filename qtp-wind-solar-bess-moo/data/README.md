# Data Directory

Place the processed NetCDF input files here before running the scripts, for example:

- `MPDEV2018.nc`
- `MPDEV2030.nc`
- `MPDEV2050.nc`

The NetCDF files are not included in this GitHub-ready folder because they are
derived from multiple external geospatial and energy datasets and may be too
large for routine version control.

The expected variables include:

- `Wind_Power`
- `Solar_Power`
- `Wind_LCOE`
- `O_Wind_LCOE`
- `Solar_LCOE`
- `O_Solar_LCOE`
- `Wind_Solar_LCOE`
- `O_Wind_Solar_LCOE`
- `chec2019`
- `chgriddis`

See the manuscript methods section for the data sources and preprocessing
steps used to generate these processed inputs.
