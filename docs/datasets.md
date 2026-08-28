# Built-in datasets

*Generated from `src/envoi/configs/ee_catalog.yml` by `scripts/generate_dataset_docs.py` — edit the catalog, not this file.*

envoi ships with **26 datasets** ready to use — no downloads, no configuration. Each entry below shows the dataset's name followed by the catalog key in `code font` — that key is what you pass to `extract()`:

```python
from envoi import extract

extract(points_dataframe, {
    "batch_id": "terrain",
    "datasets": ["dem_copernicus_glo30"],
    "settings": {"statistics": ["mean"], "window_size_m": 200},
})
```

To add your own local rasters or Earth Engine assets, register them with `update_catalog()` — see [advanced_usage.md](advanced_usage.md).

## Contents

- [Terrain](#terrain) (1 dataset)
- [Climate](#climate) (3 datasets)
- [Land cover / land use](#land-cover--land-use) (3 datasets)
- [Satellite imagery](#satellite-imagery) (3 datasets)
- [Vegetation & productivity](#vegetation--productivity) (7 datasets)
- [Human impact](#human-impact) (8 datasets)
- [Other](#other) (1 dataset)

## Terrain

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**Copernicus DEM GLO-30**](#copernicus-dem-glo-30)<br>`dem_copernicus_glo30` | Copernicus DEM GLO-30 - global digital surface model derived from TanDEM-X radar data | 30 meters | static | continuous |

#### Copernicus DEM GLO-30

Copernicus DEM GLO-30 - global digital surface model derived from TanDEM-X radar data.

| | |
| --- | --- |
| **Use in `extract()`** | `dem_copernicus_glo30` |
| **Source** | Earth Engine — `COPERNICUS/DEM/GLO30` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | static |
| **Default bands** | `DEM` |
| **Derived bands available** | `slope`, `aspect` |
| **Licence** | Copernicus Data License (free for most uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30) · [Provider documentation](https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf) |

> **Cite as:** © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

## Climate

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**ERA5 Monthly Aggregates**](#era5-monthly-aggregates)<br>`climate_era5_monthly` | ERA5 monthly aggregates - global climate reanalysis from ECMWF | 27830 meters | monthly, 1979-present | continuous |
| [**TerraClimate Monthly Climate & Water Balance**](#terraclimate-monthly-climate--water-balance)<br>`climate_terraclimate_monthly` | TerraClimate - monthly global climate and water-balance variables (temperature, precipitation, ET, runoff, soil moisture, snow water equivalent, PDSI, water deficit, radiation, vapor pressure, wind speed) | 4638 meters | monthly, 1958-present | continuous |
| [**WorldClim BIO Variables V1**](#worldclim-bio-variables-v1)<br>`climate_worldclim_v1_bioclim` | WorldClim v1 bioclimatic variables - 19 variables derived from temperature and precipitation | 927 meters | static, average over 1960-1990 | continuous |

#### ERA5 Monthly Aggregates

ERA5 monthly aggregates - global climate reanalysis from ECMWF

| | |
| --- | --- |
| **Use in `extract()`** | `climate_era5_monthly` |
| **Source** | Earth Engine — `ECMWF/ERA5/MONTHLY` |
| **Values** | continuous |
| **Spatial resolution** | 27830 meters |
| **Temporal resolution** | monthly |
| **Temporal coverage** | 1979-present |
| **Default bands** | all bands |
| **Licence** | Copernicus C3S/CAMS License |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY) · [Provider documentation](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) |

Extraction notes:

- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home

#### TerraClimate Monthly Climate & Water Balance

TerraClimate - monthly global climate and water-balance variables (temperature, precipitation, ET, runoff, soil moisture, snow water equivalent, PDSI, water deficit, radiation, vapor pressure, wind speed).

| | |
| --- | --- |
| **Use in `extract()`** | `climate_terraclimate_monthly` |
| **Source** | Earth Engine — `IDAHO_EPSCOR/TERRACLIMATE` |
| **Values** | continuous |
| **Spatial resolution** | 4638 meters |
| **Temporal resolution** | monthly |
| **Temporal coverage** | 1958-present |
| **Default bands** | all bands |
| **Licence** | CC0 (Public Domain) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE) · [Provider documentation](https://www.climatologylab.org/terraclimate.html) |

Extraction notes:

- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958-2015. Scientific Data 5:170191. https://doi.org/10.1038/sdata.2017.191

#### WorldClim BIO Variables V1

WorldClim v1 bioclimatic variables - 19 variables derived from temperature and precipitation.

| | |
| --- | --- |
| **Use in `extract()`** | `climate_worldclim_v1_bioclim` |
| **Source** | Earth Engine — `WORLDCLIM/V1/BIO` |
| **Values** | continuous |
| **Spatial resolution** | 927 meters |
| **Temporal resolution** | static |
| **Temporal coverage** | average over 1960-1990 |
| **Default bands** | all bands |
| **Licence** | CC-BY-SA-4.0 |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO) · [Provider documentation](https://www.worldclim.org/data/bioclim.html) |

> **Cite as:** Hijmans, R.J., Cameron, S.E., Parra, J.L., Jones, P.G., & Jarvis, A. (2005). Very high resolution interpolated climate surfaces for global land areas. International Journal of Climatology, 25, 1965-1978.

## Land cover / land use

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**Copernicus Global Land Cover 100 m**](#copernicus-global-land-cover-100-m)<br>`lulc_copernicus_lc100` | Copernicus Global Land Service Land Cover 100m - global land cover maps from PROBA-V | 100 meters | annual, 2015-2019 | categorical |
| [**ESA WorldCover 10 m (2021)**](#esa-worldcover-10-m-2021)<br>`lulc_worldcover_2021` | ESA WorldCover 2021 - global land use / land cover map with 11 classes | 10 meters | static, 2021 | categorical |
| [**SBTN Natural Lands Map v1.1**](#sbtn-natural-lands-map-v11)<br>`lulc_naturallands_2020` | SBTN Natural Lands Map v1.1 - global map distinguishing natural from non-natural land cover, produced for the Science Based Targets Network | 30 meters | static, 2020 | categorical |

#### Copernicus Global Land Cover 100 m

Copernicus Global Land Service Land Cover 100m - global land cover maps from PROBA-V.

| | |
| --- | --- |
| **Use in `extract()`** | `lulc_copernicus_lc100` |
| **Source** | Earth Engine — `COPERNICUS/Landcover/100m/Proba-V-C3/Global` |
| **Values** | categorical |
| **Spatial resolution** | 100 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2015-2019 |
| **Default bands** | `discrete_classification` |
| **Licence** | Fully free and open to all users |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) · [Provider documentation](https://land.copernicus.eu/global/products/lc) |

> **Cite as:** Buchhorn, M., Smets, B., Bertels, L., Roo, B. D., Lesiv, M., Tsendbazar, N.-E., Herold, M., & Fritz, S. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (Version V3.0.1) [Data set]. Zenodo.

#### ESA WorldCover 10 m (2021)

ESA WorldCover 2021 - global land use / land cover map with 11 classes.

| | |
| --- | --- |
| **Use in `extract()`** | `lulc_worldcover_2021` |
| **Source** | Earth Engine — `ESA/WorldCover/v200` |
| **Values** | categorical |
| **Spatial resolution** | 10 meters |
| **Temporal resolution** | static |
| **Temporal coverage** | 2021 |
| **Default bands** | all bands |
| **Licence** | CC-BY-4.0 |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) · [Provider documentation](https://esa-worldcover.org) |

> **Cite as:** Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O., 2022. ESA WorldCover 10 m 2021 v200.

#### SBTN Natural Lands Map v1.1

SBTN Natural Lands Map v1.1 - global map distinguishing natural from non-natural land cover, produced for the Science Based Targets Network.

| | |
| --- | --- |
| **Use in `extract()`** | `lulc_naturallands_2020` |
| **Source** | Earth Engine — `WRI/SBTN/naturalLands/v1_1/2020` |
| **Values** | categorical |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | static |
| **Temporal coverage** | 2020 |
| **Default bands** | all bands |
| **Licence** | CC-BY-SA-4.0 |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/WRI_SBTN_naturalLands_v1_1_2020) · [Provider documentation](https://github.com/wri/natural-lands-map/tree/main) |

> **Cite as:** Mazur, E., Sims, M., Goldman, E., Schneider, M., Pirri, M. D., Beatty, C. R., Stolle, F., & Stevenson, M. (2025). SBTN Natural Lands Map v1.1: Technical Documentation. Science Based Targets Network.

## Satellite imagery

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**Landsat 32-Day Surface Reflectance Composite**](#landsat-32-day-surface-reflectance-composite)<br>`sr_landsat_32day` | Landsat 32-day surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal) | 30 meters | 32 days, 1984-present | continuous |
| [**Landsat 8-Day Surface Reflectance Composite**](#landsat-8-day-surface-reflectance-composite)<br>`sr_landsat_8day` | Landsat 8-day surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal) | 30 meters | 8 days, 1984-present | continuous |
| [**Landsat Annual Surface Reflectance Composite**](#landsat-annual-surface-reflectance-composite)<br>`sr_landsat_annual` | Landsat annual surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal) | 30 meters | annual, 1984-present | continuous |

#### Landsat 32-Day Surface Reflectance Composite

Landsat 32-day surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal).

| | |
| --- | --- |
| **Use in `extract()`** | `sr_landsat_32day` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_32DAY` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | 32 days |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_32DAY) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat surface reflectance products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat 8-Day Surface Reflectance Composite

Landsat 8-day surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal).

| | |
| --- | --- |
| **Use in `extract()`** | `sr_landsat_8day` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_8DAY` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | 8 days |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_8DAY) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat surface reflectance products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat Annual Surface Reflectance Composite

Landsat annual surface reflectance composite - global composites of Landsat C2 L2 surface reflectance (blue, green, red, nir, swir1, swir2, thermal).

| | |
| --- | --- |
| **Use in `extract()`** | `sr_landsat_annual` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_ANNUAL) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat surface reflectance products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

## Vegetation & productivity

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**ESA CCI Above-Ground Biomass v6.0**](#esa-cci-above-ground-biomass-v60)<br>`agb_esa_cci` | ESA CCI Above-Ground Biomass v6.0 - global forest above-ground biomass (Mg/ha) with per-pixel uncertainty | 100 meters | irregular (selected years), 2007, 2010, 2015-2022 | continuous |
| [**Landsat 32-Day NDVI Composite**](#landsat-32-day-ndvi-composite)<br>`ndvi_landsat_32day` | Landsat 32-day NDVI - global composites of NDVI | 30 meters | 32 days, 1984-present | continuous |
| [**Landsat 8-Day EVI Composite**](#landsat-8-day-evi-composite)<br>`evi_landsat_8day` | Landsat 8-day EVI - global composites of Enhanced Vegetation Index | 30 meters | 8 days, 1984-present | continuous |
| [**Landsat 8-Day NDVI Composite**](#landsat-8-day-ndvi-composite)<br>`ndvi_landsat_8day` | Landsat 8-day NDVI - global composites of NDVI | 30 meters | 8 days, 1984-present | continuous |
| [**Landsat Annual EVI Composite**](#landsat-annual-evi-composite)<br>`evi_landsat_annual` | Landsat Annual EVI - global composites of Enhanced Vegetation Index | 30 meters | annual, 1984-present | continuous |
| [**Landsat Annual NDVI Composite**](#landsat-annual-ndvi-composite)<br>`ndvi_landsat_annual` | Landsat Annual NDVI - global composites of NDVI | 30 meters | annual, 1984-present | continuous |
| [**MODIS Terra Net Primary Production (MOD17A3HGF v061)**](#modis-terra-net-primary-production-mod17a3hgf-v061)<br>`npp_modis_terra` | MODIS Terra Net Primary Productivity (MOD17A3HGF v061) - NPP and GPP (kg C/m²) plus QC | 500 meters | annual, 2001-2024 | continuous |

#### ESA CCI Above-Ground Biomass v6.0

ESA CCI Above-Ground Biomass v6.0 - global forest above-ground biomass (Mg/ha) with per-pixel uncertainty.

| | |
| --- | --- |
| **Use in `extract()`** | `agb_esa_cci` |
| **Source** | Earth Engine — `ESA/CCI/Above_Ground_Biomass/V6_0` |
| **Values** | continuous |
| **Spatial resolution** | 100 meters |
| **Temporal resolution** | irregular (selected years) |
| **Temporal coverage** | 2007, 2010, 2015-2022 |
| **Default bands** | all bands |
| **Licence** | ESA CCI Biomass License (free with citation) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_CCI_Above_Ground_Biomass_V6_0) · [Provider documentation](https://climate.esa.int/en/projects/biomass/) |

> **Cite as:** Santoro, M., & Cartus, O. (2025). ESA Biomass Climate Change Initiative (Biomass_cci): Global datasets of forest above-ground biomass for the years 2007, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021 and 2022, v6.0. NERC EDS Centre for Environmental Data Analysis. https://doi.org/10.5285/95913ffb6467447ca72c4e9d8cf30501

#### Landsat 32-Day NDVI Composite

Landsat 32-day NDVI - global composites of NDVI.

| | |
| --- | --- |
| **Use in `extract()`** | `ndvi_landsat_32day` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDVI` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | 32 days |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_32DAY_NDVI) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat NDVI products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat 8-Day EVI Composite

Landsat 8-day EVI - global composites of Enhanced Vegetation Index.

| | |
| --- | --- |
| **Use in `extract()`** | `evi_landsat_8day` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_8DAY_EVI` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | 8 days |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_8DAY_EVI) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat EVI products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat 8-Day NDVI Composite

Landsat 8-day NDVI - global composites of NDVI.

| | |
| --- | --- |
| **Use in `extract()`** | `ndvi_landsat_8day` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | 8 days |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_8DAY_NDVI) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat NDVI products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat Annual EVI Composite

Landsat Annual EVI - global composites of Enhanced Vegetation Index.

| | |
| --- | --- |
| **Use in `extract()`** | `evi_landsat_annual` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_EVI` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_ANNUAL_EVI) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat EVI products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### Landsat Annual NDVI Composite

Landsat Annual NDVI - global composites of NDVI.

| | |
| --- | --- |
| **Use in `extract()`** | `ndvi_landsat_annual` |
| **Source** | Earth Engine — `LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI` |
| **Values** | continuous |
| **Spatial resolution** | 30 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 1984-present |
| **Default bands** | all bands |
| **Licence** | Public Domain (free for all uses) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_ANNUAL_NDVI) · [Provider documentation](https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices) |

Extraction notes:

- Native resolution is set manually to 30 m (the asset does not report a usable scale to Earth Engine).
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Landsat NDVI products courtesy of the U.S. Geological Survey Earth Resources Observation and Science Center.

#### MODIS Terra Net Primary Production (MOD17A3HGF v061)

MODIS Terra Net Primary Productivity (MOD17A3HGF v061) - NPP and GPP (kg C/m²) plus QC.

| | |
| --- | --- |
| **Use in `extract()`** | `npp_modis_terra` |
| **Source** | Earth Engine — `MODIS/061/MOD17A3HGF` |
| **Values** | continuous |
| **Spatial resolution** | 500 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2024 |
| **Default bands** | all bands |
| **Licence** | Public Domain (no restrictions on use) |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A3HGF) · [Provider documentation](https://doi.org/10.5067/MODIS/MOD17A3HGF.061) |

Extraction notes:

- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Running, S., & Zhao, M. (2021). MODIS/Terra Net Primary Production Gap-Filled Yearly L4 Global 500m SIN Grid V061 [Data set]. NASA Land Processes Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD17A3HGF.061

## Human impact

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**Human Impact Index (HII)**](#human-impact-index-hii)<br>`human_impact_index` | Human Impact Index (HII) - cumulative measure of human influence on terrestrial ecosystems | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Infrastructure**](#human-impact-index-infrastructure)<br>`hii_driver_infrastructure` | Human Impact Index driver - infrastructure | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Land Use**](#human-impact-index-land-use)<br>`hii_driver_land_use` | Human Impact Index driver - land use | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Population Density**](#human-impact-index-population-density)<br>`hii_driver_population_density` | Human Impact Index driver - population density | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Power**](#human-impact-index-power)<br>`hii_driver_power` | Human Impact Index driver - power | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Railways**](#human-impact-index-railways)<br>`hii_driver_railways` | Human Impact Index driver - railways | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Roads**](#human-impact-index-roads)<br>`hii_driver_roads` | Human Impact Index driver - roads | 300 meters | annual, 2001-2020 | continuous |
| [**Human Impact Index: Water**](#human-impact-index-water)<br>`hii_driver_water` | Human Impact Index driver - water | 300 meters | annual, 2001-2020 | continuous |

#### Human Impact Index (HII)

Human Impact Index (HII) - cumulative measure of human influence on terrestrial ecosystems.

| | |
| --- | --- |
| **Use in `extract()`** | `human_impact_index` |
| **Source** | Earth Engine — `projects/HII/v1/hii` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Infrastructure

Human Impact Index driver - infrastructure.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_infrastructure` |
| **Source** | Earth Engine — `projects/HII/v1/driver/infrastructure` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Land Use

Human Impact Index driver - land use.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_land_use` |
| **Source** | Earth Engine — `projects/HII/v1/driver/land_use` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Population Density

Human Impact Index driver - population density.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_population_density` |
| **Source** | Earth Engine — `projects/HII/v1/driver/population_density` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Power

Human Impact Index driver - power.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_power` |
| **Source** | Earth Engine — `projects/HII/v1/driver/power` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Railways

Human Impact Index driver - railways.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_railways` |
| **Source** | Earth Engine — `projects/HII/v1/driver/railways` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Roads

Human Impact Index driver - roads.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_roads` |
| **Source** | Earth Engine — `projects/HII/v1/driver/roads` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

#### Human Impact Index: Water

Human Impact Index driver - water.

| | |
| --- | --- |
| **Use in `extract()`** | `hii_driver_water` |
| **Source** | Earth Engine — `projects/HII/v1/driver/water` |
| **Values** | continuous |
| **Spatial resolution** | 300 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2001-2020 |
| **Default bands** | all bands |
| **Licence** | CC BY-NC-SA-3.0 |
| **Links** | [Earth Engine catalog](https://code.earthengine.google.com/f904097220e577cad2e0dc5379371c91) · [Provider documentation](https://wcshumanfootprint.org/) |

> **Cite as:** Sanderson, E. W., Fisher, K., Robinson, N., Sampson, D., Duncan, A., & Royte, L. (2022). The march of the human footprint.

## Other

| Dataset | What it is | Resolution | Temporal | Values |
| --- | --- | --- | --- | --- |
| [**Google Satellite Embedding V1 (Annual)**](#google-satellite-embedding-v1-annual)<br>`aef_satellite_embeddings` | Google Satellite Embeddings V1 - 64-dimensional embeddings | 10 meters | annual, 2017-2025 | continuous |

#### Google Satellite Embedding V1 (Annual)

Google Satellite Embeddings V1 - 64-dimensional embeddings.

| | |
| --- | --- |
| **Use in `extract()`** | `aef_satellite_embeddings` |
| **Source** | Earth Engine — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| **Values** | continuous |
| **Spatial resolution** | 10 meters |
| **Temporal resolution** | annual |
| **Temporal coverage** | 2017-2025 |
| **Default bands** | all bands |
| **Licence** | CC-BY-4.0 |
| **Links** | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) · [Provider documentation](http://arxiv.org/abs/2507.22291) |

Extraction notes:

- Tiled collection: the tile matching each point's UTM zone is selected, so points near tile edges get the right image.
- Image selection from a sample date uses the `contains` policy (`contains` = the image whose time interval covers the date; `nearest` = the image with the closest timestamp).

> **Cite as:** Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., Zhang, C., Shelhamer, E., Lahera, E., Wiles, O., Ilyushchenko, S., Gorelick, N., Zhang, L. L., Alj, S., Schechter, E., Askay, S., Guinan, O., Moore, R., Boukouvalas, A., & Kohli, P. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv preprint arXiv:2507.22291.

---

The catalog source, with every field for every entry, is [`src/envoi/configs/ee_catalog.yml`](../src/envoi/configs/ee_catalog.yml). The same information is available programmatically via `list_datasets("full")`.
