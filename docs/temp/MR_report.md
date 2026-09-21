


- Finding1- README #l45
On the pip installation line, I ran into an error when running it on my mac with a macOS13.
I figured out that the last version of pyproj and rasterio do not have a downloadable built in version for macOS<14 (see pyproj documentation here https://pypi.org/project/pyproj/#files). I dont fully understand why pip does not install a version that is compatible with the user OS, but that was my case.
There were two ways I could solve the issue:
Installing the dependencies through conda directly. Here I think conda install the libraries to compile the packages. I did it myself on macOS13 and I could instal envio through pip afterwards.
Upgrading macOS to a >14 version. Obviously, this solved the problem.


Maybe you want to have an issue on Github where to refer in case some users have the same problem.

- Finding2- README #l58
Three of the four credentials placement options show no code to run.
For a non very experienced user, it would be nice to add a chunk of code for the three first cases:

```python
from envoi import init_gee
init_gee()
```

- Finding 3- README Overall code chunks

I think is clear for us, but thinking on non experienced users that want to test the command line, nothing spells out what goes in the bash terminal or in python script/notebook. Maybe explicitly mention it would help.

- Finding 4- Dataproduct type on web server

The dataproduct for each run on the server has to be either all of it tabular or raster. Maybe some users want to download tabular and raster for same or different collections in one run.

Maybe consider adding the tabular/raster as an option within the data product choice.

-Finding 5- extract.py #l422 Output with some redundant columns

when the user provides points in a crs different than wgs84, on the csv output (both stats and qc) there are some columns that I think are redundant: lat_original, lon_original, and also the lat_wgs84 and lon_wgs84, these last two are added as the last two columns of the output.
I personally do not see the use of having the lat_original and lon_original columns. I would delete them.
I would add the wgs84 columns after the original columns, not at the end of the output dataset.

I tried on dev mode and modifying the 422++ lines to these ones could do it:
```python
            if "lat_original" in df_copy.columns:
                insert_at = len(core_columns)
                for output_frame in (stats_df, qc_df):
                    lat_wgs84 = output_frame["lat"]
                    lon_wgs84 = output_frame["lon"]
                    output_frame["lat"] = df_copy.loc[output_frame.index, "lat_original"]
                    output_frame["lon"] = df_copy.loc[output_frame.index, "lon_original"]
                    output_frame.insert(insert_at, f"{latitude_column}_wgs84", lat_wgs84)
                    output_frame.insert(insert_at + 1, f"{longitude_column}_wgs84", lon_wgs84)
                stats_df = stats_df.drop(columns=["lat_original", "lon_original"])
```

- Finding 6- small bug for categorical datasets with Nas
I run a test set of points for all collections. On the lulc_naturallands collection, some output cells were empty although the qc file showed no absent data. 
It turns out that the regex that assigns the content for those cells does not recognize the name of the bands for that collection, which "classification_class_*_count". The regular expression does not do anything with them and they have Nas. 
A change on the reg expression on _output_assembly.py line 32 could fix it: 
```python
_CLASS_COLUMN_RE = re.compile(r"^(?:\w+_)?class_-?\d+_(count|fraction)$")
```
