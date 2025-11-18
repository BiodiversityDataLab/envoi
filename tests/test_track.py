
import pandas as pd
from pathlib import Path

# Import your actual Prefect flow from the pipeline file
from flow.prefect_biodiversity_pipeline import biodata_enrichment_flow

print("--- This script will trigger a full Prefect flow run ---")

# 1. Create your sample DataFrame in memory
data = {'id': [1, 2], 'lat': [59.3, 59.4], 'lon': [18.1, 18.2]}
points_df = pd.DataFrame(data)

# 2. Save the DataFrame to a temporary CSV file
#    Your flow is designed to read from a file, so we create one.
temp_csv_path = Path("temp_points_for_flow.csv")
points_df.to_csv(temp_csv_path, index=False)
print(f"--- Sample data saved to '{temp_csv_path}' ---")

# 3. Define the parameters for your Prefect flow
#    We will use the "groups mode" since that is what your flow is designed for.
flow_parameters = {
    "input_csv": str(temp_csv_path),
    "groups_yaml": "configs/groups.yml",
    "catalog_yaml": "configs/catalog.yml",
    "window_m": 500,
    "out_dir": "out"
}

# 4. Call your Prefect flow with the parameters
#    This is a blocking call. The script will wait for the flow to finish.
if __name__ == "__main__":
    print("\n--- Starting the Prefect 'biodata_enrichment_flow' ---")
    try:
        # Running the flow function directly triggers a Prefect run
        result = biodata_enrichment_flow(**flow_parameters)
        print("\n--- Prefect flow finished successfully! ---")
        print("Flow output paths:", result)
    except Exception as e:
        print(f"\n--- The Prefect flow failed: {e} ---")