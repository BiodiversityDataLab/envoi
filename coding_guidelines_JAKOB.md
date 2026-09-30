# Coding guidelines


## Purpose

This guide captures project-specific coding and documentation patterns for the codebase. The goal is consistency, readability, and low-friction collaboration
for both human and AI contributors. Follow this guide for code structure, naming, comments, docstrings, and repo-specific implementation choices. Follow `pyproject.toml` for basic formatting and linting.

(Formatting note: Keep double blank lines between sections in markdown file.)


## Core principles

- Prefer clarity over cleverness or brevity.
- Keep behavior config-driven where practical, avoid hardcoding values in code.
- Avoid unnecessary abstractions or layers where there is no clear benefit.
- Keep pipeline code explicit and easy to trace step by step.
- Code is rarely self-documenting, add comments to help yourself and contributors.
- Make the smallest change that fits the current architecture.
- Follow general PEP guidelines unless in conflict with these guidelines.


## Key design patterns

### General structure of Tasks

Task classes should keep configuration loading in `__init__()`, orchestration in
`run_task()`, and dataframe processing logic in small and modular methods.

Good example:

```python
class DarwinCoreProcessingTask:
    """Process biodiversity records that follow the Darwin Core (DwC) schema."""

    def __init__(self, run_folder_path: Path, logger: logging.Logger) -> None:
        task_config = data_configs.darwin_core

        self.dataset_dir: Path = dataset_dir(data_root=DATA_DIR)
        self.run_folder_path: Path = run_folder_path
        self.logger: logging.Logger = logger
        self.input_path: Path = dataset_path(
            task_config.input_path,
            data_root=DATA_DIR,
        )
        self.main_output_path: Path = task_output_path(
            "darwin_core",
            task_config.outputs.main,
            data_root=DATA_DIR,
        )
        self.required_input_cols: list[str] = list(task_config.required_input_cols)

    def run_task(self) -> None:
        """Run the Darwin Core processing workflow."""
        validate_input_files([self.input_path], self.logger, self.allowed_input_types)
        available_input_cols, self.input_separator = read_delimited_header(
            self.input_path
        )
        validate_data_schema(available_input_cols, self.required_input_cols, self.logger)

        df = self.load_input_data(available_input_cols)
        df = self.drop_invalid_rows(df)
        df = self.parse_and_validate_dates(df)
        df = self.finalize_output_dataframe(df)

        validate_data_schema(df, self.required_input_cols, self.logger)
        ensure_output_directories_exist([self.main_output_path.parent], self.logger)
        df.write_parquet(self.main_output_path)

    def load_input_data(self, available_input_cols: list[str]) -> pl.DataFrame:
        ...

    def drop_invalid_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        ...

    def parse_and_validate_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        ...
```

Why:
- `__init__()` stays focused on task configuration and shared attributes.
- `run_task()` remains readable as a short orchestration method, exposing the
  flow from start to finish.
- Clearly delimited methods keep task logic modular without hiding the main workflow.


### Task helpers and task constants

Use helper methods when they make the task easier to read, isolate meaningful
dataframe logic, or implement behavior that is reused. Do not split code into
extra methods only to make methods shorter.

Prefer:
- Public methods for main task phases called directly from `run_task()` or
  intentionally reused outside the class.
- Private methods with a leading underscore for internal helpers called only by
  other methods.
- Direct inline code for one-off transformations that are only one or two lines
  and are clearer at the call site.
- Task-specific constants as attributes loaded in `__init__()`, ideally from the
  task config.

Avoid:
- Methods that only call another method and does no processing by itself.
- Helpers that hide a single non-reusable line behind a new name.
- Module-level variables for task-specific IDs, column names, thresholds, or
  output settings that belong in the task config or task attributes.
- Public helper names for methods that are implementation details.

Bad example:

```python
TASK_ID = "environmental_data"
DATE_COL = "eventDate"


class EnvironmentalDataTask:
    def __init__(self, run_folder_path: Path, logger: logging.Logger) -> None:
        self.output_path = task_output_path(TASK_ID, "combined_env_data.parquet")

    def run_task(self) -> None:
        df = self.prepare_dates(df)

    def prepare_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.add_yearly_event_date_midpoint(df)

    def build_config(self, batch_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return self.resolve_config(batch_id, config)
```

Better:

```python
class EnvironmentalDataTask:
    def __init__(self, run_folder_path: Path, logger: logging.Logger) -> None:
        task_config = data_configs.environmental_data

        self.task_id: str = "environmental_data"
        self.date_col: str = task_config.date_col
        self.output_path = task_output_path(
            self.task_id,
            task_config.outputs.main,
        )

    def run_task(self) -> None:
        if self.use_yearly_date_midpoint:
            df = self.add_yearly_event_date_midpoint(df)

    def _build_config(self, batch_id: str, config: dict[str, Any]) -> dict[str, Any]:
        ...
```

Why:
- The task setup is easier to review when task-specific settings are initialized
  with the rest of the task configuration.
- Public methods describe the task's main processing stages; private helper
  names mark implementation details.
- Thin wrappers and one-line helpers add indirection without making the behavior
  easier to understand or test.


### Paths and global configs

Use `ROOT_DIR`, `DATA_DIR`, and the path resolver helpers from `src/paths.py`.
Do not build important paths from ad hoc relative assumptions scattered across
the codebase. Biodiversity source files should resolve through the active
dataset root, task outputs and run folders should resolve through the active
workspace, and shared environmental rasters should resolve through the
environmental-data root.

Prefer:

```python
from src.paths import DATA_DIR, ROOT_DIR, dataset_path, task_output_path

source_path = dataset_path(task_config.input_path, data_root=DATA_DIR)
output_path = task_output_path("darwin_core", task_config.outputs.main)
```

Use `global_configs.yaml` for settings that are shared across multiple tasks,
such as the active dataset, active workspace, and top-level data folder names.
Keep task-specific paths and options in the config file that lives next to the
task code.

Why:
- It avoids inconsistent assumptions about where data lives.
- It prevents cross-task settings from being copied into several task configs.


### Class attributes versus dataframe flow

Store config-derived values used across methods as class attributes. Pass
dataframes and other transient objects between methods instead of storing them
as mutable `self.df` state.

Preferred pattern:

```python
df = self.data_cleaning(df)
df = self.generate_site_ids(df)
```

Avoid:

```python
self.df = self.data_cleaning(self.df)
self.df = self.generate_site_ids(self.df)
```

Why:
- Passing `df` keeps state explicit and makes the processing flow more obvious.
- It is easier to test and reason about.
- Config belongs to the task instance; transient data usually does not.


### Config-driven code

Use config `yaml` files for values that are likely to vary by dataset, run, or
task setup.
- input and output paths,
- column names and mappings,
- dtype expectations,
- thresholds and options,
- selected modes or task settings.

Why:
- Config keeps the pipeline flexible.
- Settings and assumptions become transparent to the user.

In general, class attributes derived from config should match the names provided
in the config `yaml` file, to avoid confusion.


#### Config structure

Task-related configs live in one or more `*.yaml` files in the same subdirectory
that holds the `*.py` task file itself. The config can hold settings for one or
more tasks.

General structure:
- Each config file should contain one or more task-specific top-level sections
  that correspond to task modules in the same directory.
- Comments explaining the key-value pairs should be next to the key-value pair.
  If the comment is longer, place the comment above the key-value pair.
- All variables within that config should follow snake case conventions.
- Lists for numbers or settings with only few values should be in brackets [].
- Longer lists for strings, e.g. column names, should be listed under the variable name with a dash separating items.
- Standard indentation is 2 spaces.
- Before every task config, and logical subset of that config, place a heading
using inline comments.

Good example of a config for multiple tasks:
```yaml
# --------------- DarwinCoreProcessingTask ---------------

darwin_core:

  input_path: source/occurrence.txt  # Standard format for unzipped GBIF downloads.
  allowed_input_types: [".txt", ".csv"]
  outputs:
    main: processed_bio_data.parquet  # Full canonical output.
    location_date: location_date.parquet  # Site-date output.
    summary_stats: bio_data_stats.json  # Summary stats for review.

  infer_schema_length: null
  n_rows: 100
  date_filter:
    enabled: true
    date_range: [2010-01-01, 2020-12-31]

  required_input_cols:
    - decimalLatitude
    - decimalLongitude
    - eventDate
    - species
    - occurrenceStatus

# --------------- EnvironmentalDataTask ---------------

environmental_data:

  input_task: darwin_core
  input_path: location_date.parquet  # Output filename from DwC task.
  allowed_input_types: [".parquet"]
  outputs:
    envoi: envoi_outputs  # Raw files written by envoi.
    main: combined_env_data.parquet  # Final combined tabular output.
  input_crs: null

  required_input_cols:
    - locationID
    - decimalLatitude
    - decimalLongitude
    - geodeticDatum
    - coordinateUncertaintyInMeters
    - eventDate

  extraction_configs:
    topography:
      active: true
      datasets: [dem_glo30]
      settings:
        output_type: tabular
        statistics: [mean, std]
        window_size_m: [1000, 5000]
        output_file_format: parquet
```


### Shared validation

Use shared validation helpers from `src/validation/data_and_schema.py` directly
in task methods.

Prefer this pattern in `run_task()`:

```python
validate_input_files([self.input_data_path], self.logger, self.allowed_input_types)
df = pl.read_parquet(self.input_data_path)
validate_data_schema(df, self.required_input_cols, self.logger)

... processing ...

ensure_output_directories_exist([self.output_data_path.parent], self.logger)
check_invalid_values(df, self.required_input_cols, logger=self.logger)
```

Rules:
- Use `validate_input_files(...)` for existence, emptiness, and suffix checks.
- Use `validate_data_schema(data, required_cols, logger)` where `data` is
  typically a dataframe.
- Use `ensure_output_directories_exist(...)` before writing outputs.
- Use `check_invalid_values(...)` when you need invalid value logging and,
  optionally, a row mask for filtering. This should typically be run before
  writing outputs.
- Keep input/output validation explicit in `run_task()` when it is only one or
  two lines, rather than in nested methods.
- Name config-derived attributes after their config keys unless that would
  collide with an existing method name.

Avoid:
- Methods that mostly duplicate existing validators.
- Thin task-local methods that only read a dataframe and call one validator.
- Separate wrappers whose only purpose is to log dataframe shape.


#### Validation vs tests

Separate runtime validation from automated tests.

Use:
- `src/validation/` for reusable runtime checks across different modules.
- task-local validation functions when the logic is specific to one task.
- `src/tests/` for the future automated test suite.

Good pattern:
- shared file, schema, and invalid-value checks in `src/validation/data_and_schema.py`,
- task-specific integrity checks inside the relevant task module.

Why:
- Runtime validation is part of normal pipeline execution.
- Automated tests are a separate concern and should not be imported by runtime
  code.


### Tool preferences

- Prefer `polars` for core tabular processing. Use `pandas` or `geopandas` only where required by downstream APIs.
- Prefer existing libraries for solving a task; only create custom solutions where truly needed.


## Readability

### Naming conventions

Choose names that describe the actual contract of the component.

Prefer:

- `DarwinCoreProcessingTask` over `DataLoadingTask`.
- `paths.py` over `definitions.py`.
- `data_configs.yaml` over `configs.yaml`.
- `log_and_filter_low_coordinate_precision` over `coordinate_filter`.
- `build_summary_stats` over `calculate_stats`.
- `input_path` over `data_path` when the config key is `input_path`.
- `merge_extracted_data` over `write_tabular_output` when a method returns a dataframe.

Why:
- Explicit names make purpose and assumptions visible.
- Too generic names or abbreviations create confusion.


### Blank lines and logical blocks

Use blank lines to separate logical stages of a method. Typical pattern:

1. Input setup.
2. Validation.
3. Main transformation block(s).
4. Output or return block.

Good example:

```python
# Validate input files and columns before processing
validate_input_files([self.input_path], self.logger, self.allowed_input_types)
available_input_cols, self.input_separator = read_delimited_header(self.input_path)
validate_data_schema(available_input_cols, self.required_input_cols, self.logger)

# Apply the main data cleaning steps in sequence
df = self.drop_invalid_rows(df)
df = self.parse_and_validate_dates(df)

# Use only select dates if specified by user
if self.date_filter_enabled:
    df = apply_date_filter(df, self.date_filter_range, self.logger)

# Write the cleaned records and summary outputs
df.write_parquet(self.main_output_path)
self.summary_stats_path.write_text(
    json.dumps(summary_stats, indent=2),
    encoding="utf-8",
)
```

Bad example:

```python
validate_input_files([self.input_path], self.logger, self.allowed_input_types)
available_input_cols, self.input_separator = read_delimited_header(self.input_path)
validate_data_schema(available_input_cols, self.required_input_cols, self.logger)
df = self.load_input_data(available_input_cols)
df = self.drop_invalid_rows(df)
df = self.parse_and_validate_dates(df)
df = self.finalize_output_dataframe(df)
validate_data_schema(df, self.required_input_cols, self.logger)
ensure_output_directories_exist([self.main_output_path.parent], self.logger)
df.write_parquet(self.main_output_path)
```

Why:
- It makes pipeline methods feel sequential rather than dense.
- It makes the code more digestible for human readers.
- Logical separation and readability matters more than the exact number of lines.


### Explicit conditional control flow

Keep optional task behavior visible in `run_task()` when the condition affects
the main processing path. The helper method should do the work for the enabled
case; it should not hide whether the optional behavior is active.

Prefer:

```python
# Use only select dates if specified by user
if self.date_filter_enabled:
    df = apply_date_filter(df, self.date_filter_range, self.logger)
```

Avoid:

```python
df = apply_date_filter(
    df=df,
    enabled=self.date_filter_enabled,
    date_range=self.date_filter_range,
    logger=self.logger,
)
```

Why:
- `run_task()` remains the readable orchestration layer.
- Optional behavior is easier to review against config.
- Shared helpers stay focused on one operation instead of mixing control flow
  with transformation logic.


## Code documentation

### Typehints

Use type hints on method arguments, return values, and important attributes.
Prefer concrete standard-library and Polars types over vague catch-all types.

Good example:

```python
def merge_extracted_data(
    self,
    tabular_output_paths: list[Path],
    input_row_count: int,
) -> pl.DataFrame | None:
    """Merge tabular extraction outputs into one dataframe."""
```

Why:
- Type hints make task contracts easier to scan.
- They make it clearer where `None` is a valid outcome.
- They reduce ambiguity in dataframe-processing helpers.


### Docstrings

#### General rules

- Use sentence case and end docstrings with periods.
- Keep one-line docstrings for simple methods and helpers only, typically ones
  that involve just one or two steps and very few arguments.
- Use multi-line docstrings for classes, `__init__`, `run_task`, and methods with
  non-trivial behavior. Start with a summary line, then use a blank line before additional detail or section headings. Keep a blank line between the final line
  and ending triple quotes.
- Be concrete about important assumptions that matter for correct usage.
- Do not use docstrings to narrate every line of code. Explain purpose,
  contract, key behavior, inputs and outputs.

#### Class docstrings

Use the class docstring to explain the task or component at a high level. In three to four lines, make the purpose and the main assumptions obvious.

Good example:

```python
class DarwinCoreProcessingTask:
    """Process biodiversity records that follow the Darwin Core (DwC) schema.

    The task expects a delimited text file whose columns use Darwin Core field
    names. It validates the input schema, removes invalid / unusable records,
    standardizes and adds date and location fields, and writes standardized
    parquet outputs for downstream tasks such as analysis and modeling.

    """
```

Why:
- A reader should know what the class does before reading `__init__()` or
  `run_task()`.
- Important assumptions, such as accepted input format, belong here.


#### `__init__` docstring with `Args` and `Attributes`

For task classes, place the detailed attribute list in `__init__()` after the
`Args` section. The `Args` and `Attributes` sections of the docstring should be
lists with dashes. Do not repeat `run_folder_path`, `logger` and other arguments
passed to `__init__()` in `Attributes` unless the class stores some non-obvious variation of them.

Good example:

```python
def __init__(self, run_folder_path: Path, logger: logging.Logger) -> None:
    """Load task configuration and initialize shared task attributes.

    Args:
    - run_folder_path: Folder for outputs and logs for this DAG run.
    - logger: Shared logger for the current DAG run.

    Attributes:
    - dataset_dir: Dataset-specific folder inside shared data directory.
    - input_path: Input biodiversity CSV inside the sibling data directory.
    - main_output_path: Output parquet path for cleaned biodiversity data.
    - location_date_output_path: Output parquet path for location-date data.
    - required_input_cols: Columns that must exist in the source data.

    """
```

Why:
- It documents the task contract where the attributes are initialized.
- It makes config review faster.
- It keeps the attribute list focused on task configuration rather than
  repeating obvious constructor arguments.


#### `run_task` docstring with key steps

`run_task()` should describe the main phases of the workflow, not every minor
operation. Use a short intro sentence followed by a flat bullet list.

Good example:

```python
def run_task(self) -> None:
    """Run the Darwin Core processing workflow.

    - Validate the input file path, file type, and required source columns.
    - Load the configured DwC columns that are present in the source file.
    - Drop invalid rows, review coordinate precision, validate dates, and
      remove duplicate IDs.
    - Generate missing locationIDs.
    - Infer missing abundance information from repeated occurrence rows.
    - Finalize the output dataframe, export summary statistics, and write
      downstream outputs.
    """
```

Why:
- `run_task()` is the orchestration method, so the major stages should be easy
  to scan.
- A short step list helps new contributors understand control flow quickly.


### Multi-line method docstring with `Args` and `Returns`

Use this structure for all other methods.

Good example:

```python
def drop_invalid_rows(self, df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows where required source columns are null, NaN, or infinite.

    Args:
    - df: Loaded dataframe to filter.

    Returns:
    - Dataframe with invalid rows removed.

    """
```

Why:
- This is enough structure for most transformation methods.
- The reader gets purpose, input, and output without noise.


### Method docstring with `Uses attributes`

When a method depends on important class attributes from `self`, but those
values are not real call-time arguments, document them in a `Uses attributes`
section rather than listing them under `Args`.

Use this when the attribute dependency is important for understanding what the
method does. Skip it when the attribute use is trivial or obvious from the
method name and class context.

Good example:

```python
def summarize_taxonomy(df: pl.DataFrame, taxonomy_col: str) -> pl.DataFrame:
    """Summarize record counts by a configured taxonomic column.

    Args:
    - df: Dataframe with biodiversity records.
    - taxonomy_col: Taxonomic column used for grouping.

    Uses attributes:
    - logger: Task logger used to record the summary size.

    Returns:
    - Dataframe with one row per taxonomic value.

    """
```

Why:
- `Args` should describe the true function signature.
- Important dependencies on `self` should still be visible to the reader.
- This keeps method contracts explicit without repeating configuration values at
  every call site.


### One-line docstring

Use a one-line docstring only when the function is truly simple.

Good example:

```python
def parse_dtype_config(raw_dtype_config: dict[str, str]) -> dict[str, pl.DataType]:
    """Convert configured dtype strings to Polars dtypes."""
```

Why:
- Short helpers should stay compact.
- If a one-line docstring starts stretching, switch to a multi-line form.


### Inline comments

Inline comments can be used generously when they improve readability. Short comments at the start of a new logical block or processing step are encouraged, even when the block itself is fairly simple.

Rules:
- Start every sentence with a capital letter.
- Leave a blank line before an inline comment that introduces the next logical
  block.
- Keep comments short, usually one line.
- Prefer comments that mark the purpose of the next block.
- Use longer comments only when the logic or assumption is genuinely complex.
- Do not narrate every single line or repeat what a trivial assignment already
  says.

Good example:

```python
# Validate input files and columns before processing
validate_input_files([self.input_path], self.logger, self.allowed_input_types)

# Apply the main data cleaning steps in sequence
df = self.drop_invalid_rows(df)
df = self.parse_and_validate_dates(df)

# Use only select dates if specified by user
if self.date_filter_enabled:
    df = apply_date_filter(df, self.date_filter_range, self.logger)

# Write the cleaned records, site-date output, and summary statistics
df.write_parquet(self.main_output_path)
```

Bad example:

```python
# Set the output path
self.output_path = ROOT_DIR / config.output_path

# Set the logger
self.logger = logger
```

Why:
- Short block-level comments make long task methods easier to read.
- Readers should be able to skim a method by scanning the comments first.
- Redundant line-by-line commentary still adds noise, so comment the block or
  purpose rather than every assignment.
