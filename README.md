# Pandas

Data analysis and machine learning notes, built around **Pandas**. The
documentation is ordered the way a data science project actually runs: the
process first, then how to explore data, then the Pandas techniques that make
those steps practical.

## Documentation

| File | Contents |
| --- | --- |
| [01.Data Science Process.md](01.Data%20Science%20Process.md) | The eight steps, from defining a problem to deployment - each with checklists, real code, anti-patterns, and a full Titanic walkthrough |
| [02.EDA.md](02.EDA.md) | Exploratory Data Analysis in depth - structure, data quality, distributions, outliers, relationships, time series, and visual EDA |
| [Pandas_Notes.pdf](Pandas_Notes.pdf) | Pandas reference notes |
| [Pandas_Assignment_Problems.pdf](Pandas_Assignment_Problems.pdf) | Assignment problem set |

## Notebooks

| File | Contents |
| --- | --- |
| [pandas_tutorial.ipynb](pandas_tutorial.ipynb) | Core Pandas walkthrough - Series, DataFrames, I/O, selection, filtering, cleaning, grouping, merging |
| [03.Pandas.ipynb](03.Pandas.ipynb) | Continued Pandas work on the larger datasets - string methods, dtypes, `query`, `melt`/`pivot`, plotting, plus Titanic and Air Quality analysis |
| [04.Assignment.ipynb](04.Assignment.ipynb) | Solutions to the assignment problems - IRIS and Titanic |

Run them in order. `pandas_tutorial.ipynb` assumes nothing; `03.Pandas.ipynb` and
`04.Assignment.ipynb` assume both the tutorial and the two `.md` documents.

## Datasets

| File | Shape | Description |
| --- | --- | --- |
| `Titanic-Dataset.csv` | 891 x 12 | Passenger manifest. Binary target `Survived` (61.62 / 38.38). 177 missing ages, 687 missing cabins. The worked example for the full process. |
| `IRIS.csv` | 150 x 5 | Three iris species, perfectly balanced at 50 each. No missing values; 3 duplicate rows. |
| `globalAirQuality.csv` | 18000 x 15 | Hourly air quality for 50 cities across 38 countries over 16 days. Clean panel, **synthetic** - see the note below. |
| `raw_data.csv` | 11 x 6 | Small and deliberately dirty: 1 duplicate row, 7 null cells, a non-unique `id`. |
| `cleaned_data.csv` | 10 x 6 | Output of the cleaning in `03.Pandas.ipynb`. Kept as an example of flawed output - it still has a null and a bogus `income = 0`. |
| `employee_data.csv` | - | Small employee table, used for CSV/JSON I/O demos. |
| `employee_data.json` | - | Same data as JSON. |

### A note on the Air Quality data

`globalAirQuality.csv` is not real air quality data, and the documentation proves
it rather than assuming it:

- `aqi` correlates **0.556** with `pm25` and **0.532** with `no2`, as a composite
  AQI index should.
- `aqi` correlates **0.005** with `temperature`, **0.004** with `wind_speed`, and
  **-0.002** with `co`. In real data these are the strongest weather relationships
  that exist.
- Daily mean AQI over 16 days has a standard deviation of **0.47 on a mean of
  104.60** - a 0.45% coefficient of variation. Real air quality varies by 30-50%
  day to day.
- Every city's mean falls within a **5.99 point** band, from 101.38 to 107.37.
  Real global AQI ranges from roughly 25 to 220.

It is fine as a Pandas exercise - it exercises `resample`, `groupby`, datetime
indexing and plotting on a clean panel. It is not usable for any real modelling
claim. The analysis is written up in
[02.EDA.md section 4.8](02.EDA.md#48-correlation-is-not-everything).

## Setup

Requires **Python 3.14** and:

```
pandas 3.0.6
numpy 2.5.2
matplotlib 3.11.2
```

Those three cover everything in the notebooks and in the pandas/numpy parts of the
documentation. The modeling sections also reference `scikit-learn`, which is not
required to read them:

```
pip install scikit-learn      # Step 6, 7 in 01.Data Science Process.md
pip install scipy seaborn    # optional: scipy for chi-square, seaborn for charts
```

```bash
git clone https://github.com/md-zeon/Pandas.git
cd Pandas
pip install pandas numpy matplotlib
jupyter notebook
```

The notebooks read CSVs with relative paths, so **launch Jupyter from the
repository root** or the `read_csv` calls will fail.

## What the documentation emphasises

A few ideas run through both documents because they cause the most wasted work:

- **EDA is decision-making, not chart-making.** Its output is a numbered list of
  findings, each with evidence and a consequence - not a dashboard.
- **Order of operations is load-bearing.** Deduplicate *before* imputing. One
  duplicated row in `raw_data.csv` moves the mean age from 32.75 to 33.29.
  Impute *within* a group when the group's distribution differs.
- **Correlations must be checked within groups.** `petal_length` and `petal_width`
  correlate at 0.963 pooled, but at 0.306 within setosa and 0.322 within
  virginica. The pooled figure is almost entirely a species effect.
- **Outliers are flags, not errors.** Three Titanic passengers paid $512.33 -
  identical values in triplicate, all first class. Deleting them removes the
  richest passengers from the dataset.
- **Check correlations against domain knowledge.** That is how the synthetic Air
  Quality data gives itself away.
- **Always have a baseline.** Majority-class accuracy is 0.6162 on Titanic. A model
  that does not beat it has learned nothing.
- **End every cleaning function with an assertion.** `assert df.isna().sum().sum() == 0`
  would have caught the surviving null in `cleaned_data.csv`.
