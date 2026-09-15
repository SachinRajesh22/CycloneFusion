# CycloneFusion

CycloneFusion is a Streamlit prototype for historical cyclone twin search and explainable early-risk intelligence.

## Run locally

1. Place the provided dataset in this folder as:

   `dataset/odisha_cyclone_tracks_clean.csv`

2. Install dependencies:

   ```bash
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   .venv/bin/python -m streamlit run app.py
   ```

The app also supports uploading the CSV from the sidebar if the default file is not present. It will also fall back to `odisha_cyclone_tracks_clean.csv` or `odisha_cyclone_tracks_clean(1).csv` in the project root.

## Important

This is a decision-support prototype for demonstration. It compares the selected cyclone state with historical observations from the provided dataset and shows what happened next in those historical cases. It is not an official forecast, warning system, or replacement for IMD/government advisories.
