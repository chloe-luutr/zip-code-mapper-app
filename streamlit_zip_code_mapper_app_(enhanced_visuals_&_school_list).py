import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import math
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
from pathlib import Path # Import pathlib

st.set_page_config(layout="wide")
# MODIFIED: App title
st.title("School Roles & Ad ZIPs Map Generator")

st.markdown("""
This application generates a map visualizing school roles and advertisement target ZIP codes.
- The **US ZIP Codes Master File** is loaded automatically.
- **Upload a single CSV file** containing your ZIP code data. This file should include:
    - A `zip` column (mandatory).
    - Optionally, a `teachers` column for the number of teachers.
    - Optionally, a `TAs` column for the number of teaching assistants.
    (If `teachers` or `TAs` columns are missing or have no values for a ZIP, they will be treated as 0 for that ZIP).
The map will show:
- School locations with pie charts for `teachers` and `TAs` (if data provided).
- 5 and 10-mile coverage radii around locations with role data.
- All unique ZIPs from your file marked with serial numbers.
- An OpenStreetMap basemap with Latitude/Longitude grid.
""")

###############################################################################
# FILE PATHS FOR BUILT-IN DATA
###############################################################################
BASE_DIR = Path(__file__).resolve().parent
MASTER_ZIP_FILE_PATH = BASE_DIR / "us_zip_master.csv" # Assumes this file is in the repo

###############################################################################
# HELPER FUNCTIONS
###############################################################################

@st.cache_data
def load_us_zip_codes_from_repo(csv_file_path: Path) -> gpd.GeoDataFrame:
    """Loads US ZIP code data from a CSV file in the repository."""
    try:
        if not csv_file_path.is_file():
            st.error(f"Critical Error: US ZIP Codes Master File ('{csv_file_path.name}') not found at {csv_file_path}. Ensure it's in the GitHub repository.")
            return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")
        
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f: first_line = f.readline()
        delimiter = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
        
        df = pd.read_csv(csv_file_path, delimiter=delimiter, dtype={'zip': str, 'Zip Code': str})
        original_columns = list(df.columns); df.columns = df.columns.str.strip().str.lower()
        
        zip_col, lat_col, lon_col = None, None, None
        if 'zip' in df.columns: zip_col = 'zip'
        elif 'zip code' in df.columns: df.rename(columns={'zip code': 'zip'}, inplace=True); zip_col = 'zip'
        
        if 'latitude' in df.columns: lat_col = 'latitude'
        elif 'lat' in df.columns: lat_col = 'lat'
        if 'longitude' in df.columns: lon_col = 'longitude'
        elif 'lon' in df.columns: lon_col = 'lon'
        elif 'long' in df.columns: lon_col = 'long'

        if not (lat_col and lon_col) and 'geo point' in df.columns:
            try:
                lat_lon_split = df['geo point'].astype(str).str.split(',', expand=True)
                df['latitude_parsed'] = pd.to_numeric(lat_lon_split[0], errors='coerce')
                df['longitude_parsed'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
                lat_col, lon_col = 'latitude_parsed', 'longitude_parsed'
            except Exception: pass

        if not zip_col or not lat_col or not lon_col:
            st.error("US ZIP Master CSV: Could not identify 'zip', 'latitude', 'longitude' (or 'Geo Point').")
            return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

        df[zip_col] = df[zip_col].astype(str).str.strip().str.zfill(5)
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        df.dropna(subset=[lat_col, lon_col, zip_col], inplace=True)
        if df.empty: st.error("No valid coordinate/ZIP data in US ZIP Master CSV."); return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf[[zip_col, 'geometry']].rename(columns={zip_col: 'zip'})
    except Exception as e:
        st.error(f"Error loading US ZIP Codes Master File: {e}")
        return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

# MODIFIED: Function to load consolidated data from a single CSV upload
@st.cache_data
def load_map_data_from_upload(uploaded_file_object) -> pd.DataFrame:
    """
    Loads map data from a single uploaded CSV file.
    Expected columns: 'zip' (mandatory), 'teachers' (optional), 'TAs' (optional).
    """
    if uploaded_file_object is None: 
        return pd.DataFrame(columns=['zip', 'teachers', 'TAs']) # Return empty with expected columns
    try:
        uploaded_file_object.seek(0)
        try:
            first_lines_bytes = uploaded_file_object.read(1024)
            first_lines_str = first_lines_bytes.decode('utf-8-sig').splitlines()[0]
        except UnicodeDecodeError:
            uploaded_file_object.seek(0) # Reset seek if first decode failed
            first_lines_bytes = uploaded_file_object.read(1024)
            first_lines_str = first_lines_bytes.decode('latin1', errors='ignore').splitlines()[0]
        except IndexError: # Empty file
             st.warning("Uploaded CSV file appears to be empty.")
             return pd.DataFrame(columns=['zip', 'teachers', 'TAs'])


        delimiter = ';' if ';' in first_lines_str and first_lines_str.count(';') >= first_lines_str.count(',') else ','
        
        uploaded_file_object.seek(0)
        df = pd.read_csv(uploaded_file_object, delimiter=delimiter, encoding='utf-8-sig', encoding_errors='ignore')
        df.columns = df.columns.str.strip().str.lower()
        
        if 'zip' not in df.columns and 'zip code' not in df.columns:
            st.error("Uploaded CSV must contain a 'zip' or 'zip code' column.")
            return pd.DataFrame(columns=['zip', 'teachers', 'TAs'])
        if 'zip code' in df.columns and 'zip' not in df.columns:
            df.rename(columns={'zip code': 'zip'}, inplace=True)
            
        df['zip'] = df['zip'].astype(str).str.strip().str.zfill(5)

        # Handle optional 'teachers' and 'TAs' columns
        for col_name in ['teachers', 'TAs']:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)
            else:
                df[col_name] = 0 # Add column with zeros if it doesn't exist
        
        return df[['zip', 'teachers', 'TAs']].drop_duplicates(subset=['zip']) # Keep unique zips, sum roles later if needed
    
    except Exception as e:
        st.error(f"Error loading or processing uploaded CSV: {e}")
        return pd.DataFrame(columns=['zip', 'teachers', 'TAs'])


def geodesic_buffer_original(lon, lat, miles):
    radius_m = miles * 1609.34; wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    return transform(project_back, transform(project_fwd, Point(lon, lat)).buffer(radius_m))

def create_geodesic_buffers_for_schools_original(gdf_schools_data, radii=(5,10)):
    if gdf_schools_data.empty or 'geometry' not in gdf_schools_data.columns: return
    for r in radii:
        col_name = f"buffer_{r}"
        poly_list = [geodesic_buffer_original(row.geometry.x, row.geometry.y, r) if row.geometry and isinstance(row.geometry, Point) else None for _, row in gdf_schools_data.iterrows()]
        gdf_schools_data[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326")

def plot_pie_chart_original(ax, x_center, y_center, counts_dict, radius, role_colors):
    total = sum(counts_dict.values())
    if total <= 0 or radius <=0 : return
    
    items = sorted(counts_dict.items(), key=lambda item: item[0]) 
    values = [v for _, v in items]; 
    fracs = [v / total for v in values]
    
    min_angle_deg = 1 
    angles_deg = [max(f * 360, min_angle_deg if f > 0 else 0) for f in fracs]
    
    sum_angles_deg = sum(angles_deg)
    if sum_angles_deg > 360:
        angles_deg = [a * (360 / sum_angles_deg) for a in angles_deg]
        
    current_angle_start = 0
    for i, (role, value) in enumerate(items):
        if value > 0: 
            angle_extent = angles_deg[i]
            wedge = Wedge(center=(x_center, y_center), r=radius, 
                          theta1=current_angle_start, theta2=current_angle_start + angle_extent,
                          facecolor=role_colors.get(role, plt.cm.get_cmap('Greys')(0.5)), 
                          edgecolor='white', linewidth=0.5, alpha=0.85, zorder=10) 
            ax.add_patch(wedge)
            current_angle_start += angle_extent

###############################################################################
# MAIN PLOT FUNCTION
###############################################################################
# MODIFIED: Function signature to accept single df_map_data
def main_plot_from_original_script(gdf_us, df_map_data):
    gdf_us['zip'] = gdf_us['zip'].astype(str).str.zfill(5)
    
    # df_map_data contains 'zip', 'teachers', 'TAs'
    if df_map_data.empty or 'zip' not in df_map_data.columns:
        st.warning("No valid data from uploaded CSV to display.")
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No data to map", ha='center'); return fig
    
    df_map_data['zip'] = df_map_data['zip'].astype(str).str.zfill(5)

    # All unique ZIPs from the input are considered for serial numbering (Ad ZIPs)
    df_ads_data = df_map_data[['zip']].copy().drop_duplicates()
    # School data is the full map data
    df_schools_data = df_map_data.copy()

    relevant_zips = set(df_map_data['zip'].unique())
    
    if not relevant_zips: # Should be caught by df_map_data.empty earlier
        st.warning("No relevant ZIPs from uploaded data to display."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No ZIPs to map", ha='center'); return fig

    gdf_filtered = gdf_us[gdf_us['zip'].isin(relevant_zips)].copy()
    if gdf_filtered.empty:
        st.warning("None of the ZIPs from your file were found in the US ZIP master database."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "Input ZIPs not in master DB", ha='center'); return fig

    # Merge to get geometries
    gdf_ads_merged = pd.merge(df_ads_data, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
    gdf_schools_merged = pd.merge(df_schools_data, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])

    if not gdf_ads_merged.empty: 
        gdf_ads_merged = gpd.GeoDataFrame(gdf_ads_merged, geometry='geometry', crs="EPSG:4326")
    else: # Ensure it's an empty GeoDataFrame with geometry if no ad zips found/matched
        gdf_ads_merged = gpd.GeoDataFrame(columns=['zip', 'geometry'], geometry='geometry', crs="EPSG:4326")


    if not gdf_schools_merged.empty: 
        gdf_schools_merged = gpd.GeoDataFrame(gdf_schools_merged, geometry='geometry', crs="EPSG:4326")
    else: # Ensure it's an empty GeoDataFrame with geometry
        gdf_schools_merged = gpd.GeoDataFrame(columns=['zip', 'teachers', 'TAs', 'geometry'], geometry='geometry', crs="EPSG:4326")


    if gdf_schools_merged.empty and gdf_ads_merged.empty:
        st.warning("No geographic data could be matched for the ZIPs in your file."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No geodata for input ZIPs", ha='center'); return fig

    # Define teacher_cols based on fixed column names from the new input format
    teacher_cols = []
    if 'teachers' in gdf_schools_merged.columns and gdf_schools_merged['teachers'].sum() > 0:
        teacher_cols.append('teachers')
    if 'TAs' in gdf_schools_merged.columns and gdf_schools_merged['TAs'].sum() > 0:
        teacher_cols.append('TAs')
    
    if not teacher_cols and not gdf_schools_merged.empty : 
        st.info("No 'teachers' or 'TAs' counts found in the uploaded data for pie charts.")
    elif teacher_cols:
        st.info(f"Using role columns for pie charts: {', '.join(teacher_cols)}")


    if not gdf_schools_merged.empty and 'geometry' in gdf_schools_merged.columns and not gdf_schools_merged.geometry.is_empty.all() and any(gdf_schools_merged[col].sum() > 0 for col in ['teachers', 'TAs'] if col in gdf_schools_merged):
        create_geodesic_buffers_for_schools_original(gdf_schools_merged, radii=(5,10)) 

    gdf_ads_3857 = gdf_ads_merged.to_crs(epsg=3857) if not gdf_ads_merged.empty else \
                   gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs="EPSG:3857")
    gdf_schools_3857 = gdf_schools_merged.to_crs(epsg=3857) if not gdf_schools_merged.empty else \
                       gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs="EPSG:3857")
    gdf_filtered_3857 = gdf_filtered.to_crs(epsg=3857) if not gdf_filtered.empty else \
                        gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs="EPSG:3857")

    if not gdf_schools_merged.empty and not gdf_schools_3857.empty and 'geometry' in gdf_schools_merged.columns:
        if 'buffer_5' in gdf_schools_merged.columns and gdf_schools_merged['buffer_5'].notna().any():
            projected_buffer_5 = gpd.GeoSeries(gdf_schools_merged['buffer_5'][gdf_schools_merged['buffer_5'].notna()], crs="EPSG:4326").to_crs(epsg=3857)
            gdf_schools_3857.loc[projected_buffer_5.index, 'buffer_5_3857'] = projected_buffer_5

        if 'buffer_10' in gdf_schools_merged.columns and gdf_schools_merged['buffer_10'].notna().any():
            projected_buffer_10 = gpd.GeoSeries(gdf_schools_merged['buffer_10'][gdf_schools_merged['buffer_10'].notna()], crs="EPSG:4326").to_crs(epsg=3857)
            gdf_schools_3857.loc[projected_buffer_10.index, 'buffer_10_3857'] = projected_buffer_10

    combined_bounds_gdf = pd.concat([
        g for g in [gdf_filtered_3857, gdf_ads_3857, gdf_schools_3857] 
        if not g.empty and 'geometry' in g.columns and not g.geometry.is_empty.all()
    ])
    if combined_bounds_gdf.empty or combined_bounds_gdf.total_bounds is None or any(np.isnan(combined_bounds_gdf.total_bounds)):
        minx, miny, maxx, maxy = -14000000, 2800000, -7000000, 6300000 
    else: minx, miny, maxx, maxy = combined_bounds_gdf.total_bounds
    
    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6
    expand_factor = st.session_state.get('map_expand_factor_orig_v3', 1.5) 
    pad_x, pad_y = expand_factor * w * 0.1, expand_factor * h * 0.1

    min_jobs_val, max_jobs_val = float('inf'), 0
    if teacher_cols and not gdf_schools_merged.empty: 
        for _, row in gdf_schools_merged.iterrows(): 
            total = sum(pd.to_numeric(row.get(tc, 0), errors='coerce') or 0 for tc in teacher_cols)
            if total > max_jobs_val: max_jobs_val = total
            if total < min_jobs_val and total > 0: min_jobs_val = total
    if min_jobs_val == float('inf'): min_jobs_val = 0
    if max_jobs_val == 0 and min_jobs_val == 0: max_jobs_val = 1 

    BIGGEST_PIE_RADIUS_orig = st.session_state.get('pie_radius_scale_orig_v3', 3000.0)
    get_pie_radius_orig = lambda total_jobs: BIGGEST_PIE_RADIUS_orig * math.sqrt(max(0, total_jobs) / max_jobs_val) if max_jobs_val > 0 else 0

    fig, ax = plt.subplots(figsize=(12,10))
    
    role_color_map = {}
    if teacher_cols:
        # Use fixed colors for 'teachers' and 'TAs' for consistency
        palette = plt.cm.get_cmap('tab10')
        color_idx = 0
        if 'teachers' in teacher_cols:
            role_color_map['teachers'] = palette(color_idx)
            color_idx +=1
        if 'TAs' in teacher_cols:
            role_color_map['TAs'] = palette(color_idx)

    if not gdf_filtered_3857.empty and 'geometry' in gdf_filtered_3857.columns:
        ax.plot(gdf_filtered_3857.geometry.x, gdf_filtered_3857.geometry.y, 'o', color='lightgray', alpha=0.4, markersize=8, label="Contextual ZIPs (US Master)", zorder=1)
    
    zip_serial_map = {}
    if not df_ads_data.empty: # df_ads_data now comes from unique zips in uploaded file
        zip_serial_map = {str(zip_code).zfill(5): i+1 for i, zip_code in enumerate(df_ads_data['zip'].unique())} 
        
        if not gdf_ads_3857.empty and 'geometry' in gdf_ads_3857.columns: 
            gdf_ads_3857.plot(ax=ax, marker='s', color='green', markersize=40, label="Input ZIP Locations", zorder=3, edgecolor='darkgreen')
            
            # temp_ads_with_zip_proj is essentially gdf_ads_3857 if gdf_ads_merged was correctly formed
            for _, row_proj in gdf_ads_3857.iterrows(): # Iterate directly over gdf_ads_3857
                # gdf_ads_3857 was merged from df_ads_data which has 'zip'
                # Find the original ZIP for this projected geometry.
                # This requires gdf_ads_merged to have 'zip' before to_crs, which it does.
                original_zip_row = gdf_ads_merged[gdf_ads_merged.geometry == row_proj.geometry.buffer(1e-9).representative_point().buffer(1e-9)] # A way to match, might need improvement
                
                # Simpler: gdf_ads_3857 should have an index that matches gdf_ads_merged
                if row_proj.name in gdf_ads_merged.index:
                    original_zip = gdf_ads_merged.loc[row_proj.name, 'zip']
                    serial = zip_serial_map.get(original_zip)
                    if serial is not None and row_proj.geometry:
                        ax.text(row_proj.geometry.x, row_proj.geometry.y, str(serial), 
                                color='black', fontsize=7, ha='center', va='center', zorder=12,
                                bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round,pad=0.2'))
                # else: st.write(f"Could not find original zip for serial on row {row_proj.name}")


    if not gdf_schools_3857.empty and 'geometry' in gdf_schools_3857.columns:
        if 'buffer_5_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_5_3857'].notna().any():
            gdf_schools_3857['buffer_5_3857'][gdf_schools_3857['buffer_5_3857'].notna()].plot(
                ax=ax, edgecolor='red', facecolor='none', alpha=0.5, linewidth=1.0, zorder=4
            )
        if 'buffer_10_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_10_3857'].notna().any():
            gdf_schools_3857['buffer_10_3857'][gdf_schools_3857['buffer_10_3857'].notna()].plot(
                ax=ax, edgecolor='orange', facecolor='none', alpha=0.6, linewidth=1.5, zorder=3 
            )

    if teacher_cols and not gdf_schools_3857.empty and 'geometry' in gdf_schools_3857.columns:
        for idx, row_proj in gdf_schools_3857.iterrows(): 
            if row_proj.geometry is None or row_proj.geometry.is_empty: continue
            
            if idx in gdf_schools_merged.index:
                original_row = gdf_schools_merged.loc[idx]
                counts_dict = {}
                if 'teachers' in teacher_cols and original_row.get('teachers', 0) > 0:
                    counts_dict['teachers'] = original_row['teachers']
                if 'TAs' in teacher_cols and original_row.get('TAs', 0) > 0:
                    counts_dict['TAs'] = original_row['TAs']
                
                if counts_dict: 
                    total_jobs_at_school = sum(counts_dict.values())
                    r_pie = get_pie_radius_orig(total_jobs_at_school)
                    
                    if r_pie > 0: 
                        plot_pie_chart_original(ax, row_proj.geometry.x, row_proj.geometry.y, counts_dict, r_pie, role_color_map)
                    elif total_jobs_at_school > 0 : 
                         ax.plot(row_proj.geometry.x, row_proj.geometry.y, marker='P', color='darkviolet', markersize=30, alpha=0.7, zorder=5, markeredgecolor='black')
                # If no counts_dict, it means no roles > 0, so it won't plot a pie. Marker for ad_zip already handled.
            # else: st.write(f"idx {idx} not in gdf_schools_merged for pie chart")
                 
    elif not gdf_schools_3857.empty and not teacher_cols and 'geometry' in gdf_schools_3857.columns : 
         # This case is less likely now with fixed teacher_cols, but as a fallback if no roles defined
         gdf_schools_3857.plot(ax=ax, marker='P', color='darkviolet', markersize=60, label="Input ZIPs (No Role Data)", zorder=5, alpha=0.8, edgecolor='black')

    try: 
        basemap_crs = "EPSG:3857" 
        if not gdf_filtered_3857.empty and 'geometry' in gdf_filtered_3857.columns and gdf_filtered_3857.crs:
            basemap_crs = gdf_filtered_3857.crs.to_string()
        elif not combined_bounds_gdf.empty and 'geometry' in combined_bounds_gdf.columns and combined_bounds_gdf.crs:
             basemap_crs = combined_bounds_gdf.crs.to_string()

        ctx.add_basemap(ax, crs=basemap_crs, 
                        source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=6)
    except Exception as e: st.warning(f"Could not add basemap: {e}")

    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
    
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    num_xticks = st.session_state.get('num_grid_ticks_orig_v3', 10)
    num_yticks = st.session_state.get('num_grid_ticks_orig_v3', 10) 
    
    plot_xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=num_xticks)
    plot_yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=num_yticks)
    
    valid_y_for_xtick_transform = ax.get_ylim()[0] if not np.isnan(ax.get_ylim()[0]) else 0
    valid_x_for_ytick_transform = ax.get_xlim()[0] if not np.isnan(ax.get_xlim()[0]) else 0

    xticks_latlon = [transformer.transform(x, valid_y_for_xtick_transform)[0] for x in plot_xticks] 
    yticks_latlon = [transformer.transform(valid_x_for_ytick_transform, y)[1] for y in plot_yticks] 
    
    ax.set_xticks(plot_xticks); ax.set_xticklabels([f"{lon:.2f}°" for lon in xticks_latlon], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(plot_yticks); ax.set_yticklabels([f"{lat:.2f}°" for lat in yticks_latlon], fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')

    handles, labels = [], []
    if not gdf_filtered_3857.empty and 'geometry' in gdf_filtered_3857.columns and not gdf_filtered_3857.geometry.is_empty.all():
        handles.append(mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', markersize=5, alpha=0.4)); labels.append('Contextual ZIPs (US Master)')
    if not gdf_ads_3857.empty and 'geometry' in gdf_ads_3857.columns and not gdf_ads_3857.geometry.is_empty.all(): # Changed label
        handles.append(mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=7, markeredgecolor='darkgreen')); labels.append('Input ZIP Locations')
    
    # Buffers only if schools with roles exist
    if not gdf_schools_3857.empty and 'geometry' in gdf_schools_3857.columns and not gdf_schools_3857.geometry.is_empty.all() and teacher_cols:
        if 'buffer_5_3857' in gdf_schools_3857 and gdf_schools_3857['buffer_5_3857'].notna().any(): 
            handles.append(mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.0, alpha=0.5)); labels.append('5-mile Role Coverage')
        if 'buffer_10_3857' in gdf_schools_3857 and gdf_schools_3857['buffer_10_3857'].notna().any(): 
            handles.append(mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, alpha=0.6)); labels.append('10-mile Role Coverage')
    
    if teacher_cols and not gdf_schools_merged.empty: 
        if 'teachers' in role_color_map:
            handles.append(mpatches.Patch(color=role_color_map['teachers'], label='Teachers'))
        if 'TAs' in role_color_map:
            handles.append(mpatches.Patch(color=role_color_map['TAs'], label='TAs'))
            
    current_handles_ax, current_labels_ax = ax.get_legend_handles_labels()
    final_legend_items = {}
    for handle, label in zip(current_handles_ax, current_labels_ax): # Add auto-generated legend items first
        if label not in final_legend_items: final_legend_items[label] = handle
    for handle, label in zip(handles, labels):  # Add manually created ones, avoid duplicates
         if label not in final_legend_items: final_legend_items[label] = handle
            
    if final_legend_items:
        ax.legend(final_legend_items.values(), final_legend_items.keys(), 
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., 
                  fontsize='small', title="Legend", title_fontsize="medium")

    if teacher_cols and not gdf_schools_merged.empty:
         ax.text(1.02, 0.5 if len(final_legend_items) < 8 else 0.2, 
                f"Role Range (per ZIP):\nMin Roles: {min_jobs_val if min_jobs_val != float('inf') else 'N/A'}\nMax Roles: {max_jobs_val if max_jobs_val > 0 else 'N/A'}",
                transform=ax.transAxes, va='top', fontsize='small',
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_title("School Roles & Ad ZIPs Map\nLat/Lon Grid, Legend & Role Range on Right", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) 
    return fig

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
# MODIFIED: UI for single CSV input
st.sidebar.header("Map Data Input")
st.sidebar.markdown("Upload a single CSV file with your ZIP code data. The map will be generated based on this file.")

uploaded_map_data_file = st.sidebar.file_uploader(
    "CSV File: Must contain a 'zip' (or 'zip code') column. Optionally include 'teachers' and 'TAs' columns for role counts.", 
    type="csv", 
    key="map_data_upload_v1", # New key for the single uploader
    help="Columns should be: zip (text), teachers (numeric, optional), TAs (numeric, optional). All unique ZIPs will get serial numbers. Roles data will generate pie charts and coverage radii."
)

st.sidebar.header("Map Display Options")
if 'map_expand_factor_orig_v3' not in st.session_state: st.session_state.map_expand_factor_orig_v3 = 1.5
st.session_state.map_expand_factor_orig_v3 = st.sidebar.slider("Map Zoom/Expand Factor:", min_value=0.5, max_value=5.0, value=st.session_state.map_expand_factor_orig_v3, step=0.1, key="map_expand_slider_orig_v3")

if 'pie_radius_scale_orig_v3' not in st.session_state: st.session_state.pie_radius_scale_orig_v3 = 3000.0
st.session_state.pie_radius_scale_orig_v3 = st.sidebar.slider("Pie Chart Max Radius Scale (map units):", min_value=500.0, max_value=10000.0, value=st.session_state.pie_radius_scale_orig_v3, step=100.0, key="pie_scale_slider_orig_v3")

if 'num_grid_ticks_orig_v3' not in st.session_state: st.session_state.num_grid_ticks_orig_v3 = 10
st.session_state.num_grid_ticks_orig_v3 = st.sidebar.slider("Number of Lat/Lon Grid Ticks:", min_value=3, max_value=30, value=st.session_state.num_grid_ticks_orig_v3, step=1, key="grid_ticks_slider_orig_v3")

gdf_us_data = load_us_zip_codes_from_repo(MASTER_ZIP_FILE_PATH)

if gdf_us_data.empty:
    st.error("ERROR: Could not load US ZIP Codes Master File from repository. App cannot proceed. Ensure 'us_zip_master.csv' is in the GitHub repository and correctly formatted.")
    st.stop()

# MODIFIED: Main app logic for single file input
if uploaded_map_data_file: 
    
    df_map_data_loaded = load_map_data_from_upload(uploaded_map_data_file)

    if not gdf_us_data.empty and not df_map_data_loaded.empty :
        st.info("Input data processed. Generating map...")
        try:
            # Pass df_map_data_loaded directly
            map_figure = main_plot_from_original_script(gdf_us_data, df_map_data_loaded)
            st.pyplot(map_figure)
            st.success("Map generated successfully!")
            
            fn = 'school_ad_zip_map.png' # Generic name now
            img_bytes = io.BytesIO()
            map_figure.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
            img_bytes.seek(0)
            st.download_button(label="Download Map as PNG", data=img_bytes, file_name=fn, mime="image/png")
        except Exception as e: 
            st.error(f"Error during map generation: {e}")
            st.exception(e) 
    elif gdf_us_data.empty: 
        st.error("Cannot generate map because US ZIP Code master data failed to load.")
    elif df_map_data_loaded.empty and uploaded_map_data_file: # File was uploaded but processed to empty
        st.warning("Uploaded CSV file did not contain valid data or could not be processed. Please check the file format and content.")
    else: # No file uploaded yet, or other unhandled case
         st.warning("Map could not be generated. Please upload a CSV file with ZIP code data.")
else:
    st.sidebar.info("To generate a map, please upload a CSV file with ZIP code data using the option above.")
    st.info("Awaiting data input via CSV upload...")

st.markdown("---")
st.markdown("Streamlit app for visualizing school roles and ad ZIPs based on original mapping script logic.")
