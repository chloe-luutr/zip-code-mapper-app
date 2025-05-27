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
st.title("School Roles & Ad ZIPs Map Generator (Replicates combined_map.png)")

st.markdown("""
This app replicates the functionality of the original `zip-code-maper.py` script.
- The **US ZIP Codes Master File** is loaded automatically from the app's repository.
- **Paste your Ad Target ZIPs** into the text box.
- **Upload your School Open Roles CSV file** to generate the map.
The map will show:
- School locations with pie charts representing open roles.
- 5 and 10-mile coverage radii (lines) around schools.
- Ad ZIPs with serial numbers.
- An OpenStreetMap basemap with Latitude/Longitude grid.
""")

###############################################################################
# FILE PATHS FOR BUILT-IN DATA
###############################################################################
BASE_DIR = Path(__file__).resolve().parent
MASTER_ZIP_FILE_PATH = BASE_DIR / "us_zip_master.csv" # Assumes this file is in the repo

###############################################################################
# HELPER FUNCTIONS (Adapted from zip-code-maper.py)
###############################################################################

@st.cache_data # Cache data loading
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

def parse_ad_target_zips_from_text(zip_code_text_input: str) -> pd.DataFrame:
    """Parses a list of Ad Target ZIP codes from a text input area."""
    if not zip_code_text_input.strip(): return pd.DataFrame(columns=['zip'])
    zips = [z.strip() for z in pd.Series(zip_code_text_input.splitlines()).str.split(r'[\s,]+').explode() if z.strip().isdigit()]
    valid_zips = [z.zfill(5) for z in zips if len(z.zfill(5)) == 5]
    if not valid_zips: st.warning("No valid 5-digit Ad Target ZIP codes found in input text."); return pd.DataFrame(columns=['zip'])
    return pd.DataFrame(list(set(valid_zips)), columns=['zip']).drop_duplicates(subset=['zip'])


@st.cache_data
def load_school_requests_from_upload(uploaded_file_object) -> pd.DataFrame:
    if uploaded_file_object is None: return pd.DataFrame()
    try:
        uploaded_file_object.seek(0); first_lines_bytes = uploaded_file_object.read(1024)
        uploaded_file_object.seek(0); 
        # Handle potential decoding errors more gracefully for the delimiter check
        try:
            first_lines_str = first_lines_bytes.decode('utf-8-sig').splitlines()[0]
        except UnicodeDecodeError:
            first_lines_str = first_lines_bytes.decode('latin1', errors='ignore').splitlines()[0] # Fallback encoding

        delimiter = ';' if ';' in first_lines_str and first_lines_str.count(';') >= first_lines_str.count(',') else ','
        
        # Reset seek for pd.read_csv
        uploaded_file_object.seek(0)
        df = pd.read_csv(uploaded_file_object, delimiter=delimiter, encoding='utf-8-sig', encoding_errors='ignore') # Add encoding_errors
        df.columns = df.columns.str.strip().str.lower()
        
        # Try to find zip column more flexibly
        zip_col_found = None
        if 'zip' in df.columns:
            zip_col_found = 'zip'
        elif 'zip code' in df.columns:
            zip_col_found = 'zip code'
            df.rename(columns={'zip code': 'zip'}, inplace=True)
        
        if not zip_col_found:
            st.error("School Open Roles CSV must contain a 'zip' or 'zip code' column.")
            return pd.DataFrame()
            
        df['zip'] = df['zip'].astype(str).str.strip().str.zfill(5)
        return df
    except Exception as e:
        st.error(f"Error loading School Open Roles: {e}")
        return pd.DataFrame()

def geodesic_buffer_original(lon, lat, miles):
    radius_m = miles * 1609.34; wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    return transform(project_back, transform(project_fwd, Point(lon, lat)).buffer(radius_m))

def create_geodesic_buffers_for_schools_original(gdf_schools_data, radii=(5,10)): # Renamed param for clarity
    """ Creates geodesic buffers and adds them as columns to gdf_schools_data. """
    if gdf_schools_data.empty or 'geometry' not in gdf_schools_data.columns: return
    for r in radii:
        col_name = f"buffer_{r}"
        poly_list = [geodesic_buffer_original(row.geometry.x, row.geometry.y, r) if row.geometry and isinstance(row.geometry, Point) else None for _, row in gdf_schools_data.iterrows()]
        gdf_schools_data[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326") # Modifies the passed GeoDataFrame in place

def plot_pie_chart_original(ax, x_center, y_center, counts_dict, radius, role_colors):
    total = sum(counts_dict.values())
    if total <= 0 or radius <=0 : return # Guard clause for no data or zero radius
    
    items = sorted(counts_dict.items(), key=lambda item: item[0]) # Sort by role name for consistent color assignment
    values = [v for _, v in items]; 
    fracs = [v / total for v in values]
    
    # Ensure even tiny slices get a minimum angle to be visible
    min_angle_deg = 1 # Minimum angle in degrees for a slice
    angles_deg = [max(f * 360, min_angle_deg if f > 0 else 0) for f in fracs]
    
    # Normalize angles if they exceed 360 due to min_angle_deg enforcement
    sum_angles_deg = sum(angles_deg)
    if sum_angles_deg > 360:
        angles_deg = [a * (360 / sum_angles_deg) for a in angles_deg]
        
    current_angle_start = 0
    for i, (role, value) in enumerate(items):
        if value > 0: # Only plot wedges for roles with counts
            angle_extent = angles_deg[i]
            wedge = Wedge(center=(x_center, y_center), r=radius, 
                          theta1=current_angle_start, theta2=current_angle_start + angle_extent,
                          facecolor=role_colors.get(role, plt.cm.get_cmap('Greys')(0.5)), # Fallback color
                          edgecolor='white', linewidth=0.5, alpha=0.85, zorder=10) # Added zorder to ensure pies are on top
            ax.add_patch(wedge)
            current_angle_start += angle_extent

###############################################################################
# MAIN PLOT FUNCTION (Adapted from zip-code-maper.py)
###############################################################################
def main_plot_from_original_script(gdf_us, df_ads, df_schools):
    gdf_us['zip'] = gdf_us['zip'].astype(str).str.zfill(5)
    if not df_ads.empty: df_ads['zip'] = df_ads['zip'].astype(str).str.zfill(5)
    if not df_schools.empty: df_schools['zip'] = df_schools['zip'].astype(str).str.zfill(5)

    relevant_zips = set()
    if not df_ads.empty: relevant_zips.update(df_ads['zip'].unique())
    if not df_schools.empty: relevant_zips.update(df_schools['zip'].unique())
    
    if not relevant_zips:
        st.warning("No relevant ZIPs from Ad or School data to display."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No ZIPs to map", ha='center'); return fig

    gdf_filtered = gdf_us[gdf_us['zip'].isin(relevant_zips)].copy()
    if gdf_filtered.empty:
        st.warning("None of the Ad/School ZIPs found in US ZIP master."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "ZIPs not in master", ha='center'); return fig

    # Merge to get geometries for Ad and School ZIPs
    gdf_ads_merged = pd.merge(df_ads, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry']) if not df_ads.empty else gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")
    gdf_schools_merged = pd.merge(df_schools, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry']) if not df_schools.empty else gpd.GeoDataFrame(columns=['zip', 'geometry'] + list(df_schools.columns.drop('zip', errors='ignore')), crs="EPSG:4326") # Preserve other school columns

    if not gdf_ads_merged.empty: gdf_ads_merged = gpd.GeoDataFrame(gdf_ads_merged, geometry='geometry', crs="EPSG:4326")
    if not gdf_schools_merged.empty: gdf_schools_merged = gpd.GeoDataFrame(gdf_schools_merged, geometry='geometry', crs="EPSG:4326")


    if gdf_schools_merged.empty and gdf_ads_merged.empty:
        st.warning("No geodata for Ad/School ZIPs after merge."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No geodata for Ad/School ZIPs", ha='center'); return fig

    teacher_cols = []
    if not gdf_schools_merged.empty:
        # Identify numeric columns in the original df_schools that are not 'zip' for roles
        # This assumes roles are numeric counts.
        potential_role_cols = df_schools.select_dtypes(include=np.number).columns
        teacher_cols = [c for c in potential_role_cols if c.lower().strip() != 'zip']
        
        if not teacher_cols: 
            st.info("No numeric columns (excluding 'zip') identified as 'roles' for pie charts in School Open Roles data.")
        else: 
            st.info(f"Identified role columns for pie charts: {', '.join(teacher_cols)}")
            # Ensure these columns are present in gdf_schools_merged
            for tc in teacher_cols:
                if tc not in gdf_schools_merged.columns:
                    # This case should ideally not happen if merge is correct
                    st.warning(f"Role column '{tc}' not found in merged school data. Pie charts for this role might be affected.")


    # Create buffers on gdf_schools_merged (which has WGS84 geometries)
    if not gdf_schools_merged.empty:
        create_geodesic_buffers_for_schools_original(gdf_schools_merged, radii=(5,10)) # Modifies gdf_schools_merged

    # Project to Web Mercator (EPSG:3857) for plotting with contextily
    gdf_ads_3857      = gdf_ads_merged.to_crs(epsg=3857) if not gdf_ads_merged.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_schools_3857  = gdf_schools_merged.to_crs(epsg=3857) if not gdf_schools_merged.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_filtered_3857 = gdf_filtered.to_crs(epsg=3857)

    # Add projected buffers to gdf_schools_3857 from gdf_schools_merged
    # This ensures indices align if gdf_schools_merged was the source for gdf_schools_3857
    if not gdf_schools_merged.empty and not gdf_schools_3857.empty:
        if 'buffer_5' in gdf_schools_merged.columns:
            # Project the GeoSeries and assign, ensuring index alignment
            projected_buffer_5 = gpd.GeoSeries(gdf_schools_merged['buffer_5'], crs="EPSG:4326").to_crs(epsg=3857)
            gdf_schools_3857['buffer_5_3857'] = projected_buffer_5.set_axis(gdf_schools_3857.index, axis='index', copy=False)

        if 'buffer_10' in gdf_schools_merged.columns:
            projected_buffer_10 = gpd.GeoSeries(gdf_schools_merged['buffer_10'], crs="EPSG:4326").to_crs(epsg=3857)
            gdf_schools_3857['buffer_10_3857'] = projected_buffer_10.set_axis(gdf_schools_3857.index, axis='index', copy=False)


    # Determine map bounds
    combined_bounds_gdf = pd.concat([g for g in [gdf_filtered_3857, gdf_ads_3857, gdf_schools_3857] if not g.empty and 'geometry' in g.columns and not g.geometry.is_empty.all()])
    if combined_bounds_gdf.empty or combined_bounds_gdf.total_bounds is None or any(np.isnan(combined_bounds_gdf.total_bounds)):
        minx, miny, maxx, maxy = -14000000, 2800000, -7000000, 6300000 # Default fallback bounds
    else: minx, miny, maxx, maxy = combined_bounds_gdf.total_bounds
    
    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6
    expand_factor = st.session_state.get('map_expand_factor_orig_v3', 1.5) 
    pad_x, pad_y = expand_factor * w * 0.1, expand_factor * h * 0.1

    # Calculate min/max jobs for pie chart scaling
    min_jobs_val, max_jobs_val = float('inf'), 0
    if teacher_cols and not gdf_schools_merged.empty: # Use gdf_schools_merged for role data
        for _, row in gdf_schools_merged.iterrows(): 
            total = sum(pd.to_numeric(row.get(tc, 0), errors='coerce') or 0 for tc in teacher_cols)
            if total > max_jobs_val: max_jobs_val = total
            if total < min_jobs_val and total > 0: min_jobs_val = total
    if min_jobs_val == float('inf'): min_jobs_val = 0
    if max_jobs_val == 0 and min_jobs_val == 0: max_jobs_val = 1 # Avoid division by zero if no jobs

    BIGGEST_PIE_RADIUS_orig = st.session_state.get('pie_radius_scale_orig_v3', 3000.0)
    get_pie_radius_orig = lambda total_jobs: BIGGEST_PIE_RADIUS_orig * math.sqrt(max(0, total_jobs) / max_jobs_val) if max_jobs_val > 0 else 0

    fig, ax = plt.subplots(figsize=(12,10))
    
    # Define role colors
    role_color_map = {}
    if teacher_cols:
        # Sort teacher_cols to ensure consistent color mapping if order changes
        sorted_teacher_cols = sorted(list(set(teacher_cols)))
        palette = plt.cm.get_cmap('tab10', len(sorted_teacher_cols)) if len(sorted_teacher_cols) <= 10 else plt.cm.get_cmap('tab20', len(sorted_teacher_cols))
        for i, role in enumerate(sorted_teacher_cols): 
            role_color_map[role] = palette(i)

    # 1. Plot contextual ZIPs (background)
    if not gdf_filtered_3857.empty:
        ax.plot(gdf_filtered_3857.geometry.x, gdf_filtered_3857.geometry.y, 'o', color='lightgray', alpha=0.4, markersize=8, label="Contextual ZIPs", zorder=1)
    
    # 2. Plot Ad ZIPs and their serial numbers
    zip_serial_map = {}
    if not df_ads.empty: 
        # Create serial map from the original df_ads to maintain consistent numbering
        zip_serial_map = {str(zip_code).zfill(5): i+1 for i, zip_code in enumerate(df_ads['zip'].unique())} # Use unique zips from input
        
        if not gdf_ads_3857.empty: # gdf_ads_3857 contains geometry
            # Plot Ad ZIP markers
            gdf_ads_3857.plot(ax=ax, marker='s', color='green', markersize=40, label="Ad ZIPs", zorder=3, edgecolor='darkgreen')
            # Add serial numbers
            # We need to map ZIPs from gdf_ads_3857 back to their original form if necessary, or use gdf_ads_merged
            temp_ads_with_zip_proj = gdf_ads_merged[['zip', 'geometry']].to_crs(epsg=3857) if not gdf_ads_merged.empty else gpd.GeoDataFrame()

            if not temp_ads_with_zip_proj.empty:
                for _, row_proj in temp_ads_with_zip_proj.iterrows():
                    serial = zip_serial_map.get(row_proj['zip'])
                    if serial is not None and row_proj.geometry:
                        ax.text(row_proj.geometry.x, row_proj.geometry.y, str(serial), color='black', fontsize=7, ha='center', va='center', zorder=5,
                                bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2'))


    # 3. Plot school coverage buffers
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_5_3857'].notna().any():
            # Plot the GeoSeries of buffer geometries
            gdf_schools_3857['buffer_5_3857'][gdf_schools_3857['buffer_5_3857'].notna()].plot(
                ax=ax, edgecolor='red', facecolor='none', alpha=0.5, linewidth=1.0, zorder=4
            )
        if 'buffer_10_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_10_3857'].notna().any():
            gdf_schools_3857['buffer_10_3857'][gdf_schools_3857['buffer_10_3857'].notna()].plot(
                ax=ax, edgecolor='orange', facecolor='none', alpha=0.6, linewidth=1.5, zorder=3 # zorder 3 so 5-mile is on top
            )

    # 4. Plot school locations: Pie charts or markers
    if teacher_cols and not gdf_schools_3857.empty:
        # Iterate through projected school locations for plotting
        for idx, row_proj in gdf_schools_3857.iterrows(): 
            if row_proj.geometry is None or row_proj.geometry.is_empty: continue
            
            # Get original data for roles from gdf_schools_merged using the index
            if idx in gdf_schools_merged.index:
                original_row = gdf_schools_merged.loc[idx]
                counts_dict = {tc: pd.to_numeric(original_row.get(tc, 0), errors='coerce') or 0 for tc in teacher_cols}
                counts_dict = {k: v for k, v in counts_dict.items() if v > 0} # Only include roles with counts > 0
                
                if counts_dict: # If there are any roles with counts
                    total_jobs_at_school = sum(counts_dict.values())
                    r_pie = get_pie_radius_orig(total_jobs_at_school)
                    
                    # MODIFIED: Plot pie if radius is greater than 0 (was > 500)
                    if r_pie > 0: 
                        plot_pie_chart_original(ax, row_proj.geometry.x, row_proj.geometry.y, counts_dict, r_pie, role_color_map)
                    elif total_jobs_at_school > 0 : # If jobs exist but pie is too small, plot a small marker
                         ax.plot(row_proj.geometry.x, row_proj.geometry.y, marker='P', color='darkviolet', markersize=30, alpha=0.7, zorder=5, markeredgecolor='black')
                else: # No roles with counts > 0, plot a default marker
                    ax.plot(row_proj.geometry.x, row_proj.geometry.y, marker='P', color='gray', markersize=30, alpha=0.7, zorder=5, markeredgecolor='black', label="School (No Role Data)" if "School (No Role Data)" not in ax.get_legend_handles_labels()[1] else "")
            else: # Should not happen if indexing is correct
                 ax.plot(row_proj.geometry.x, row_proj.geometry.y, marker='X', color='black', markersize=30, zorder=5) # Error marker
                 
    elif not gdf_schools_3857.empty: # No teacher_cols identified, plot default markers for all schools
         gdf_schools_3857.plot(ax=ax, marker='P', color='darkviolet', markersize=60, label="School Locations (No Role Data)", zorder=5, alpha=0.8, edgecolor='black')

    # 5. Add basemap
    try: 
        # Ensure CRS is correctly passed as string for contextily
        # MODIFIED: Removed attribution_position or set to a standard value like ctx.Place.BOTTOM_RIGHT
        # If ctx.Place is not available, removing attribution_position is safest.
        # For now, let's try removing it to use the default.
        ctx.add_basemap(ax, crs=gdf_filtered_3857.crs.to_string() if not gdf_filtered_3857.empty else "EPSG:3857", 
                        source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=6)
    except Exception as e: st.warning(f"Could not add basemap: {e}")

    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
    
    # Configure Lat/Lon grid
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    num_xticks = st.session_state.get('num_grid_ticks_orig_v3', 10)
    num_yticks = st.session_state.get('num_grid_ticks_orig_v3', 10) # Can be different if desired
    
    plot_xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=num_xticks)
    plot_yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=num_yticks)
    
    # Ensure transformation doesn't fail for edge cases by providing valid y for x-ticks and x for y-ticks
    valid_y_for_xtick_transform = ax.get_ylim()[0] if not np.isnan(ax.get_ylim()[0]) else 0
    valid_x_for_ytick_transform = ax.get_xlim()[0] if not np.isnan(ax.get_xlim()[0]) else 0

    xticks_latlon = [transformer.transform(x, valid_y_for_xtick_transform)[0] for x in plot_xticks] 
    yticks_latlon = [transformer.transform(valid_x_for_ytick_transform, y)[1] for y in plot_yticks] 
    
    ax.set_xticks(plot_xticks); ax.set_xticklabels([f"{lon:.2f}°" for lon in xticks_latlon], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(plot_yticks); ax.set_yticklabels([f"{lat:.2f}°" for lat in yticks_latlon], fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')

    # Create legend
    handles, labels = [], []
    # Contextual ZIPs
    if not gdf_filtered_3857.empty :
        handles.append(mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', markersize=5, alpha=0.4)); labels.append('Contextual ZIPs')
    # Ad ZIPs
    if not gdf_ads_3857.empty:
        handles.append(mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=7, markeredgecolor='darkgreen')); labels.append('Ad ZIPs')
    # School Buffers
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857 and gdf_schools_3857['buffer_5_3857'].notna().any(): 
            handles.append(mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.0, alpha=0.5)); labels.append('5-mile School Coverage')
        if 'buffer_10_3857' in gdf_schools_3857 and gdf_schools_3857['buffer_10_3857'].notna().any(): 
            handles.append(mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, alpha=0.6)); labels.append('10-mile School Coverage')
    
    # School Roles (from pie charts) or default school marker
    if teacher_cols and not gdf_schools_merged.empty: # Use merged for checking if teacher_cols applies
        for role, color in sorted(role_color_map.items()): # Sort for consistent legend order
            handles.append(mpatches.Patch(color=color, label=role.replace('_', ' ').title()))
            # labels.append(role.replace('_', ' ').title()) # Label already included in Patch
    elif not gdf_schools_3857.empty and not teacher_cols: # Default school marker if no roles
        handles.append(mlines.Line2D([], [], color='darkviolet', marker='P', linestyle='None', markersize=8, label='School Locations')); 
        # labels.append('School Locations')
    elif not gdf_schools_3857.empty and any(row.get('total_jobs',0)==0 for _, row in gdf_schools_merged.iterrows()) : # For schools with no role data plotted as gray
        if not any("School (No Role Data)" in lab for lab in labels):
             handles.append(mlines.Line2D([], [], color='gray', marker='P', linestyle='None', markersize=8, label='School (No Role Data)'));
             # labels.append('School (No Role Data)')


    # Combine handles and labels, removing duplicates by label
    # Get existing handles/labels from plot elements like gdf_ads_3857.plot()
    current_handles_ax, current_labels_ax = ax.get_legend_handles_labels()
    
    final_legend_items = {}
    for handle, label in zip(current_handles_ax, current_labels_ax):
        if label not in final_legend_items: final_legend_items[label] = handle
    for handle, label in zip(handles, labels): # For manually created legend items
         if label not in final_legend_items: final_legend_items[label] = handle
            
    if final_legend_items:
        ax.legend(final_legend_items.values(), final_legend_items.keys(), 
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., 
                  fontsize='small', title="Legend", title_fontsize="medium")

    # Add Job Range text
    if teacher_cols and not gdf_schools_merged.empty:
         ax.text(1.02, 0.5 if len(final_legend_items) < 8 else 0.2, # Adjust y position based on legend height
                f"School Job Range:\nMin Roles: {min_jobs_val if min_jobs_val != float('inf') else 'N/A'}\nMax Roles: {max_jobs_val if max_jobs_val > 0 else 'N/A'}",
                transform=ax.transAxes, va='top', fontsize='small',
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_title("Schools (Pie Charts + Coverage), Ad ZIPs, OSM Basemap\nLat/Lon Grid, Legend & Job Range on Right", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust rect to ensure legend fits
    return fig
    # --- End of main_plot_from_original_script ---

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
st.sidebar.header("1. Paste Ad Target ZIP Codes")
ad_target_zips_text = st.sidebar.text_area(
    "Paste Ad Target ZIPs (for serial numbers on map, comma/space/newline separated):", 
    height=100, 
    key="ad_zips_text_v3"
)

st.sidebar.header("2. Upload School Open Roles File (CSV)")
uploaded_school_requests_file = st.sidebar.file_uploader(
    "School Open Roles File (must contain 'zip' or 'zip code' and numeric role count columns e.g., 'TA', 'Teacher'):", 
    type="csv", 
    key="school_requests_orig_v3"
)

st.sidebar.header("Map Display Options")
# Session state for sliders to remember values
if 'map_expand_factor_orig_v3' not in st.session_state: st.session_state.map_expand_factor_orig_v3 = 1.5
st.session_state.map_expand_factor_orig_v3 = st.sidebar.slider("Map Zoom/Expand Factor:", min_value=0.5, max_value=5.0, value=st.session_state.map_expand_factor_orig_v3, step=0.1, key="map_expand_slider_orig_v3")

if 'pie_radius_scale_orig_v3' not in st.session_state: st.session_state.pie_radius_scale_orig_v3 = 3000.0
st.session_state.pie_radius_scale_orig_v3 = st.sidebar.slider("Pie Chart Max Radius Scale (map units):", min_value=500.0, max_value=10000.0, value=st.session_state.pie_radius_scale_orig_v3, step=100.0, key="pie_scale_slider_orig_v3")

if 'num_grid_ticks_orig_v3' not in st.session_state: st.session_state.num_grid_ticks_orig_v3 = 10
st.session_state.num_grid_ticks_orig_v3 = st.sidebar.slider("Number of Lat/Lon Grid Ticks:", min_value=3, max_value=30, value=st.session_state.num_grid_ticks_orig_v3, step=1, key="grid_ticks_slider_orig_v3")

# Load master US ZIP data automatically from repository
gdf_us_data = load_us_zip_codes_from_repo(MASTER_ZIP_FILE_PATH)

if gdf_us_data.empty:
    st.error("ERROR: Could not load US ZIP Codes Master File from repository. App cannot proceed. Ensure 'us_zip_master.csv' is in the GitHub repository and correctly formatted.")
    st.stop()

# Main app logic
if ad_target_zips_text.strip() or uploaded_school_requests_file: # Allow map if either is provided
    
    df_ads_data = parse_ad_target_zips_from_text(ad_target_zips_text)
    df_schools_data = load_school_requests_from_upload(uploaded_school_requests_file)

    # Proceed if we have any data to plot, even if one input is missing but we have US zips
    if not gdf_us_data.empty and (not df_ads_data.empty or not df_schools_data.empty) :
        st.info("Data loaded. Generating map...")
        try:
            map_figure = main_plot_from_original_script(gdf_us_data, df_ads_data, df_schools_data)
            st.pyplot(map_figure)
            st.success("Map generated successfully!")
            
            fn = 'combined_map_streamlit_v4.png' # Incremented version
            img_bytes = io.BytesIO()
            map_figure.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
            img_bytes.seek(0)
            st.download_button(label="Download Map as PNG", data=img_bytes, file_name=fn, mime="image/png")
        except Exception as e: 
            st.error(f"Error during map generation: {e}")
            st.exception(e) # Shows full traceback for debugging
    elif gdf_us_data.empty:
        st.error("Cannot generate map because US ZIP Code master data failed to load.")
    else: # gdf_us_data is loaded, but both ad_zips and school_roles are empty/invalid
        st.warning("Map could not be generated. Please provide Ad Target ZIPs and/or upload a valid School Open Roles file.")
else:
    st.sidebar.info("Please paste Ad Target ZIPs and/or upload the School Open Roles CSV file to generate the map.")
    st.info("Awaiting inputs...")

st.markdown("---")
st.markdown("Streamlit app for visualizing school roles and ad ZIPs based on original mapping script logic.")

