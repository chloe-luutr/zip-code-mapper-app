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

st.set_page_config(layout="wide")
st.title("School Roles & Ad ZIPs Map Generator (Replicates combined_map.png)")

st.markdown("""
This app replicates the functionality of the original `zip-code-maper.py` script.
Upload your three CSV files to generate a map showing:
- School locations with pie charts representing open roles (e.g., TA, Teacher).
- 5 and 10-mile coverage radii (lines) around schools.
- Ad ZIPs with serial numbers.
- An OpenStreetMap basemap with Latitude/Longitude grid.
""")

###############################################################################
# HELPER FUNCTIONS (Directly from or adapted from zip-code-maper.py)
###############################################################################

@st.cache_data # Cache data loading
def load_us_zip_codes_from_upload(uploaded_file_object) -> gpd.GeoDataFrame:
    """Loads US ZIP code data from an uploaded CSV file object."""
    if uploaded_file_object is None: return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")
    try:
        # Attempt to determine delimiter
        uploaded_file_object.seek(0)
        first_lines_bytes = uploaded_file_object.read(1024) # Read first 1KB to check
        uploaded_file_object.seek(0) 
        first_lines_str = first_lines_bytes.decode('utf-8-sig', errors='ignore').splitlines()
        delimiter = ';' if first_lines_str and ';' in first_lines_str[0] and first_lines_str[0].count(';') >= first_lines_str[0].count(',') else ','
        
        df = pd.read_csv(uploaded_file_object, delimiter=delimiter, dtype={'zip': str, 'Zip Code': str}) # Read zip as string
        df.columns = df.columns.str.strip().str.lower()
        
        zip_col, lat_col, lon_col = None, None, None
        
        if 'zip' in df.columns: zip_col = 'zip'
        elif 'zip code' in df.columns: df.rename(columns={'zip code': 'zip'}, inplace=True); zip_col = 'zip'
        
        if 'latitude' in df.columns: lat_col = 'latitude'
        elif 'lat' in df.columns: lat_col = 'lat'
        
        if 'longitude' in df.columns: lon_col = 'longitude'
        elif 'lon' in df.columns: lon_col = 'lon'
        elif 'long' in df.columns: lon_col = 'long'

        # Fallback to Geo Point if separate lat/lon not found or ambiguous
        if not (lat_col and lon_col) and 'geo point' in df.columns:
            try:
                st.info("Attempting to parse 'geo point' for coordinates...")
                lat_lon_split = df['geo point'].astype(str).str.split(',', expand=True)
                df['latitude_parsed'] = pd.to_numeric(lat_lon_split[0], errors='coerce')
                df['longitude_parsed'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
                lat_col, lon_col = 'latitude_parsed', 'longitude_parsed'
            except Exception as e_gp:
                st.warning(f"Could not effectively parse 'geo point': {e_gp}")


        if not zip_col or not lat_col or not lon_col:
            st.error("US ZIP Codes CSV: Could not identify required 'zip', 'latitude', and 'longitude' columns (or parse 'geo point'). Please check headers.")
            return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

        df[zip_col] = df[zip_col].astype(str).str.strip().str.zfill(5)
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        df.dropna(subset=[lat_col, lon_col, zip_col], inplace=True)
        if df.empty: st.error("No valid coordinate or ZIP data in US ZIP Codes CSV after cleaning."); return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), # Lon, Lat order
            crs="EPSG:4326"
        )
        return gdf[[zip_col, 'geometry']].rename(columns={zip_col: 'zip'}) # Ensure 'zip' column
    except Exception as e:
        st.error(f"Error loading US ZIP Codes: {e}")
        return gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326")

@st.cache_data
def load_zips_to_sort_from_upload(uploaded_file_object) -> pd.DataFrame:
    """Loads ZIP codes to sort (Ad ZIPs) from an uploaded CSV file object."""
    if uploaded_file_object is None: return pd.DataFrame(columns=['zip'])
    try:
        uploaded_file_object.seek(0)
        first_line_bytes = uploaded_file_object.read(1024)
        uploaded_file_object.seek(0)
        first_line_str = first_line_bytes.decode('utf-8-sig', errors='ignore').splitlines()[0]
        delimiter = ';' if ';' in first_line_str and first_line_str.count(';') >= first_line_str.count(',') else ','

        # Check if the first line itself is a list of zips (original script logic)
        potential_zips_in_header = [z.strip() for z in first_line_str.split(delimiter) if z.strip().isdigit() and len(z.strip())==5]
        
        if len(potential_zips_in_header) > 1 and len(potential_zips_in_header) == len(first_line_str.split(delimiter)): # Header is likely zips
             df = pd.DataFrame(potential_zips_in_header, columns=['zip'])
        else:
            df = pd.read_csv(uploaded_file_object, delimiter=delimiter, dtype=str)
            df.columns = df.columns.str.strip().str.lower()
            if 'zip' in df.columns:
                df = df[['zip']]
            elif df.shape[1] == 1: # Assume first column if only one
                df = df.rename(columns={df.columns[0]: 'zip'})[['zip']]
            else:
                 st.error("Zips to Sort CSV: Could not find 'zip' column or parse as a simple list/header of ZIPs.")
                 return pd.DataFrame(columns=['zip'])
        
        df['zip'] = df['zip'].astype(str).str.strip().str.zfill(5)
        df = df[df['zip'].str.match(r'^\d{5}$')].drop_duplicates(subset=['zip'])
        return df
    except Exception as e:
        st.error(f"Error loading Zips to Sort (Ad ZIPs): {e}")
        return pd.DataFrame(columns=['zip'])

@st.cache_data
def load_school_requests_from_upload(uploaded_file_object) -> pd.DataFrame:
    """Loads school requests/open roles from an uploaded CSV file object."""
    if uploaded_file_object is None: return pd.DataFrame()
    try:
        uploaded_file_object.seek(0)
        first_lines_bytes = uploaded_file_object.read(1024)
        uploaded_file_object.seek(0)
        first_lines_str = first_lines_bytes.decode('utf-8-sig', errors='ignore').splitlines()[0]
        delimiter = ';' if ';' in first_lines_str and first_line_str.count(';') >= first_line_str.count(',') else ','

        df = pd.read_csv(uploaded_file_object, delimiter=delimiter)
        df.columns = df.columns.str.strip().str.lower()
        if 'zip' not in df.columns:
            st.error("School Requests (Open Roles) CSV must contain a 'zip' column.")
            return pd.DataFrame()
        df['zip'] = df['zip'].astype(str).str.strip().str.zfill(5)
        return df
    except Exception as e:
        st.error(f"Error loading School Requests (Open Roles): {e}")
        return pd.DataFrame()

def geodesic_buffer_original(lon, lat, miles):
    radius_m = miles * 1609.34
    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj = pyproj.CRS.from_proj4(
       f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    center = Point(lon, lat)
    center_aeqd = transform(project_fwd, center)
    buffer_aeqd = center_aeqd.buffer(radius_m)
    return transform(project_back, buffer_aeqd)

def create_geodesic_buffers_for_schools_original(gdf_schools, radii=(5,10)):
    if gdf_schools.empty or 'geometry' not in gdf_schools.columns: return
    for r in radii:
        col_name = f"buffer_{r}"
        poly_list = []
        for _, row in gdf_schools.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not isinstance(geom, Point): poly_list.append(None); continue
            poly = geodesic_buffer_original(geom.x, geom.y, r)
            poly_list.append(poly)
        gdf_schools[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326")

def plot_pie_chart_original(ax, x_center, y_center, counts_dict, radius, role_colors):
    """Plots a pie chart on the given Matplotlib axes, using provided role_colors."""
    total = sum(counts_dict.values())
    if total <= 0 or radius <=0 : return # Also check for non-positive radius
    
    # Sort items by role name for consistent color mapping if role_colors is a dict
    items = sorted(counts_dict.items(), key=lambda item: item[0]) 
    
    values = [v for _, v in items]
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
            wedge = Wedge(
                center=(x_center, y_center), r=radius, 
                theta1=current_angle_start, theta2=current_angle_start + angle_extent,
                facecolor=role_colors.get(role, plt.cm.get_cmap('Greys')(0.5)), # Default color if role not in map
                edgecolor='white', linewidth=0.5, alpha=0.85
            )
            ax.add_patch(wedge)
            current_angle_start += angle_extent


###############################################################################
# MAIN PLOT FUNCTION (Adapted from zip-code-maper.py)
###############################################################################
def main_plot_from_original_script(gdf_us, df_ads, df_schools):
    # Ensure 'zip' columns are consistently formatted as 5-digit strings
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

    gdf_ads_merged = pd.merge(df_ads, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry']) if not df_ads.empty else gpd.GeoDataFrame()
    gdf_schools_merged = pd.merge(df_schools, gdf_filtered[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry']) if not df_schools.empty else gpd.GeoDataFrame()

    if not gdf_ads_merged.empty: gdf_ads_merged = gpd.GeoDataFrame(gdf_ads_merged, geometry='geometry', crs="EPSG:4326")
    if not gdf_schools_merged.empty: gdf_schools_merged = gpd.GeoDataFrame(gdf_schools_merged, geometry='geometry', crs="EPSG:4326")

    if gdf_schools_merged.empty and gdf_ads_merged.empty:
        st.warning("No geodata for Ad/School ZIPs after merge."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No geodata for Ad/School ZIPs", ha='center'); return fig

    # Identify teacher/role columns (numeric, not 'zip' or geometry-related)
    teacher_cols = []
    if not gdf_schools_merged.empty:
        excluded_cols = ['zip', 'geometry'] + [col for col in gdf_schools_merged.columns if 'buffer' in col]
        numeric_cols = gdf_schools_merged.select_dtypes(include=np.number).columns
        teacher_cols = [c for c in numeric_cols if c not in excluded_cols]
        if not teacher_cols and not df_schools.empty: # df_schools is the original uploaded roles file
            # Fallback to check original df_schools if gdf_schools_merged lost numeric types or columns
            original_numeric_cols = df_schools.select_dtypes(include=np.number).columns
            teacher_cols = [c for c in original_numeric_cols if c.lower().strip() != 'zip'] # simple check

        if not teacher_cols: st.info("No numeric columns identified as 'roles' for pie charts in School Open Roles data.")
        else: st.info(f"Identified role columns for pie charts: {', '.join(teacher_cols)}")


    if not gdf_schools_merged.empty:
        create_geodesic_buffers_for_schools_original(gdf_schools_merged, radii=(5,10))

    gdf_ads_3857      = gdf_ads_merged.to_crs(epsg=3857) if not gdf_ads_merged.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_schools_3857  = gdf_schools_merged.to_crs(epsg=3857) if not gdf_schools_merged.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_filtered_3857 = gdf_filtered.to_crs(epsg=3857)

    if not gdf_schools_merged.empty:
        if 'buffer_5' in gdf_schools_merged.columns: gdf_schools_3857['buffer_5_3857']  = gpd.GeoSeries(gdf_schools_merged['buffer_5'], crs="EPSG:4326").to_crs(epsg=3857)
        if 'buffer_10' in gdf_schools_merged.columns: gdf_schools_3857['buffer_10_3857'] = gpd.GeoSeries(gdf_schools_merged['buffer_10'], crs="EPSG:4326").to_crs(epsg=3857)

    combined_bounds_gdf = pd.concat([g for g in [gdf_filtered_3857, gdf_ads_3857, gdf_schools_3857] if not g.empty])
    if combined_bounds_gdf.empty or combined_bounds_gdf.total_bounds is None or any(np.isnan(combined_bounds_gdf.total_bounds)):
        minx, miny, maxx, maxy = -14000000, 2800000, -7000000, 6300000 
    else: minx, miny, maxx, maxy = combined_bounds_gdf.total_bounds
    
    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6
    expand_factor = st.session_state.get('map_expand_factor_orig', 1.5) 
    pad_x, pad_y = expand_factor * w * 0.1, expand_factor * h * 0.1

    min_jobs_val, max_jobs_val = float('inf'), 0
    if teacher_cols and not gdf_schools_merged.empty:
        for _, row in gdf_schools_merged.iterrows(): 
            total = sum(pd.to_numeric(row.get(tc, 0), errors='coerce') or 0 for tc in teacher_cols) # Ensure numeric sum
            if total > max_jobs_val: max_jobs_val = total
            if total < min_jobs_val and total > 0: min_jobs_val = total
    if min_jobs_val == float('inf'): min_jobs_val = 0
    if max_jobs_val == 0 and min_jobs_val == 0: max_jobs_val = 1 

    BIGGEST_PIE_RADIUS_orig = st.session_state.get('pie_radius_scale_orig', 3000.0)
    get_pie_radius_orig = lambda total_jobs: BIGGEST_PIE_RADIUS_orig * math.sqrt(max(0, total_jobs) / max_jobs_val) if max_jobs_val > 0 else 0

    fig, ax = plt.subplots(figsize=(12,10))

    # Role colors for pie charts and legend
    # Define a fixed color map for roles if teacher_cols is not too long, otherwise cycle
    role_color_map = {}
    if teacher_cols:
        palette = plt.cm.get_cmap('tab10', len(teacher_cols)) if len(teacher_cols) <= 10 else plt.cm.get_cmap('tab20', len(teacher_cols))
        for i, role in enumerate(sorted(list(set(teacher_cols)))): # Sort for consistency
            role_color_map[role] = palette(i)


    # 1) Background zips with serial numbers
    if not gdf_filtered_3857.empty:
        ax.plot(gdf_filtered_3857.geometry.x, gdf_filtered_3857.geometry.y, 'o', color='lightgray', alpha=0.4, markersize=8, label="Contextual ZIPs", zorder=1)
    
    zip_serial_map = {}
    if not df_ads.empty: # Use df_ads which has the original list for serial numbers
        zip_serial_map = {str(zip_code).zfill(5): i+1 for i, zip_code in enumerate(df_ads['zip'])}
        # Plot serial numbers on Ad ZIP locations if they are in gdf_ads_3857
        if not gdf_ads_3857.empty:
            for _, row in gdf_ads_3857.iterrows():
                # Get original zip from gdf_ads_merged that corresponds to this projected point
                original_zip_row = gdf_ads_merged[gdf_ads_merged.geometry == row.geometry.to_crs("EPSG:4326").iloc[0] if hasattr(row.geometry, 'to_crs') else row.geometry] # Ensure comparison in WGS84
                if not original_zip_row.empty:
                    serial = zip_serial_map.get(original_zip_row['zip'].iloc[0])
                    if serial is not None and row.geometry:
                        ax.text(row.geometry.x, row.geometry.y, str(serial), color='black', fontsize=7, ha='center', va='center', zorder=5)


    # 2) Ad zips (highlighted)
    if not gdf_ads_3857.empty:
        gdf_ads_3857.plot(ax=ax, marker='s', color='green', markersize=40, label="Ad ZIPs", zorder=3, edgecolor='darkgreen')

    # 3) Coverage polygons (lines)
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_5_3857'].notna().any():
            gdf_schools_3857[gdf_schools_3857['buffer_5_3857'].notna()].plot(ax=ax, edgecolor='red', facecolor='none', alpha=0.5, linewidth=1.0, zorder=4)
        if 'buffer_10_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_10_3857'].notna().any():
            gdf_schools_3857[gdf_schools_3857['buffer_10_3857'].notna()].plot(ax=ax, edgecolor='orange', facecolor='none', alpha=0.6, linewidth=1.5, zorder=3)

    # 4) Pie charts for schools
    if teacher_cols and not gdf_schools_3857.empty:
        for idx, row_proj in gdf_schools_3857.iterrows(): 
            if row_proj.geometry is None or row_proj.geometry.is_empty: continue
            # Get original data row using index from gdf_schools_merged
            original_row = gdf_schools_merged.loc[idx]
            counts_dict = {tc: pd.to_numeric(original_row.get(tc, 0), errors='coerce') or 0 for tc in teacher_cols}
            counts_dict = {k: v for k, v in counts_dict.items() if v > 0} # Filter out zero counts for pie

            if counts_dict:
                total_jobs_at_school = sum(counts_dict.values())
                r_pie = get_pie_radius_orig(total_jobs_at_school)
                if r_pie > 500: # Only plot if radius is reasonably large (avoid tiny pies)
                    plot_pie_chart_original(ax, row_proj.geometry.x, row_proj.geometry.y, counts_dict, r_pie, role_color_map)
    elif not gdf_schools_3857.empty: 
         gdf_schools_3857.plot(ax=ax, marker='P', color='darkviolet', markersize=60, label="School Locations (No Role Data)", zorder=5, alpha=0.8, edgecolor='black')

    try: ctx.add_basemap(ax, crs=gdf_filtered_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=6)
    except Exception as e: st.warning(f"Could not add basemap: {e}")

    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
    
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    num_xticks = st.session_state.get('num_grid_ticks_orig', 10)
    num_yticks = st.session_state.get('num_grid_ticks_orig', 10)
    plot_xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=num_xticks)
    plot_yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=num_yticks)
    xticks_latlon = [transformer.transform(x, ax.get_ylim()[0])[0] for x in plot_xticks] 
    yticks_latlon = [transformer.transform(ax.get_xlim()[0], y)[1] for y in plot_yticks] 
    ax.set_xticks(plot_xticks); ax.set_xticklabels([f"{lon:.2f}°" for lon in xticks_latlon], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(plot_yticks); ax.set_yticklabels([f"{lat:.2f}°" for lat in yticks_latlon], fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')

    handles, labels = [], []
    if not gdf_filtered_3857.empty and not any("Contextual ZIPs" in lab for lab in ax.get_legend_handles_labels()[1]):
        handles.append(mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', markersize=5, alpha=0.4)); labels.append('Contextual ZIPs')
    if not gdf_ads_3857.empty and not any("Ad ZIPs" in lab for lab in ax.get_legend_handles_labels()[1]):
        handles.append(mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=7)); labels.append('Ad ZIPs')
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857: handles.append(mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.0, alpha=0.5)); labels.append('5-mile School Coverage')
        if 'buffer_10_3857' in gdf_schools_3857: handles.append(mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, alpha=0.6)); labels.append('10-mile School Coverage')
    if teacher_cols and not gdf_schools_plot.empty:
        for role, color in role_color_map.items():
            handles.append(mpatches.Patch(color=color, label=role.replace('_', ' ').title()))
            labels.append(role.replace('_', ' ').title())
    elif not gdf_schools_3857.empty and not teacher_cols and not any("School Locations" in lab for lab in ax.get_legend_handles_labels()[1]):
        handles.append(mlines.Line2D([], [], color='darkviolet', marker='P', linestyle='None', markersize=8)); labels.append('School Locations')
    
    current_handles_ax, current_labels_ax = ax.get_legend_handles_labels()
    combined_handles = current_handles_ax + [h for i, h in enumerate(handles) if labels[i] not in current_labels_ax]
    combined_labels = current_labels_ax + [l for i, l in enumerate(labels) if l not in current_labels_ax]
    
    # Remove duplicates from combined lists while preserving order
    final_legend_items = {}
    for handle, label in zip(combined_handles, combined_labels):
        if label not in final_legend_items:
            final_legend_items[label] = handle
            
    if final_legend_items:
        ax.legend(final_legend_items.values(), final_legend_items.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small', title="Legend", title_fontsize="medium")

    if teacher_cols and not gdf_schools_plot.empty:
         ax.text(1.02, 0.5 if len(final_legend_items) < 8 else 0.2, 
                f"School Job Range:\nMin Roles: {min_jobs_val}\nMax Roles: {max_jobs_val}",
                transform=ax.transAxes, va='top', fontsize='small',
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_title("Schools (Pie Charts + Coverage), Ad ZIPs, OSM Basemap\nLat/Lon Grid, Legend & Job Range on Right", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    return fig

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
st.sidebar.header("Upload Data Files (CSV)")
uploaded_us_zips_file = st.sidebar.file_uploader("1. US ZIP Codes Master File (zip, latitude, longitude / Geo Point)", type="csv", key="us_zips_orig_v2")
uploaded_zips_to_sort_file = st.sidebar.file_uploader("2. Ad Target ZIPs File (zip)", type="csv", key="zips_to_sort_orig_v2")
uploaded_school_requests_file = st.sidebar.file_uploader("3. School Open Roles File (zip, role_counts...)", type="csv", key="school_requests_orig_v2")

st.sidebar.header("Map Display Options")
if 'map_expand_factor_orig' not in st.session_state: st.session_state.map_expand_factor_orig = 1.5
st.session_state.map_expand_factor_orig = st.sidebar.slider("Map Zoom/Expand Factor:", min_value=0.5, max_value=5.0, value=st.session_state.map_expand_factor_orig, step=0.1, key="map_expand_slider_orig_v2")

if 'pie_radius_scale_orig' not in st.session_state: st.session_state.pie_radius_scale_orig = 3000.0
st.session_state.pie_radius_scale_orig = st.sidebar.slider("Pie Chart Max Radius Scale (map units):", min_value=500.0, max_value=10000.0, value=st.session_state.pie_radius_scale_orig, step=100.0, key="pie_scale_slider_orig_v2")

if 'num_grid_ticks_orig' not in st.session_state: st.session_state.num_grid_ticks_orig = 10
st.session_state.num_grid_ticks_orig = st.sidebar.slider("Number of Lat/Lon Grid Ticks:", min_value=3, max_value=30, value=st.session_state.num_grid_ticks_orig, step=1, key="grid_ticks_slider_orig_v2")

if uploaded_us_zips_file and uploaded_zips_to_sort_file and uploaded_school_requests_file:
    st.sidebar.success("All files uploaded!")
    gdf_us_data = load_us_zip_codes_from_upload(uploaded_us_zips_file)
    df_ads_data = load_zips_to_sort_from_upload(uploaded_zips_to_sort_file)
    df_schools_data = load_school_requests_from_upload(uploaded_school_requests_file)

    if not gdf_us_data.empty and (not df_ads_data.empty or not df_schools_data.empty) :
        st.info("Data loaded. Generating map...")
        try:
            map_figure = main_plot_from_original_script(gdf_us_data, df_ads_data, df_schools_data)
            st.pyplot(map_figure)
            st.success("Map generated successfully!")
            fn = 'combined_map_streamlit.png'
            img_bytes = io.BytesIO(); map_figure.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
            img_bytes.seek(0)
            st.download_button(label="Download Map as PNG", data=img_bytes, file_name=fn, mime="image/png")
        except Exception as e: st.error(f"Error during map generation: {e}"); st.exception(e)
    else:
        st.warning("Map could not be generated. Ensure files are valid, contain data, and ZIPs in Ad/School files exist in US ZIP Master.")
else:
    st.sidebar.info("Please upload all three CSV files to generate the map.")
    st.info("Awaiting file uploads...")
st.markdown("---")

