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
st.title("School Roles & Ad ZIPs Map Generator")

st.markdown("""
This app replicates the functionality of the `zip-code-maper.py` script.
Upload your three CSV files to generate a map showing:
- School locations with pie charts representing open roles.
- 5 and 10-mile coverage radii around schools.
- Ad ZIPs.
- An OpenStreetMap basemap with Lat/Lon grid.
""")

###############################################################################
# HELPER FUNCTIONS (Adapted from zip-code-maper.py)
###############################################################################

@st.cache_data # Cache data loading
def load_us_zip_codes_original(uploaded_file_object) -> gpd.GeoDataFrame:
    """Loads US ZIP code data from an uploaded CSV file object."""
    if uploaded_file_object is None: return gpd.GeoDataFrame()
    try:
        # Try to determine delimiter
        first_lines = uploaded_file_object.getvalue().decode('utf-8-sig').splitlines()[:2]
        delimiter = ';' if first_lines and ';' in first_lines[0] and first_lines[0].count(';') >= first_lines[0].count(',') else ','
        uploaded_file_object.seek(0) # Reset buffer position

        df = pd.read_csv(uploaded_file_object, delimiter=delimiter, dtype={'zip': str, 'Zip Code': str})
        df.columns = df.columns.str.strip().str.lower()
        
        zip_col = None
        if 'zip' in df.columns: zip_col = 'zip'
        elif 'zip code' in df.columns: df.rename(columns={'zip code': 'zip'}, inplace=True); zip_col = 'zip'
        
        lat_col, lon_col = None, None
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_col, lon_col = 'latitude', 'longitude'
        elif 'geo point' in df.columns: # Handle "lat,lon" string
            try:
                lat_lon_split = df['geo point'].astype(str).str.split(',', expand=True)
                df['latitude_parsed'] = pd.to_numeric(lat_lon_split[0], errors='coerce')
                df['longitude_parsed'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
                lat_col, lon_col = 'latitude_parsed', 'longitude_parsed'
            except Exception: pass # Will fail later if still None
        
        if not zip_col or not lat_col or not lon_col:
            st.error("US ZIP Codes CSV must contain identifiable columns for ZIP, Latitude, and Longitude (or 'Geo Point').")
            return gpd.GeoDataFrame()

        df.dropna(subset=[lat_col, lon_col], inplace=True)
        if df.empty: st.error("No valid coordinate data in US ZIP Codes CSV."); return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), # Ensure correct order for points_from_xy
            crs="EPSG:4326"
        )
        return gdf[['zip', 'geometry']] # Keep only essential columns
    except Exception as e:
        st.error(f"Error loading US ZIP Codes: {e}")
        return gpd.GeoDataFrame()

@st.cache_data
def load_zips_to_sort_original(uploaded_file_object) -> pd.DataFrame:
    """Loads ZIP codes to sort (Ad ZIPs) from an uploaded CSV file object."""
    if uploaded_file_object is None: return pd.DataFrame(columns=['zip'])
    try:
        # Read the first line to check if it's all digits (potential header of zips)
        first_line_bytes = uploaded_file_object.readline()
        uploaded_file_object.seek(0) # Reset buffer
        first_line_str = first_line_bytes.decode('utf-8-sig').strip()
        
        # Determine delimiter
        delimiter = ';' if ';' in first_line_str and first_line_str.count(';') >= first_line_str.count(',') else ','

        is_header_zips = all(col.strip().isdigit() for col in first_line_str.split(delimiter))

        if is_header_zips:
            # If header is zips, read them as data, no actual header
            df = pd.DataFrame(first_line_str.split(delimiter), columns=['zip'])
        else:
            # Read normally, assuming a header row
            df = pd.read_csv(uploaded_file_object, delimiter=delimiter, dtype=str) # Read all as string initially
            df.columns = df.columns.str.strip().str.lower()
            if 'zip' not in df.columns and df.shape[1] > 0: # If no 'zip' column, assume first column
                df.rename(columns={df.columns[0]: 'zip'}, inplace=True)
            elif 'zip' not in df.columns:
                 st.error("Zips to Sort CSV must contain a 'zip' column or have ZIPs in the first column/header.")
                 return pd.DataFrame(columns=['zip'])
        
        df['zip'] = df['zip'].astype(str).str.strip().str.zfill(5)
        df = df[df['zip'].str.match(r'^\d{5}$')].drop_duplicates(subset=['zip'])
        return df[['zip']]
    except Exception as e:
        st.error(f"Error loading Zips to Sort (Ad ZIPs): {e}")
        return pd.DataFrame(columns=['zip'])

@st.cache_data
def load_school_requests_original(uploaded_file_object) -> pd.DataFrame:
    """Loads school requests/open roles from an uploaded CSV file object."""
    if uploaded_file_object is None: return pd.DataFrame()
    try:
        first_lines = uploaded_file_object.getvalue().decode('utf-8-sig').splitlines()[:2]
        delimiter = ';' if first_lines and ';' in first_lines[0] and first_lines[0].count(';') >= first_lines[0].count(',') else ','
        uploaded_file_object.seek(0)

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

def plot_pie_chart_original(ax, x_center, y_center, counts_dict, radius):
    total = sum(counts_dict.values())
    if total <= 0: return
    items = sorted(counts_dict.items()) # Sort for consistent color order
    values = [v for _, v in items]
    fracs = [v / total for v in values]
    # Ensure small slices are still visible by setting a minimum percentage for angle calculation
    min_angle_deg = 1 # Minimum degrees for a slice to be visible
    angles_deg = [max(f * 360, min_angle_deg if f > 0 else 0) for f in fracs]
    # Normalize angles if they exceed 360 due to min_angle_deg
    sum_angles_deg = sum(angles_deg)
    if sum_angles_deg > 360:
        angles_deg = [a * (360 / sum_angles_deg) for a in angles_deg]

    angles = np.cumsum([0] + angles_deg)
    
    # Use a more diverse color palette if many categories
    num_colors = len(values)
    if num_colors <= 10: colors = plt.cm.get_cmap('tab10', num_colors).colors
    elif num_colors <= 20: colors = plt.cm.get_cmap('tab20', num_colors).colors
    else: colors = plt.cm.get_cmap('nipy_spectral', num_colors).colors


    for i in range(len(values)):
        if values[i] > 0 : # Only plot wedges for non-zero values
            wedge = Wedge(
                center=(x_center, y_center), r=radius, theta1=angles[i], theta2=angles[i+1],
                facecolor=colors[i % len(colors)], edgecolor='white', linewidth=0.5, alpha=0.8
            )
            ax.add_patch(wedge)

###############################################################################
# MAIN PLOT FUNCTION (Adapted from zip-code-maper.py)
###############################################################################
def main_plot_original(gdf_us, df_ads, df_schools):
    if gdf_us.empty: st.error("US ZIP Code data is missing or invalid."); return plt.figure() # Return empty fig
    if df_ads.empty and df_schools.empty: st.warning("Both Ad ZIPs and School Roles data are missing. Map may be empty."); # Allow plotting if one is present
    
    # Ensure ZIPs are strings for merging
    gdf_us['zip'] = gdf_us['zip'].astype(str).str.zfill(5)
    if not df_ads.empty: df_ads['zip'] = df_ads['zip'].astype(str).str.zfill(5)
    if not df_schools.empty: df_schools['zip'] = df_schools['zip'].astype(str).str.zfill(5)

    # Determine relevant zips for initial filtering (map bounds context)
    relevant_zips_set = set()
    if not df_ads.empty: relevant_zips_set.update(df_ads['zip'].unique())
    if not df_schools.empty: relevant_zips_set.update(df_schools['zip'].unique())
    
    if not relevant_zips_set:
        st.warning("No relevant ZIPs from Ad or School data to display.")
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No ZIPs to map", ha='center'); return fig

    gdf_filtered_context = gdf_us[gdf_us['zip'].isin(relevant_zips_set)].copy()
    if gdf_filtered_context.empty:
        st.warning("None of the provided Ad/School ZIPs were found in the US ZIP master list.")
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "ZIPs not in master list", ha='center'); return fig

    # Prepare GeoDataFrames for ads and schools
    gdf_ads_plot = gpd.GeoDataFrame()
    if not df_ads.empty:
        gdf_ads_plot = pd.merge(df_ads, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not gdf_ads_plot.empty: gdf_ads_plot = gpd.GeoDataFrame(gdf_ads_plot, geometry='geometry', crs="EPSG:4326")

    gdf_schools_plot = gpd.GeoDataFrame()
    if not df_schools.empty:
        gdf_schools_plot = pd.merge(df_schools, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not gdf_schools_plot.empty: gdf_schools_plot = gpd.GeoDataFrame(gdf_schools_plot, geometry='geometry', crs="EPSG:4326")

    if gdf_schools_plot.empty and gdf_ads_plot.empty:
        st.warning("No matching ZIP geometry found for Ad or School data after merging.")
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No geodata for Ad/School ZIPs", ha='center'); return fig

    # Identify teacher/role columns (numeric columns excluding 'zip', and known non-role columns)
    teacher_cols = []
    if not gdf_schools_plot.empty:
        excluded_cols = ['zip', 'geometry', 'latitude', 'longitude', 'lat', 'lon'] # Add more if needed
        potential_role_cols = [c for c in gdf_schools_plot.columns if c not in excluded_cols and pd.api.types.is_numeric_dtype(gdf_schools_plot[c])]
        # Heuristic: assume columns with small integer values are more likely to be role counts
        for col in potential_role_cols:
            if gdf_schools_plot[col].dropna().max() < 1000 and (gdf_schools_plot[col].dropna() % 1 == 0).all(): # Max count < 1000, all integers
                 teacher_cols.append(col)
        if not teacher_cols: st.info("No numeric columns identified as 'roles' for pie charts in School Requests data.")


    if not gdf_schools_plot.empty:
        create_geodesic_buffers_for_schools_original(gdf_schools_plot, radii=(5,10))

    # Projections
    gdf_ads_3857      = gdf_ads_plot.to_crs(epsg=3857) if not gdf_ads_plot.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_schools_3857  = gdf_schools_plot.to_crs(epsg=3857) if not gdf_schools_plot.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_filtered_3857 = gdf_filtered_context.to_crs(epsg=3857)

    # Project buffers for schools
    if not gdf_schools_plot.empty:
        if 'buffer_5' in gdf_schools_plot.columns: 
            gdf_schools_3857['buffer_5_3857']  = gpd.GeoSeries(gdf_schools_plot['buffer_5'], crs="EPSG:4326").to_crs(epsg=3857)
        if 'buffer_10' in gdf_schools_plot.columns: 
            gdf_schools_3857['buffer_10_3857'] = gpd.GeoSeries(gdf_schools_plot['buffer_10'], crs="EPSG:4326").to_crs(epsg=3857)

    # Determine map bounds
    combined_bounds_gdf = pd.concat([gdf_filtered_3857, gdf_ads_3857, gdf_schools_3857])
    if combined_bounds_gdf.empty or combined_bounds_gdf.total_bounds is None or any(np.isnan(combined_bounds_gdf.total_bounds)):
        st.warning("Cannot determine map bounds. Map may not display correctly.")
        # Fallback bounds (e.g., continental US)
        minx, miny, maxx, maxy = -14000000, 2800000, -7000000, 6300000 
    else:
        minx, miny, maxx, maxy = combined_bounds_gdf.total_bounds
    
    w = maxx - minx if maxx > minx else 1000000
    h = maxy - miny if maxy > miny else 1000000
    expand_factor = st.session_state.get('map_expand_factor', 1.5) # Get from session state or default
    pad_x, pad_y = expand_factor * w * 0.1, expand_factor * h * 0.1


    # Calculate min/max jobs for pie chart scaling
    min_jobs, max_jobs = float('inf'), 0
    if teacher_cols and not gdf_schools_plot.empty:
        for _, row in gdf_schools_plot.iterrows(): # Use original GDF for role counts
            total = sum(row.get(tc, 0) for tc in teacher_cols if pd.notna(row.get(tc,0)))
            if total > max_jobs: max_jobs = total
            if total < min_jobs and total > 0: min_jobs = total
    if min_jobs == float('inf'): min_jobs = 0
    if max_jobs == 0 and min_jobs == 0: max_jobs = 1 # Avoid division by zero

    BIGGEST_PIE_RADIUS = st.session_state.get('pie_radius_scale', 3000.0) # Get from session state or default
    get_pie_radius = lambda total_jobs: BIGGEST_PIE_RADIUS * math.sqrt(max(0, total_jobs) / max_jobs) if max_jobs > 0 else 0

    fig, ax = plt.subplots(figsize=(12,10)) # Adjusted figsize

    # 1) Background zips (from gdf_filtered_context)
    if not gdf_filtered_3857.empty:
        gdf_filtered_3857.plot(ax=ax, marker='o', color='gray', alpha=0.2, markersize=8, label="Contextual ZIPs", zorder=1)

    # Serial numbers for Ad ZIPs (from df_ads)
    zip_serial_map = {}
    if not df_ads.empty:
        zip_serial_map = {zip_code: i+1 for i, zip_code in enumerate(df_ads['zip'])}
        # Plot Ad ZIP serial numbers on top of the contextual ZIPs if they match
        for _, row in gdf_filtered_3857.iterrows():
            serial = zip_serial_map.get(row['zip'])
            if serial is not None and row.geometry:
                ax.text(row.geometry.x, row.geometry.y, str(serial), color='black', fontsize=7, ha='center', va='center', zorder=2)


    # 2) Ad zips (highlighted)
    if not gdf_ads_3857.empty:
        gdf_ads_3857.plot(ax=ax, marker='s', color='green', markersize=40, label="Ad ZIPs", zorder=3, edgecolor='darkgreen')

    # 3) Coverage polygons (lines, as in original script)
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_5_3857'].notna().any():
            gdf_schools_3857[gdf_schools_3857['buffer_5_3857'].notna()].plot(ax=ax, edgecolor='red', facecolor='none', alpha=0.5, linewidth=1.0, zorder=4)
        if 'buffer_10_3857' in gdf_schools_3857.columns and gdf_schools_3857['buffer_10_3857'].notna().any():
            gdf_schools_3857[gdf_schools_3857['buffer_10_3857'].notna()].plot(ax=ax, edgecolor='orange', facecolor='none', alpha=0.6, linewidth=1.5, zorder=3)

    # 4) Pie charts for schools
    if teacher_cols and not gdf_schools_3857.empty:
        for _, row in gdf_schools_3857.iterrows(): # Iterate over projected schools
            if row.geometry is None or row.geometry.is_empty: continue
            counts_dict = {tc: gdf_schools_plot.loc[row.name, tc] for tc in teacher_cols if pd.notna(gdf_schools_plot.loc[row.name, tc]) and gdf_schools_plot.loc[row.name, tc] != 0} # Get counts from original df
            if counts_dict:
                total_jobs_at_school = sum(counts_dict.values())
                r_pie = get_pie_radius(total_jobs_at_school)
                if r_pie > 0: # Only plot if radius is positive
                    plot_pie_chart_original(ax, row.geometry.x, row.geometry.y, counts_dict, r_pie)
    elif not gdf_schools_3857.empty: # If no teacher_cols but schools exist, plot them as points
         gdf_schools_3857.plot(ax=ax, marker='P', color='darkviolet', markersize=60, label="School Locations (No Role Data)", zorder=5, alpha=0.8, edgecolor='black')


    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf_filtered_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=6)
    except Exception as e: st.warning(f"Could not add basemap: {e}")

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    
    # Lat/Lon Grid (as in original script)
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    num_xticks = st.session_state.get('num_grid_ticks', 10) # Default 10, configurable
    num_yticks = st.session_state.get('num_grid_ticks', 10)
    
    plot_xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=num_xticks)
    plot_yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=num_yticks)
    
    # Transform tick positions to Lat/Lon for labels
    xticks_latlon = [transformer.transform(x, ax.get_ylim()[0])[0] for x in plot_xticks] # lon values
    yticks_latlon = [transformer.transform(ax.get_xlim()[0], y)[1] for y in plot_yticks] # lat values

    ax.set_xticks(plot_xticks)
    ax.set_xticklabels([f"{lon:.2f}°" for lon in xticks_latlon], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(plot_yticks)
    ax.set_yticklabels([f"{lat:.2f}°" for lat in yticks_latlon], fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')


    # Legend
    handles, labels = [], []
    # Get existing handles from plots if any (e.g., gdf_ads_3857.plot adds its own)
    current_handles, current_labels = ax.get_legend_handles_labels()
    handles.extend(current_handles); labels.extend(current_labels)

    # Add custom legend items if not already present by plot labels
    if not any("Contextual ZIPs" in lab for lab in labels) and not gdf_filtered_3857.empty :
        handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=5, alpha=0.3)); labels.append('Contextual ZIPs')
    if not any("Ad ZIPs" in lab for lab in labels) and not gdf_ads_3857.empty: # If gdf_ads_3857.plot didn't add it
        handles.append(mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=7)); labels.append('Ad ZIPs')
    
    if not gdf_schools_3857.empty:
        if 'buffer_5_3857' in gdf_schools_3857: handles.append(mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.0, alpha=0.5)); labels.append('5-mile School Coverage')
        if 'buffer_10_3857' in gdf_schools_3857: handles.append(mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, alpha=0.6)); labels.append('10-mile School Coverage')

    if teacher_cols and not gdf_schools_plot.empty:
        # Create patches for pie chart legend
        # Ensure teacher_cols are sorted for consistent legend color mapping
        sorted_teacher_cols = sorted(list(set(teacher_cols))) # Use set to avoid duplicates if any
        num_pie_colors = len(sorted_teacher_cols)
        
        pie_color_palette = plt.cm.get_cmap('tab10', num_pie_colors) if num_pie_colors <=10 else plt.cm.get_cmap('tab20', num_pie_colors)

        for i, tc in enumerate(sorted_teacher_cols):
            handles.append(mpatches.Patch(color=pie_color_palette.colors[i % len(pie_color_palette.colors)], label=tc.replace('_', ' ').title()))
            labels.append(tc.replace('_', ' ').title())
    
    if handles: # Only show legend if there's something to show
        # Remove duplicate labels/handles before creating legend
        unique_legend = dict(zip(labels, handles))
        ax.legend(unique_legend.values(), unique_legend.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small', title="Legend", title_fontsize="medium")

    # Job Range Text
    if teacher_cols and not gdf_schools_plot.empty:
         ax.text(1.02, 0.5 if len(unique_legend) < 5 else 0.3, # Adjust y based on legend size
                f"School Job Range:\nMin Roles: {min_jobs}\nMax Roles: {max_jobs}",
                transform=ax.transAxes, va='top', fontsize='small',
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_title("Schools (Pie Charts + 5/10mi Coverage), Ad ZIPs, and OSM Basemap\nLegend & Job Range on Right", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for legend
    return fig

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################

# --- Sidebar for File Uploads and Controls ---
st.sidebar.header("Upload Data Files (CSV)")
uploaded_us_zips_file = st.sidebar.file_uploader("1. US ZIP Codes (zip, latitude, longitude / Geo Point)", type="csv", key="us_zips_orig")
uploaded_zips_to_sort_file = st.sidebar.file_uploader("2. Ad Target ZIPs (zip)", type="csv", key="zips_to_sort_orig")
uploaded_school_requests_file = st.sidebar.file_uploader("3. School Open Roles (zip, role_counts...)", type="csv", key="school_requests_orig")

st.sidebar.header("Map Display Options")
if 'map_expand_factor' not in st.session_state: st.session_state.map_expand_factor = 1.5
st.session_state.map_expand_factor = st.sidebar.slider("Map Zoom/Expand Factor:", min_value=0.5, max_value=5.0, value=st.session_state.map_expand_factor, step=0.1, key="map_expand_slider")

if 'pie_radius_scale' not in st.session_state: st.session_state.pie_radius_scale = 3000.0
st.session_state.pie_radius_scale = st.sidebar.slider("Pie Chart Max Radius Scale:", min_value=500.0, max_value=10000.0, value=st.session_state.pie_radius_scale, step=100.0, key="pie_scale_slider")

if 'num_grid_ticks' not in st.session_state: st.session_state.num_grid_ticks = 10
st.session_state.num_grid_ticks = st.sidebar.slider("Number of Lat/Lon Grid Ticks:", min_value=3, max_value=30, value=st.session_state.num_grid_ticks, step=1, key="grid_ticks_slider")


# --- Main App Logic ---
if uploaded_us_zips_file and uploaded_zips_to_sort_file and uploaded_school_requests_file:
    st.sidebar.success("All files uploaded!")

    gdf_us_data = load_us_zip_codes_original(uploaded_us_zips_file)
    df_ads_data = load_zips_to_sort_original(uploaded_zips_to_sort_file)
    df_schools_data = load_school_requests_original(uploaded_school_requests_file)

    if not gdf_us_data.empty and (not df_ads_data.empty or not df_schools_data.empty) :
        st.info("Data loaded. Generating map...")
        try:
            map_figure = main_plot_original(gdf_us_data, df_ads_data, df_schools_data)
            st.pyplot(map_figure)
            st.success("Map generated successfully!")

            fn = 'combined_map_streamlit.png'
            img_bytes = io.BytesIO()
            map_figure.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
            img_bytes.seek(0)
            st.download_button(label="Download Map as PNG", data=img_bytes, file_name=fn, mime="image/png")

        except Exception as e:
            st.error(f"An error occurred during map generation: {e}")
            st.exception(e) # Shows the full traceback for debugging
    else:
        st.warning("Map could not be generated. Please check that all uploaded files are valid and contain data, and that ZIPs in Ad/School files exist in the US ZIP Master.")
else:
    st.sidebar.info("Please upload all three CSV files to generate the map.")
    st.info("Awaiting file uploads...")

st.markdown("---")
st.markdown("This app is a Streamlit version of the `zip-code-maper.py` script.")

