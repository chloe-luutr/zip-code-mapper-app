import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
# from matplotlib.patches import Wedge
import math
import pyproj
from shapely.geometry import Point, MultiPolygon
from shapely.ops import transform, unary_union
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
import os
from pathlib import Path # Import pathlib

st.set_page_config(layout="wide")

st.title("Interactive ZIP Code Analyzer & K-12 School Mapper")
st.markdown("""
Paste your list of ZIP codes to analyze. The app will:
- Map these **Input ZIP Codes** and optionally their 5, 10, & 25-mile coverage radii to visualize potential overlaps.
- If K-12 school data is uploaded, it will highlight schools **within the chosen primary coverage radius of your Input ZIPs**.
- Optionally display current Ad Target ZIPs for context.
""")

###############################################################################
# HELPER FUNCTIONS
###############################################################################

# Construct the path to the master US ZIP code file relative to the script
BASE_DIR = Path(__file__).resolve().parent
MASTER_ZIP_FILE_PATH = BASE_DIR / "us_zip_master.csv" 

def load_us_zip_codes(csv_file_path: Path) -> gpd.GeoDataFrame: # Changed type hint to Path
    """Loads US ZIP code data from a CSV file in the repository."""
    try:
        if not csv_file_path.is_file(): # Use pathlib's is_file()
            st.error(f"Critical Error: The US ZIP Codes Master File ('{csv_file_path.name}') was not found at the expected location: {csv_file_path}")
            st.error("Please ensure 'us_zip_master.csv' (with 'zip' and 'Geo Point' or 'latitude'/'longitude' columns) is in the GitHub repository alongside the Streamlit script.")
            return gpd.GeoDataFrame()

        with open(csv_file_path, 'r', encoding='utf-8-sig') as f: # Added utf-8-sig for potential BOM
            first_line = f.readline()
        delimiter = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','


        df = pd.read_csv(csv_file_path, dtype={'zip': str, 'Zip Code': str}, delimiter=delimiter) 
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip().str.lower()
        
        zip_col_name = None
        if 'zip' in df.columns:
            zip_col_name = 'zip'
        elif 'zip code' in df.columns: 
            zip_col_name = 'zip code'
            df.rename(columns={'zip code': 'zip'}, inplace=True)
        else:
            for col in original_columns:
                processed_col = col.lower().strip()
                if processed_col == 'zip' or processed_col == 'zip code':
                    df.rename(columns={col: 'zip'}, inplace=True) # Rename original col name
                    zip_col_name = 'zip'
                    break
            if not zip_col_name:
                st.error(f"Master US ZIP file ('{csv_file_path.name}') must contain a 'zip' or 'Zip Code' column.")
                return gpd.GeoDataFrame()
        
        df['zip'] = df['zip'].astype(str).str.zfill(5)

        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        elif 'geo point' in df.columns:
            try:
                lat_lon_split = df['geo point'].astype(str).str.split(',', expand=True)
                df['latitude'] = pd.to_numeric(lat_lon_split[0], errors='coerce')
                df['longitude'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
                if df['latitude'].isnull().any() or df['longitude'].isnull().any():
                    st.error(f"Could not parse all 'Geo Point' values in '{csv_file_path.name}'. Ensure format is 'latitude,longitude'. Check for non-numeric values or incorrect splits.")
                    # st.dataframe(df[df['latitude'].isnull() | df['longitude'].isnull()]['geo point'].head()) # For debugging
                    return gpd.GeoDataFrame()
            except Exception as e:
                st.error(f"Error parsing 'Geo Point' column in '{csv_file_path.name}': {e}")
                return gpd.GeoDataFrame()
        else:
            st.error(f"Master US ZIP file ('{csv_file_path.name}') must contain 'latitude' & 'longitude' columns OR a 'Geo Point' column.")
            return gpd.GeoDataFrame()

        # Drop rows where essential coordinates are missing after parsing
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        if df.empty:
            st.error(f"No valid coordinate data found in '{csv_file_path.name}' after parsing.")
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf
    except FileNotFoundError:
        st.error(f"Critical Error: The US ZIP Codes Master File ('{csv_file_path.name}') was not found at path: {csv_file_path}. Please add it to your GitHub repository.")
        return gpd.GeoDataFrame()
    except Exception as e:
        st.error(f"Error loading US ZIP Codes Master File ('{csv_file_path.name}'): {e}")
        # st.exception(e) # For more detailed traceback during debugging
        return gpd.GeoDataFrame()

def load_ad_target_zips(uploaded_file_object) -> pd.DataFrame:
    if uploaded_file_object is None: return pd.DataFrame(columns=['zip'])
    try:
        try: 
            df_final = pd.read_csv(uploaded_file_object, dtype={'zip': str}, header=None, names=['zip'])
            if not df_final['zip'].astype(str).str.match(r'^\d{5}$').all(): 
                raise ValueError("Not a simple list of ZIPs")
        except (Exception, ValueError): 
            uploaded_file_object.seek(0)
            df_temp = pd.read_csv(uploaded_file_object, dtype={'zip': str})
            df_temp.columns = df_temp.columns.str.strip().str.lower()
            if 'zip' in df_temp.columns:
                df_final = df_temp[['zip']]
            else: 
                uploaded_file_object.seek(0)
                df_header_as_data = pd.read_csv(uploaded_file_object, dtype=str, nrows=1) 
                df_full_first_col_check = pd.read_csv(uploaded_file_object, dtype=str, usecols=[0], header=None)
                if all(str(z).isdigit() and len(str(z)) == 5 for z in df_header_as_data.iloc[0]):
                     df_final = pd.DataFrame(df_header_as_data.iloc[0].values, columns=['zip'])
                elif df_full_first_col_check.iloc[:,0].astype(str).str.match(r'^\d{5}$').all():
                     df_final = df_full_first_col_check.rename(columns={df_full_first_col_check.columns[0]:'zip'})
                else:
                    st.error("Ad Target ZIPs CSV: Could not find 'zip' column or parse as a simple list.")
                    return pd.DataFrame(columns=['zip'])
        df_final['zip'] = df_final['zip'].astype(str).str.zfill(5)
        return df_final[['zip']].drop_duplicates()
    except Exception as e:
        st.error(f"Error loading Ad Target ZIPs: {e}"); return pd.DataFrame(columns=['zip'])

def parse_input_zips(zip_code_text_input: str) -> pd.DataFrame:
    if not zip_code_text_input.strip(): return pd.DataFrame(columns=['zip'])
    zips = [z.strip() for z in pd.Series(zip_code_text_input.splitlines()).str.split(r'[\s,]+').explode() if z.strip()]
    valid_zips = [z for z in zips if z.isdigit() and (len(z) == 5 or (len(z) < 5 and z.zfill(5).isdigit()))]
    if not valid_zips:
        st.warning("No valid 5-digit ZIP codes found in the input text."); return pd.DataFrame(columns=['zip'])
    df = pd.DataFrame([z.zfill(5) for z in valid_zips], columns=['zip'])
    return df.drop_duplicates()

def load_k12_schools(uploaded_file_object) -> gpd.GeoDataFrame:
    if uploaded_file_object is None: return gpd.GeoDataFrame()
    try:
        df = pd.read_csv(uploaded_file_object)
        df.columns = df.columns.str.strip().str.lower()
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error("K-12 Schools CSV must contain 'latitude' and 'longitude' columns.")
            return gpd.GeoDataFrame()
        if 'name' not in df.columns: df['name'] = 'K-12 School'
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error loading K-12 Schools: {e}"); return gpd.GeoDataFrame()

def geodesic_buffer(lon, lat, miles):
    radius_m = miles * 1609.34 ; wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj_str = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    try: aeqd_proj = pyproj.CRS.from_proj4(aeqd_proj_str)
    except pyproj.exceptions.CRSError:
        st.warning(f"AEQD projection error for {lat},{lon}. Buffer inaccurate."); return Point(lon, lat).buffer(radius_m / 111000)
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    center = Point(lon, lat); center_aeqd = transform(project_fwd, center)
    buffer_aeqd = center_aeqd.buffer(radius_m); buffer_wgs84= transform(project_back, buffer_aeqd)
    return buffer_wgs84

def create_geodesic_buffers(gdf_points, radii=(5,10,25)): 
    if gdf_points.empty or 'geometry' not in gdf_points.columns: return gdf_points
    active_radii = [r for r in radii if r > 0] 
    for r in active_radii:
        col_name = f"buffer_{r}"
        if col_name in gdf_points.columns and gdf_points[col_name].notna().all(): continue 
        poly_list = []
        for _, row in gdf_points.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not isinstance(geom, Point): poly_list.append(None); continue
            lon, lat = geom.x, geom.y
            try: poly_list.append(geodesic_buffer(lon, lat, r))
            except Exception as e: st.warning(f"Buffer error (ZIP {row.get('zip', 'N/A')}, {r}mi): {e}"); poly_list.append(None)
        gdf_points[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326")
    return gdf_points

###############################################################################
# MAIN PLOT FUNCTION
###############################################################################
def generate_map_plot(gdf_us, df_input_zips, df_ad_targets, gdf_k12_schools=None, 
                      show_5_mile_buffer=True, show_10_mile_buffer=True, show_25_mile_buffer=False):
    if gdf_us.empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"US ZIP Data Missing or Error", ha='center'); return fig
    if df_input_zips.empty:
        st.info("Enter ZIP codes for analysis to generate map."); fig, ax = plt.subplots(); ax.text(0.5,0.5,"Enter ZIPs for Analysis", ha='center'); return fig

    gdf_input_zips_geo = pd.merge(df_input_zips, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
    if gdf_input_zips_geo.empty:
        st.warning("Input ZIPs not found in US ZIP Master File."); fig, ax = plt.subplots(); ax.text(0.5,0.5,"Input ZIPs not in US data", ha='center'); return fig
    gdf_input_zips_geo = gpd.GeoDataFrame(gdf_input_zips_geo, geometry='geometry', crs="EPSG:4326")
    
    radii_to_create = []
    if show_5_mile_buffer: radii_to_create.append(5)
    if show_10_mile_buffer: radii_to_create.append(10)
    if show_25_mile_buffer: radii_to_create.append(25)
    if radii_to_create:
        gdf_input_zips_geo = create_geodesic_buffers(gdf_input_zips_geo, radii=tuple(radii_to_create))

    gdf_ad_targets_geo = gpd.GeoDataFrame(crs="EPSG:4326")
    if not df_ad_targets.empty:
        merged_ads = pd.merge(df_ad_targets, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not merged_ads.empty: gdf_ad_targets_geo = gpd.GeoDataFrame(merged_ads, geometry='geometry', crs="EPSG:4326")

    filtered_k12_schools = gpd.GeoDataFrame(crs="EPSG:4326")
    active_school_filter_radius = 0
    if gdf_k12_schools is not None and not gdf_k12_schools.empty:
        coverage_union_for_schools = None
        # Determine the largest active buffer for filtering schools
        filter_radius_col = None
        if show_25_mile_buffer and 'buffer_25' in gdf_input_zips_geo.columns:
            filter_radius_col = 'buffer_25'; active_school_filter_radius = 25
        elif show_10_mile_buffer and 'buffer_10' in gdf_input_zips_geo.columns: 
            filter_radius_col = 'buffer_10'; active_school_filter_radius = 10
        elif show_5_mile_buffer and 'buffer_5' in gdf_input_zips_geo.columns: 
            filter_radius_col = 'buffer_5'; active_school_filter_radius = 5
        
        if filter_radius_col:
            valid_buffers = gdf_input_zips_geo[filter_radius_col].dropna()
            if not valid_buffers.empty: coverage_union_for_schools = unary_union(valid_buffers.tolist())
        
        if coverage_union_for_schools and not coverage_union_for_schools.is_empty:
            k12_proj = gdf_k12_schools.to_crs(epsg=3857) 
            coverage_proj = gpd.GeoSeries([coverage_union_for_schools], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            # Use sindex for potentially faster initial query
            possible_matches_idx = list(k12_proj.sindex.query(coverage_proj, predicate='intersects')) # intersects is safer for polygons
            if possible_matches_idx:
                candidate_schools = k12_proj.iloc[possible_matches_idx]
                # Perform precise 'within' check on candidates
                actually_within = candidate_schools.within(coverage_proj)
                filtered_k12_schools = gdf_k12_schools.iloc[candidate_schools[actually_within].index]

    gdf_input_3857    = gdf_input_zips_geo.to_crs(epsg=3857)
    gdf_ad_targets_3857 = gdf_ad_targets_geo.to_crs(epsg=3857) if not gdf_ad_targets_geo.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    
    # Plot all K12 schools if no filter radius is active, otherwise plot filtered ones
    gdf_k12_to_plot = filtered_k12_schools if active_school_filter_radius > 0 else \
                      (gdf_k12_schools if gdf_k12_schools is not None else gpd.GeoDataFrame())
    gdf_k12_plot_3857 = gdf_k12_to_plot.to_crs(epsg=3857) if not gdf_k12_to_plot.empty else gpd.GeoDataFrame(crs="EPSG:3857")


    if show_5_mile_buffer and 'buffer_5' in gdf_input_zips_geo.columns: gdf_input_3857['buffer_5_3857']  = gpd.GeoSeries(gdf_input_zips_geo['buffer_5'], crs="EPSG:4326").to_crs(epsg=3857)
    if show_10_mile_buffer and 'buffer_10' in gdf_input_zips_geo.columns: gdf_input_3857['buffer_10_3857'] = gpd.GeoSeries(gdf_input_zips_geo['buffer_10'], crs="EPSG:4326").to_crs(epsg=3857)
    if show_25_mile_buffer and 'buffer_25' in gdf_input_zips_geo.columns: gdf_input_3857['buffer_25_3857'] = gpd.GeoSeries(gdf_input_zips_geo['buffer_25'], crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(16,13))
    all_geoms_for_bounds = [gdf_input_3857, gdf_ad_targets_3857, gdf_k12_plot_3857]
    valid_geoms_for_bounds = [g for g in all_geoms_for_bounds if g is not None and not g.empty and hasattr(g, 'total_bounds') and g.total_bounds is not None]
    if not valid_geoms_for_bounds: minx, miny, maxx, maxy = -13e6, 2.5e6, -7e6, 6.5e6
    else:
        bounds_list = [gdf.total_bounds for gdf in valid_geoms_for_bounds]; minx, miny, maxx, maxy = (min(b[0] for b in bounds_list), min(b[1] for b in bounds_list), max(b[2] for b in bounds_list), max(b[3] for b in bounds_list)) if bounds_list else (-13e6, 2.5e6, -7e6, 6.5e6)
    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6; pad_x, pad_y = 0.15 * w, 0.15 * h

    gdf_input_3857.plot(ax=ax, marker='*', color='crimson', markersize=200, label="Input ZIPs", zorder=5, edgecolor='black')
    if show_5_mile_buffer and 'buffer_5_3857' in gdf_input_3857.columns and gdf_input_3857['buffer_5_3857'].notna().any():
        gdf_input_3857[gdf_input_3857['buffer_5_3857'].notna()].plot(ax=ax, facecolor='orangered', edgecolor='orangered', alpha=0.15, linewidth=1.0, zorder=2, linestyle='--')
    if show_10_mile_buffer and 'buffer_10_3857' in gdf_input_3857.columns and gdf_input_3857['buffer_10_3857'].notna().any():
        gdf_input_3857[gdf_input_3857['buffer_10_3857'].notna()].plot(ax=ax, facecolor='darkorange', edgecolor='darkorange', alpha=0.1, linewidth=1.5, zorder=1, linestyle=':')
    if show_25_mile_buffer and 'buffer_25_3857' in gdf_input_3857.columns and gdf_input_3857['buffer_25_3857'].notna().any():
        gdf_input_3857[gdf_input_3857['buffer_25_3857'].notna()].plot(ax=ax, facecolor='gold', edgecolor='gold', alpha=0.08, linewidth=2.0, zorder=0, linestyle='-')
        
    if not gdf_ad_targets_3857.empty: gdf_ad_targets_3857.plot(ax=ax, marker='s', color='limegreen', markersize=70, label="Ad Target ZIPs", zorder=4, alpha=0.8, edgecolor='darkgreen')
    
    k12_label_for_plot = "K-12 Schools"
    if not gdf_k12_plot_3857.empty:
        if active_school_filter_radius > 0: k12_label_for_plot = f"K-12 Schools (in {active_school_filter_radius}mi radius of Input ZIPs)"
        else: k12_label_for_plot = f"K-12 Schools (All Uploaded)" 
        gdf_k12_plot_3857.plot(ax=ax, marker='^', color='deepskyblue', markersize=50, label=k12_label_for_plot, zorder=3, alpha=0.9, edgecolor='black')
    elif gdf_k12_schools is not None and not gdf_k12_schools.empty and active_school_filter_radius > 0 : 
        st.info(f"No K-12 schools found within the {active_school_filter_radius}-mile radius of the input ZIP codes.")

    try: ctx.add_basemap(ax, crs=gdf_input_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=5)
    except Exception as e: st.warning(f"Could not add basemap: {e}")
    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y); ax.axis('off')

    handles, labels = [], []
    handles.append(mlines.Line2D([], [], color='crimson', marker='*', linestyle='None', markersize=12, label='Input ZIPs', markeredgecolor='black')); labels.append(f'Input ZIPs ({len(gdf_input_3857)})')
    if show_5_mile_buffer and 'buffer_5_3857' in gdf_input_3857: handles.append(mpatches.Patch(facecolor='orangered', alpha=0.3, edgecolor='orangered', linestyle='--', label='5mi Input Coverage')); labels.append('5-mile Input Coverage')
    if show_10_mile_buffer and 'buffer_10_3857' in gdf_input_3857: handles.append(mpatches.Patch(facecolor='darkorange', alpha=0.2, edgecolor='darkorange', linestyle=':', label='10mi Input Coverage')); labels.append('10-mile Input Coverage')
    if show_25_mile_buffer and 'buffer_25_3857' in gdf_input_3857: handles.append(mpatches.Patch(facecolor='gold', alpha=0.2, edgecolor='gold', linestyle='-', label='25mi Input Coverage')); labels.append('25-mile Input Coverage')
    if not gdf_ad_targets_3857.empty: handles.append(mlines.Line2D([], [], color='limegreen', marker='s', linestyle='None', markersize=8, label='Ad Target ZIPs', markeredgecolor='darkgreen')); labels.append(f'Ad Target ZIPs ({len(gdf_ad_targets_3857)})')
    if not gdf_k12_plot_3857.empty:
        current_k12_label_for_legend = "K-12 Schools"
        if active_school_filter_radius > 0: current_k12_label_for_legend = f'K-12 Schools ({len(gdf_k12_plot_3857)} in {active_school_filter_radius}mi radius)'
        else: current_k12_label_for_legend = f'K-12 Schools ({len(gdf_k12_plot_3857)} Uploaded)'
        handles.append(mlines.Line2D([], [], color='deepskyblue', marker='^', linestyle='None', markersize=8, label=current_k12_label_for_legend, markeredgecolor='black')); labels.append(current_k12_label_for_legend)

    if handles: ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small', title="Legend", title_fontsize="medium")
    ax.set_title("Input ZIP Code Coverage & K-12 School Proximity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.83, 1]); return fig

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
st.sidebar.header("1. Enter/Paste ZIP Codes for Analysis") 
zip_code_input_text = st.sidebar.text_area("Paste the list of ZIP codes you want to analyze (comma, space, or newline separated). The app will map these specific ZIPs and their coverage.", height=150, key="zip_input_area_v7", # Incremented key
help="Enter 5-digit ZIP codes.")

st.sidebar.header("2. Display Options for Input ZIPs")
show_5_mile = st.sidebar.checkbox("Show 5-mile radius", value=True, key="show_5_v7")
show_10_mile = st.sidebar.checkbox("Show 10-mile radius", value=True, key="show_10_v7")
show_25_mile = st.sidebar.checkbox("Show 25-mile radius (Indeed Ad Range)", value=False, key="show_25_v7")

st.sidebar.header("3. Upload Optional Data (CSV)") 
uploaded_ad_targets_file = st.sidebar.file_uploader("Ad Target ZIPs (Optional: zip)", type="csv", key="ad_targets_v7")
uploaded_k12_schools_file = st.sidebar.file_uploader("K-12 School Locations (Optional: name, latitude, longitude)", type="csv", key="k12_schools_v7")

# Load the master US ZIP code data automatically
# This is now done once at the beginning of the script if MASTER_ZIP_FILE_PATH is valid
gdf_us_data = None # Initialize
if 'gdf_us_data_loaded' not in st.session_state: # Load only once per session
    gdf_us_data = load_us_zip_codes(MASTER_ZIP_FILE_PATH)
    if not gdf_us_data.empty:
        st.session_state.gdf_us_data_loaded = gdf_us_data
    else:
        st.session_state.gdf_us_data_loaded = gpd.GeoDataFrame() # Store empty if failed
        st.error("Failed to load the built-in US ZIP Codes Master File. Please check the file in the repository (expected: 'us_zip_master.csv') and its format. The app cannot function without it.")
        st.stop()
else:
    gdf_us_data = st.session_state.gdf_us_data_loaded


if gdf_us_data.empty: # Double check, in case it failed to load into session state
    st.error("US ZIP Codes Master File could not be loaded. App cannot proceed.")
    st.stop()


if zip_code_input_text.strip():
    st.sidebar.success("ZIP codes for analysis provided!")
    
    df_input_zips_data = parse_input_zips(zip_code_input_text)
    df_ad_targets_data = load_ad_target_zips(uploaded_ad_targets_file) if uploaded_ad_targets_file else pd.DataFrame(columns=['zip'])
    gdf_k12_schools_data = load_k12_schools(uploaded_k12_schools_file) if uploaded_k12_schools_file else gpd.GeoDataFrame()

    data_load_success = True
    if df_input_zips_data.empty: st.warning("No valid ZIPs parsed from input text area."); data_load_success = False 
    
    if uploaded_ad_targets_file and df_ad_targets_data.empty and uploaded_ad_targets_file is not None : st.warning("Problem loading 'Ad Target ZIPs'. It will be excluded.")
    if uploaded_k12_schools_file and gdf_k12_schools_data.empty and uploaded_k12_schools_file is not None: st.warning("Problem loading 'K-12 Schools'. It will be excluded.")

    if data_load_success:
        st.info("Data ready. Generating map...")
        try:
            map_figure = generate_map_plot(gdf_us_data, df_input_zips_data, df_ad_targets_data, gdf_k12_schools_data,
                                           show_5_mile_buffer=show_5_mile, show_10_mile_buffer=show_10_mile, show_25_mile_buffer=show_25_mile)
            st.pyplot(map_figure) ; st.success("Map generated successfully!")
            fn = 'zip_analysis_map_v7.png'; img = io.BytesIO()
            map_figure.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button(label="Download Map as PNG", data=img, file_name=fn, mime="image/png")
        except Exception as e: st.error(f"Error during map generation: {e}"); st.exception(e)
    elif df_input_zips_data.empty : st.warning("Provide valid ZIPs in the text area to generate map.")
    # gdf_us_data failure is handled above with st.stop()
else:
    st.sidebar.info("Paste ZIP codes for analysis in Section 1 to generate the map.") 
    st.info("Awaiting ZIP code input for analysis...")
st.markdown("---")
st.markdown("Visualizes ZIP code coverage and K-12 school proximity for strategic planning.")
