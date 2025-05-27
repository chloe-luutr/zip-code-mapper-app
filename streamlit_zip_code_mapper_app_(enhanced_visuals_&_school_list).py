import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import math
import pyproj
from shapely.geometry import Point, MultiPolygon
from shapely.ops import transform, unary_union
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
from pathlib import Path

st.set_page_config(layout="wide")

st.title("Interactive ZIP Code Analyzer & K-12 School Mapper")
st.markdown("""
Paste your list of ZIP codes to analyze. The app will:
- Map these **Input ZIP Codes** and optionally their selected coverage radius (as filled circles) to visualize potential overlaps.
- If "Suggest Optimal ZIPs" is selected, it will recommend a subset of your input ZIPs for efficient 25-mile coverage.
- Display K-12 schools (from built-in data) **within the chosen primary coverage radius of your Input (or Recommended) ZIPs**.
- Optionally display current Ad Target ZIPs for context.
""")

###############################################################################
# HELPER FUNCTIONS
###############################################################################

BASE_DIR = Path(__file__).resolve().parent
MASTER_ZIP_FILE_PATH = BASE_DIR / "us_zip_master.csv" 
K12_SCHOOLS_FILE_PATH = BASE_DIR / "my_k12_schools.csv"

@st.cache_data
def load_us_zip_codes_cached(csv_file_path: Path) -> gpd.GeoDataFrame:
    try:
        if not csv_file_path.is_file(): return gpd.GeoDataFrame()
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f: first_line = f.readline()
        delimiter = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
        df = pd.read_csv(csv_file_path, dtype={'zip': str, 'Zip Code': str}, delimiter=delimiter) 
        original_columns = list(df.columns); df.columns = df.columns.str.strip().str.lower()
        zip_col_name = None
        if 'zip' in df.columns: zip_col_name = 'zip'
        elif 'zip code' in df.columns: df.rename(columns={'zip code': 'zip'}, inplace=True); zip_col_name = 'zip'
        else:
            for col in original_columns:
                processed_col = col.lower().strip()
                if processed_col == 'zip' or processed_col == 'zip code':
                    df.rename(columns={col: 'zip'}, inplace=True); zip_col_name = 'zip'; break
            if not zip_col_name: st.error(f"Master US ZIP file ('{csv_file_path.name}') needs 'zip'/'Zip Code' column."); return gpd.GeoDataFrame()
        df['zip'] = df['zip'].astype(str).str.zfill(5)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce'); df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        elif 'geo point' in df.columns:
            try:
                lat_lon_split = df['geo point'].astype(str).str.split(',', expand=True)
                df['latitude'] = pd.to_numeric(lat_lon_split[0], errors='coerce'); df['longitude'] = pd.to_numeric(lat_lon_split[1], errors='coerce')
                if df['latitude'].isnull().any() or df['longitude'].isnull().any():
                    st.error(f"Could not parse 'Geo Point' in '{csv_file_path.name}'."); return gpd.GeoDataFrame()
            except Exception as e: st.error(f"Error parsing 'Geo Point' in '{csv_file_path.name}': {e}"); return gpd.GeoDataFrame()
        else: st.error(f"Master US ZIP file ('{csv_file_path.name}') needs 'lat'/'lon' or 'Geo Point'."); return gpd.GeoDataFrame()
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        if df.empty: st.error(f"No valid coordinate data in '{csv_file_path.name}'."); return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    except FileNotFoundError: st.error(f"US ZIP Master File ('{csv_file_path.name}') not found."); return gpd.GeoDataFrame()
    except Exception as e: st.error(f"Error loading US ZIP Master File ('{csv_file_path.name}'): {e}"); return gpd.GeoDataFrame()

@st.cache_data
def load_k12_schools_cached(csv_file_path: Path) -> gpd.GeoDataFrame:
    try:
        if not csv_file_path.is_file(): return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f: first_line = f.readline()
        delimiter = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
        df = pd.read_csv(csv_file_path, delimiter=delimiter)
        df.columns = df.columns.str.strip().str.lower()
        lat_col, lon_col, name_col = None, None, None
        possible_lat_names = ['latitude', 'lat', 'y', 'ycoord']; possible_lon_names = ['longitude', 'lon', 'long', 'x', 'xcoord']
        possible_name_cols = ['name', 'sch_name', 'school_name', 'schoolname', 'leanm']
        for p_lat in possible_lat_names:
            if p_lat in df.columns: lat_col = p_lat; break
        for p_lon in possible_lon_names:
            if p_lon in df.columns: lon_col = p_lon; break
        for p_name in possible_name_cols:
            if p_name in df.columns: name_col = p_name; break
        if not (lat_col and lon_col): st.warning(f"K-12 Schools file ('{csv_file_path.name}') needs lat/lon. Not loaded."); return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")
        df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce'); df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
        if name_col: df['name'] = df[name_col].astype(str)
        else: df['name'] = df.iloc[:, 0].astype(str) if not df.empty else 'K-12 School'
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        if df.empty: st.warning(f"No valid K-12 school coordinates in '{csv_file_path.name}'."); return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf[['name', 'geometry']]
    except FileNotFoundError: st.warning(f"K-12 Schools file ('{csv_file_path.name}') not found."); return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")
    except Exception as e: st.warning(f"Error loading K-12 Schools ('{csv_file_path.name}'): {e}."); return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

def load_ad_target_zips(uploaded_file_object) -> pd.DataFrame:
    if uploaded_file_object is None: return pd.DataFrame(columns=['zip'])
    # Simplified parsing, assuming a single column of zips or a 'zip' header
    try:
        df = pd.read_csv(uploaded_file_object, dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        if 'zip' in df.columns:
            df_final = df[['zip']]
        elif df.shape[1] == 1: # Assume first column is zips if only one column
            df_final = df.rename(columns={df.columns[0]: 'zip'})[['zip']]
        else:
            st.error("Ad Target ZIPs CSV: Could not find 'zip' column or parse as a simple list.")
            return pd.DataFrame(columns=['zip'])
        df_final['zip'] = df_final['zip'].str.zfill(5).str.strip()
        df_final = df_final[df_final['zip'].str.match(r'^\d{5}$')].drop_duplicates()
        return df_final
    except Exception as e: st.error(f"Error loading Ad Target ZIPs: {e}"); return pd.DataFrame(columns=['zip'])

def parse_input_zips(zip_code_text_input: str) -> pd.DataFrame:
    if not zip_code_text_input.strip(): return pd.DataFrame(columns=['zip'])
    zips = [z.strip() for z in pd.Series(zip_code_text_input.splitlines()).str.split(r'[\s,]+').explode() if z.strip().isdigit()]
    valid_zips = [z.zfill(5) for z in zips if len(z.zfill(5)) == 5]
    if not valid_zips: st.warning("No valid 5-digit ZIP codes found in input."); return pd.DataFrame(columns=['zip'])
    return pd.DataFrame(list(set(valid_zips)), columns=['zip']) # Use set for unique

def geodesic_buffer(lon, lat, miles):
    radius_m = miles * 1609.34; wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj_str = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    try: aeqd_proj = pyproj.CRS.from_proj4(aeqd_proj_str)
    except pyproj.exceptions.CRSError: return Point(lon, lat).buffer(radius_m / 111000) # Fallback
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    return transform(project_back, transform(project_fwd, Point(lon, lat)).buffer(radius_m))

def create_geodesic_buffers(gdf_points, radius_miles=0): 
    if gdf_points.empty or 'geometry' not in gdf_points.columns or radius_miles == 0: return gdf_points
    col_name = f"buffer_{radius_miles}"
    poly_list = [geodesic_buffer(row.geometry.x, row.geometry.y, radius_miles) if row.geometry else None for _, row in gdf_points.iterrows()]
    gdf_points[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326")
    return gdf_points

def suggest_optimal_zips(gdf_input_zips_geo: gpd.GeoDataFrame, num_recommendations: int, radius_miles: int = 25) -> gpd.GeoDataFrame:
    if gdf_input_zips_geo.empty or 'geometry' not in gdf_input_zips_geo.columns:
        return gpd.GeoDataFrame()

    # Create 25-mile buffers for all input zips for coverage calculation
    # Ensure it's done on a projected CRS for accurate area/overlap if needed, but here for point coverage
    temp_gdf = gdf_input_zips_geo.copy()
    temp_gdf = create_geodesic_buffers(temp_gdf, radius_miles=radius_miles)
    buffer_col = f'buffer_{radius_miles}'
    if buffer_col not in temp_gdf.columns: return gpd.GeoDataFrame() # Buffers failed

    all_input_points = temp_gdf.geometry.copy() # These are the centers of input ZIPs
    covered_points_indices = set()
    recommended_zips_indices = []

    for _ in range(num_recommendations):
        best_candidate_idx = -1
        max_newly_covered = -1

        # Find the input ZIP (not yet recommended) that covers the most *new* input ZIP centers
        for idx, row in temp_gdf.iterrows():
            if idx in recommended_zips_indices:
                continue # Already recommended

            current_buffer = row[buffer_col]
            if current_buffer is None or current_buffer.is_empty:
                continue
            
            # Count how many *not-yet-covered* input points this buffer covers
            newly_covered_count = 0
            for point_idx, point_geom in all_input_points.items():
                if point_idx not in covered_points_indices and point_geom.within(current_buffer):
                    newly_covered_count += 1
            
            if newly_covered_count > max_newly_covered:
                max_newly_covered = newly_covered_count
                best_candidate_idx = idx
        
        if best_candidate_idx != -1 and max_newly_covered > 0 : # Found a good candidate
            recommended_zips_indices.append(best_candidate_idx)
            # Mark points covered by this new recommendation
            best_candidate_buffer = temp_gdf.loc[best_candidate_idx, buffer_col]
            for point_idx, point_geom in all_input_points.items():
                 if point_idx not in covered_points_indices and point_geom.within(best_candidate_buffer):
                    covered_points_indices.add(point_idx)
        else:
            break # No more candidates that cover new points, or no candidates left

    if not recommended_zips_indices: return gpd.GeoDataFrame()
    return gdf_input_zips_geo.loc[recommended_zips_indices].copy()


###############################################################################
# MAIN PLOT FUNCTION
###############################################################################
def generate_map_plot(gdf_us, df_input_zips, df_ad_targets, gdf_k12_schools_repo=None, 
                      primary_radius_miles=0, show_legend=True,
                      gdf_recommended_zips=None): # Added gdf_recommended_zips
    
    plotted_school_names = [] 
    if gdf_us.empty: fig, ax = plt.subplots(); ax.text(0.5,0.5,"US ZIP Data Missing", ha='center'); return fig, plotted_school_names
    if df_input_zips.empty: fig, ax = plt.subplots(); ax.text(0.5,0.5,"Enter ZIPs for Analysis", ha='center'); return fig, plotted_school_names

    gdf_input_zips_geo = pd.merge(df_input_zips, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
    if gdf_input_zips_geo.empty: fig, ax = plt.subplots(); ax.text(0.5,0.5,"Input ZIPs not in US data", ha='center'); return fig, plotted_school_names
    gdf_input_zips_geo = gpd.GeoDataFrame(gdf_input_zips_geo, geometry='geometry', crs="EPSG:4326")
    
    # Create primary buffer for ALL input zips if a radius is selected
    if primary_radius_miles > 0:
        gdf_input_zips_geo = create_geodesic_buffers(gdf_input_zips_geo, radius_miles=primary_radius_miles)

    # If recommendations are active, prepare their 25-mile buffers for distinct plotting
    gdf_recommended_zips_with_25mi_buffer = gpd.GeoDataFrame()
    if gdf_recommended_zips is not None and not gdf_recommended_zips.empty:
        gdf_recommended_zips_with_25mi_buffer = pd.merge(gdf_recommended_zips[['zip']], gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not gdf_recommended_zips_with_25mi_buffer.empty:
            gdf_recommended_zips_with_25mi_buffer = gpd.GeoDataFrame(gdf_recommended_zips_with_25mi_buffer, geometry='geometry', crs="EPSG:4326")
            gdf_recommended_zips_with_25mi_buffer = create_geodesic_buffers(gdf_recommended_zips_with_25mi_buffer, radius_miles=25)


    gdf_ad_targets_geo = gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:4326") 
    if not df_ad_targets.empty:
        merged_ads = pd.merge(df_ad_targets, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not merged_ads.empty: gdf_ad_targets_geo = gpd.GeoDataFrame(merged_ads, geometry='geometry', crs="EPSG:4326")

    filtered_k12_schools = gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")
    active_school_filter_radius = 0
    coverage_source_gdf = gdf_recommended_zips_with_25mi_buffer if gdf_recommended_zips is not None and not gdf_recommended_zips.empty and 'buffer_25' in gdf_recommended_zips_with_25mi_buffer else gdf_input_zips_geo
    radius_col_for_school_filter = f'buffer_{25 if gdf_recommended_zips is not None and not gdf_recommended_zips.empty else primary_radius_miles}'
    
    if gdf_k12_schools_repo is not None and not gdf_k12_schools_repo.empty and radius_col_for_school_filter in coverage_source_gdf.columns and primary_radius_miles > 0 :
        valid_buffers = coverage_source_gdf[radius_col_for_school_filter].dropna()
        if not valid_buffers.empty:
            coverage_union_for_schools = unary_union(valid_buffers.tolist())
            if coverage_union_for_schools and not coverage_union_for_schools.is_empty:
                k12_proj = gdf_k12_schools_repo.to_crs(epsg=3857) 
                coverage_proj = gpd.GeoSeries([coverage_union_for_schools], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
                possible_matches_idx = list(k12_proj.sindex.query(coverage_proj, predicate='intersects')) 
                if possible_matches_idx:
                    candidate_schools = k12_proj.iloc[possible_matches_idx]
                    actually_within = candidate_schools.within(coverage_proj)
                    filtered_k12_schools = gdf_k12_schools_repo.iloc[candidate_schools[actually_within].index].copy()
                    active_school_filter_radius = 25 if gdf_recommended_zips is not None and not gdf_recommended_zips.empty else primary_radius_miles


    # --- Projections ---
    gdf_input_3857    = gdf_input_zips_geo.to_crs(epsg=3857)
    gdf_ad_targets_3857 = gdf_ad_targets_geo.to_crs(epsg=3857) if not gdf_ad_targets_geo.empty else gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:3857")
    gdf_k12_plot_3857 = filtered_k12_schools.to_crs(epsg=3857) if not filtered_k12_schools.empty else \
                       (gdf_k12_schools_repo.to_crs(epsg=3857) if (gdf_k12_schools_repo is not None and not gdf_k12_schools_repo.empty and active_school_filter_radius == 0) \
                        else gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:3857"))
    gdf_recommended_3857 = gdf_recommended_zips_with_25mi_buffer.to_crs(epsg=3857) if not gdf_recommended_zips_with_25mi_buffer.empty else gpd.GeoDataFrame(columns=['zip', 'geometry'], crs="EPSG:3857")


    # Project primary buffer for all input zips
    primary_buffer_col_3857 = None
    if primary_radius_miles > 0 and f'buffer_{primary_radius_miles}' in gdf_input_zips_geo.columns:
        primary_buffer_col_3857 = f'buffer_{primary_radius_miles}_3857'
        gdf_input_3857[primary_buffer_col_3857] = gpd.GeoSeries(gdf_input_zips_geo[f'buffer_{primary_radius_miles}'], crs="EPSG:4326").to_crs(epsg=3857)
    
    # Project 25-mile buffer specifically for recommended zips if they exist
    if 'buffer_25' in gdf_recommended_3857.columns: # gdf_recommended_zips_with_25mi_buffer was projected
         gdf_recommended_3857['buffer_25_3857'] = gdf_recommended_3857['buffer_25'] # Already projected

    fig, ax = plt.subplots(figsize=(16,13))
    all_geoms_for_bounds = [gdf_input_3857, gdf_ad_targets_3857, gdf_k12_plot_3857, gdf_recommended_3857]
    # ... (rest of bounds calculation as before) ...
    valid_geoms_for_bounds = [g for g in all_geoms_for_bounds if g is not None and not g.empty and hasattr(g, 'total_bounds') and g.total_bounds is not None]
    if not valid_geoms_for_bounds: minx, miny, maxx, maxy = -13e6, 2.5e6, -7e6, 6.5e6
    else:
        bounds_list = [gdf.total_bounds for gdf in valid_geoms_for_bounds]; minx, miny, maxx, maxy = (min(b[0] for b in bounds_list), min(b[1] for b in bounds_list), max(b[2] for b in bounds_list), max(b[3] for b in bounds_list)) if bounds_list else (-13e6, 2.5e6, -7e6, 6.5e6)
    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6; pad_x, pad_y = 0.15 * w, 0.15 * h


    # Plot primary buffers for ALL input ZIPs if selected
    buffer_colors = {5: ('red', 'darkred', 0.25), 10: ('yellow', 'orange', 0.25), 25: ('lightseagreen', 'darkcyan', 0.2)}
    if primary_radius_miles > 0 and primary_buffer_col_3857 and gdf_input_3857[primary_buffer_col_3857].notna().any():
        if gdf_recommended_zips is None or gdf_recommended_zips.empty: # Only plot all if not in recommendation mode for this radius
            color_face, color_edge, alpha_val = buffer_colors.get(primary_radius_miles, ('gray', 'black', 0.2))
            gdf_input_3857[gdf_input_3857[primary_buffer_col_3857].notna()].plot(
                ax=ax, facecolor=color_face, edgecolor=color_edge, alpha=alpha_val, 
                linewidth=0.5, zorder=primary_radius_miles # Simple zorder based on size
            )
    
    # Plot 25-mile buffers for RECOMMENDED ZIPs if recommendations are active
    if gdf_recommended_zips is not None and not gdf_recommended_zips.empty and 'buffer_25_3857' in gdf_recommended_3857.columns and gdf_recommended_3857['buffer_25_3857'].notna().any():
        color_face, color_edge, alpha_val = buffer_colors.get(25)
        gdf_recommended_3857[gdf_recommended_3857['buffer_25_3857'].notna()].plot(
            ax=ax, facecolor=color_face, edgecolor=color_edge, alpha=alpha_val + 0.1, # Make recommended buffers slightly more prominent
            linewidth=1, zorder=26 # Ensure these are prominent
        )

    # Plot Input ZIPs (all of them)
    gdf_input_3857.plot(ax=ax, marker='o', color='red', markersize=50, label="Input ZIPs", zorder=30, edgecolor='black', linewidth=0.5)

    # Highlight Recommended ZIPs
    if gdf_recommended_zips is not None and not gdf_recommended_zips.empty:
        # Get the geometries of recommended zips from gdf_input_3857 to plot them
        recommended_plot_gdf = gdf_input_3857[gdf_input_3857['zip'].isin(gdf_recommended_zips['zip'])]
        if not recommended_plot_gdf.empty:
            recommended_plot_gdf.plot(ax=ax, marker='*', color='blue', markersize=350, label="Recommended Optimal ZIPs", zorder=35, edgecolor='white', linewidth=0.7)
        
    if not gdf_ad_targets_3857.empty: gdf_ad_targets_3857.plot(ax=ax, marker='s', color='limegreen', markersize=70, label="Ad Target ZIPs", zorder=28, alpha=0.8, edgecolor='darkgreen')
    
    k12_label_for_plot = "K-12 Schools"
    if not gdf_k12_plot_3857.empty:
        if active_school_filter_radius > 0: k12_label_for_plot = f"K-12 Schools (in {active_school_filter_radius}mi radius)"
        else: k12_label_for_plot = f"K-12 Schools (All Loaded)" 
        gdf_k12_plot_3857.plot(ax=ax, marker='^', color='dodgerblue', markersize=60, label=k12_label_for_plot, zorder=27, alpha=0.9, edgecolor='black', linewidth=0.5)
        plotted_school_names = gdf_k12_plot_3857['name'].tolist()
    elif gdf_k12_schools_repo is not None and not gdf_k12_schools_repo.empty and active_school_filter_radius > 0 : 
        st.info(f"No K-12 schools found within the {active_school_filter_radius}-mile radius.")

    try: ctx.add_basemap(ax, crs=gdf_input_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=5)
    except Exception as e: st.warning(f"Could not add basemap: {e}")
    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y); ax.axis('off')

    if show_legend:
        handles, labels = [], []
        handles.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Input ZIPs', markeredgecolor='black')); labels.append(f'Input ZIPs ({len(gdf_input_3857)})')
        if gdf_recommended_zips is not None and not gdf_recommended_zips.empty:
             handles.append(mlines.Line2D([], [], color='blue', marker='*', linestyle='None', markersize=12, label='Recommended Optimal ZIPs', markeredgecolor='white')); labels.append(f'Recommended Optimal ({len(gdf_recommended_zips)})')

        if primary_radius_miles > 0 and primary_buffer_col_3857:
            color_face, color_edge, alpha_val = buffer_colors.get(primary_radius_miles)
            handles.append(mpatches.Patch(facecolor=color_face, alpha=alpha_val, edgecolor=color_edge, label=f'{primary_radius_miles}mi Input Coverage')); labels.append(f'{primary_radius_miles}-mile Input Coverage')
        
        # Add legend for recommended 25-mile buffers if they are shown distinctly
        if gdf_recommended_zips is not None and not gdf_recommended_zips.empty and 'buffer_25_3857' in gdf_recommended_3857.columns:
            if primary_radius_miles != 25: # Only add if not already covered by primary radius legend
                color_face_rec, color_edge_rec, alpha_val_rec = buffer_colors.get(25)
                handles.append(mpatches.Patch(facecolor=color_face_rec, alpha=alpha_val_rec + 0.1, edgecolor=color_edge_rec, label='25mi Recommended Coverage')); labels.append('25-mile Recommended Coverage')


        if not gdf_ad_targets_3857.empty: handles.append(mlines.Line2D([], [], color='limegreen', marker='s', linestyle='None', markersize=8, label='Ad Target ZIPs', markeredgecolor='darkgreen')); labels.append(f'Ad Target ZIPs ({len(gdf_ad_targets_3857)})')
        if not gdf_k12_plot_3857.empty:
            current_k12_label_for_legend = f'K-12 Schools ({len(gdf_k12_plot_3857)} shown)'
            handles.append(mlines.Line2D([], [], color='dodgerblue', marker='^', linestyle='None', markersize=8, label=current_k12_label_for_legend, markeredgecolor='black')); labels.append(current_k12_label_for_legend)
        if handles: ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small', title="Legend", title_fontsize="medium")
    
    ax.set_title("ZIP Code Analysis & School Proximity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.83, 1]); return fig, plotted_school_names

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
st.sidebar.header("1. Enter/Paste ZIP Codes for Analysis") 
zip_code_input_text = st.sidebar.text_area("Paste list of ZIP codes to analyze (comma, space, or newline separated).", height=150, key="zip_input_area_v9")

st.sidebar.header("2. Display Options")
selected_radius = st.sidebar.radio(
    "Select Primary Coverage Radius for Input ZIPs:",
    options=[0, 5, 10, 25], 
    format_func=lambda x: f"{x}-mile radius" if x > 0 else "No Buffer",
    index=2, key="radius_select_v9" # Default to 10 miles
)
show_map_legend = st.sidebar.checkbox("Show Map Legend", value=True, key="show_legend_v9")

st.sidebar.header("3. Suggest Optimal ZIPs (25-mile Ad Range)")
run_recommendation = st.sidebar.checkbox("Suggest Optimal Input ZIPs", value=False, key="run_rec_v9")
num_to_recommend = 7 # Default
if run_recommendation:
    num_to_recommend = st.sidebar.number_input("Number of ZIPs to suggest (3-15):", min_value=3, max_value=15, value=7, step=1, key="num_rec_v9")

st.sidebar.header("4. Upload Optional Ad Target ZIPs (CSV)") 
uploaded_ad_targets_file = st.sidebar.file_uploader("Ad Target ZIPs (Optional: zip)", type="csv", key="ad_targets_v9")

# Load master data
if 'gdf_us_data_loaded' not in st.session_state:
    st.session_state.gdf_us_data_loaded = load_us_zip_codes_cached(MASTER_ZIP_FILE_PATH)
    if st.session_state.gdf_us_data_loaded.empty: st.error("FATAL: US ZIP Master File missing/error. App cannot proceed."); st.stop()
gdf_us_data = st.session_state.gdf_us_data_loaded

if 'gdf_k12_schools_loaded' not in st.session_state:
    st.session_state.gdf_k12_schools_loaded = load_k12_schools_cached(K12_SCHOOLS_FILE_PATH)
gdf_k12_schools_repo_data = st.session_state.gdf_k12_schools_loaded

# Main app logic
if zip_code_input_text.strip():
    st.sidebar.success("ZIP codes for analysis provided!")
    df_input_zips_data = parse_input_zips(zip_code_input_text)
    df_ad_targets_data = load_ad_target_zips(uploaded_ad_targets_file) if uploaded_ad_targets_file else pd.DataFrame(columns=['zip'])
    
    recommended_zips_df = None
    if run_recommendation and not df_input_zips_data.empty:
        # Merge with geo data to pass to suggestion function
        gdf_input_for_rec = pd.merge(df_input_zips_data, gdf_us_data[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not gdf_input_for_rec.empty:
            gdf_input_for_rec = gpd.GeoDataFrame(gdf_input_for_rec, geometry='geometry', crs="EPSG:4326")
            recommended_zips_df = suggest_optimal_zips(gdf_input_for_rec, num_to_recommend, radius_miles=25) # Use 25 miles for recommendation logic

    if not df_input_zips_data.empty:
        st.info("Data ready. Generating map...")
        try:
            map_figure, plotted_schools = generate_map_plot(
                gdf_us_data, df_input_zips_data, df_ad_targets_data, gdf_k12_schools_repo_data,
                primary_radius_miles=selected_radius, 
                show_legend=show_map_legend,
                gdf_recommended_zips=recommended_zips_df
            )
            st.pyplot(map_figure)
            st.success("Map generated successfully!")

            if recommended_zips_df is not None and not recommended_zips_df.empty:
                st.subheader(f"Top {len(recommended_zips_df)} Recommended Optimal Input ZIPs (for 25-mile efficiency):")
                st.dataframe(recommended_zips_df[['zip']], height=min(300, (len(recommended_zips_df) + 1) * 35))
            
            if plotted_schools:
                st.subheader(f"K-12 Schools Plotted ({len(plotted_schools)}):")
                school_df_to_display = pd.DataFrame(plotted_schools, columns=["School Name"])
                st.dataframe(school_df_to_display, height=min(300, (len(plotted_schools) + 1) * 35))
            elif gdf_k12_schools_repo_data is not None and not gdf_k12_schools_repo_data.empty and selected_radius > 0 :
                 st.info(f"No K-12 schools from built-in list found within selected {selected_radius}-mile radius of input ZIPs.")

            fn = 'zip_analysis_map_v10.png'; img = io.BytesIO()
            map_figure.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button(label="Download Map as PNG", data=img, file_name=fn, mime="image/png")
        except Exception as e: st.error(f"Error during map generation: {e}"); st.exception(e)
    else:
        st.warning("Please enter valid ZIP codes in Section 1 to generate the map.")
else:
    st.sidebar.info("Paste ZIP codes for analysis in Section 1 to generate the map.") 
    st.info("Awaiting ZIP code input for analysis...")
st.markdown("---")
st.markdown("Visualizes ZIP code coverage, K-12 school proximity, and suggests optimal ZIPs for strategic planning.")
