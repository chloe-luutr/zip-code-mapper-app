import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
# from matplotlib.patches import Wedge # No longer needed as pie charts are removed
import math
import pyproj
from shapely.geometry import Point, MultiPolygon
from shapely.ops import transform, unary_union
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io

st.set_page_config(layout="wide")

st.title("Interactive ZIP Code Analyzer & K-12 School Mapper")
st.markdown("""
Paste your list of primary ZIP codes. The app will:
- Map these **Input ZIP Codes** and their 5 & 10-mile coverage radii to help visualize potential overlaps.
- If K-12 school data is uploaded, it will highlight schools **within the 10-mile radius of your Input ZIPs**.
- Optionally display current Ad Target ZIPs for context.
""")

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_us_zip_codes(uploaded_file_object) -> gpd.GeoDataFrame:
    if uploaded_file_object is None: return gpd.GeoDataFrame()
    try:
        df = pd.read_csv(uploaded_file_object, dtype={'zip': str})
        df.columns = df.columns.str.strip().str.lower()
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            st.error("US ZIP Codes CSV must contain 'longitude' and 'latitude' columns.")
            return gpd.GeoDataFrame()
        df['zip'] = df['zip'].str.zfill(5)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error loading US ZIP Codes: {e}"); return gpd.GeoDataFrame()

def load_ad_target_zips(uploaded_file_object) -> pd.DataFrame:
    if uploaded_file_object is None: return pd.DataFrame(columns=['zip'])
    try:
        try: # Try simple, single column of zips first
            df_final = pd.read_csv(uploaded_file_object, dtype={'zip': str}, header=None, names=['zip'])
            if not df_final['zip'].astype(str).str.match(r'^\d{5}$').all(): # If not all zips, try with header
                raise ValueError("Not a simple list of ZIPs")
        except (Exception, ValueError): # Fallback to assuming 'zip' column with header
            uploaded_file_object.seek(0)
            df_temp = pd.read_csv(uploaded_file_object, dtype={'zip': str})
            df_temp.columns = df_temp.columns.str.strip().str.lower()
            if 'zip' in df_temp.columns:
                df_final = df_temp[['zip']]
            else: # Try to see if the first column is all zips (header might be numbers)
                uploaded_file_object.seek(0)
                df_header_as_data = pd.read_csv(uploaded_file_object, dtype=str, nrows=1) # Read first row as data
                df_full_first_col_check = pd.read_csv(uploaded_file_object, dtype=str, usecols=[0], header=None)

                if all(str(z).isdigit() and len(str(z)) == 5 for z in df_header_as_data.iloc[0]): # Header itself is zips
                     df_final = pd.DataFrame(df_header_as_data.iloc[0].values, columns=['zip'])
                elif df_full_first_col_check.iloc[:,0].astype(str).str.match(r'^\d{5}$').all(): # First column is zips
                     df_final = df_full_first_col_check.rename(columns={df_full_first_col_check.columns[0]:'zip'})
                else:
                    st.error("Ad Target ZIPs CSV: Could not find 'zip' column or parse as a simple list.")
                    return pd.DataFrame(columns=['zip'])

        df_final['zip'] = df_final['zip'].astype(str).str.zfill(5)
        return df_final[['zip']].drop_duplicates()
    except Exception as e:
        st.error(f"Error loading Ad Target ZIPs: {e}"); return pd.DataFrame(columns=['zip'])


def parse_input_zips(zip_code_text_input: str) -> pd.DataFrame:
    if not zip_code_text_input.strip():
        return pd.DataFrame(columns=['zip'])
    zips = [z.strip() for z in pd.Series(zip_code_text_input.splitlines()).str.split(r'[\s,]+').explode() if z.strip()]
    valid_zips = [z for z in zips if z.isdigit() and (len(z) == 5 or (len(z) < 5 and z.zfill(5).isdigit()))] # Allow short zips if they can be zfilled
    
    if not valid_zips:
        st.warning("No valid 5-digit ZIP codes found in the input text.")
        return pd.DataFrame(columns=['zip'])
    
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
    radius_m = miles * 1609.34
    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj_str = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    try: aeqd_proj = pyproj.CRS.from_proj4(aeqd_proj_str)
    except pyproj.exceptions.CRSError:
        st.warning(f"AEQD projection error for {lat},{lon}. Buffer may be inaccurate.")
        return Point(lon, lat).buffer(radius_m / 111000) # Rough degree buffer fallback
    project_fwd  = pyproj.Transformer.from_crs(wgs84, aeqd_proj,  always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84,  always_xy=True).transform
    center = Point(lon, lat); center_aeqd = transform(project_fwd, center)
    buffer_aeqd = center_aeqd.buffer(radius_m); buffer_wgs84= transform(project_back, buffer_aeqd)
    return buffer_wgs84

def create_geodesic_buffers(gdf_points, radii=(5,10)):
    if gdf_points.empty or 'geometry' not in gdf_points.columns: return gdf_points
    for r in radii:
        col_name = f"buffer_{r}"
        poly_list = []
        for _, row in gdf_points.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not isinstance(geom, Point): poly_list.append(None); continue
            lon, lat = geom.x, geom.y
            try: poly_list.append(geodesic_buffer(lon, lat, r))
            except Exception as e: st.warning(f"Buffer error for ZIP {row.get('zip', 'N/A')}: {e}"); poly_list.append(None)
        gdf_points[col_name] = gpd.GeoSeries(poly_list, crs="EPSG:4326")
    return gdf_points

###############################################################################
# MAIN PLOT FUNCTION
###############################################################################
def generate_map_plot(gdf_us, df_input_zips, df_ad_targets, gdf_k12_schools=None):
    if gdf_us.empty:
        st.error("US ZIP Code reference data is essential and missing."); fig, ax = plt.subplots(); ax.text(0.5,0.5,"US ZIP Data Missing", ha='center'); return fig
    if df_input_zips.empty:
        st.info("Please input primary ZIP codes to generate a map."); fig, ax = plt.subplots(); ax.text(0.5,0.5,"Input Primary ZIPs", ha='center'); return fig

    gdf_input_zips_geo = pd.merge(df_input_zips, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
    if gdf_input_zips_geo.empty:
        st.warning("None of the input ZIP codes found in US ZIP reference."); fig, ax = plt.subplots(); ax.text(0.5,0.5,"Input ZIPs not in US data", ha='center'); return fig
    gdf_input_zips_geo = gpd.GeoDataFrame(gdf_input_zips_geo, geometry='geometry', crs="EPSG:4326")
    gdf_input_zips_geo = create_geodesic_buffers(gdf_input_zips_geo, radii=(5,10)) # Buffers for input zips

    gdf_ad_targets_geo = gpd.GeoDataFrame(crs="EPSG:4326")
    if not df_ad_targets.empty:
        merged_ads = pd.merge(df_ad_targets, gdf_us[['zip','geometry']], on='zip', how='left').dropna(subset=['geometry'])
        if not merged_ads.empty: gdf_ad_targets_geo = gpd.GeoDataFrame(merged_ads, geometry='geometry', crs="EPSG:4326")

    # --- Filter K-12 schools within 10-mile radius of any input ZIP ---
    filtered_k12_schools = gpd.GeoDataFrame(crs="EPSG:4326")
    if gdf_k12_schools is not None and not gdf_k12_schools.empty and 'buffer_10' in gdf_input_zips_geo.columns:
        # Ensure buffer_10 is valid geometry
        valid_input_buffers_10 = gdf_input_zips_geo['buffer_10'].dropna()
        if not valid_input_buffers_10.empty:
            input_zips_coverage_union = unary_union(valid_input_buffers_10.tolist())
            if input_zips_coverage_union and not input_zips_coverage_union.is_empty:
                # Perform spatial join or intersection
                # Need to project both to the same projected CRS for accurate spatial operations if using .within or .intersects
                # For simplicity and assuming k12 schools are points, checking .within is often sufficient with WGS84 if areas are not too large
                # However, for robust solution, project:
                k12_proj = gdf_k12_schools.to_crs(epsg=3857)
                coverage_proj = gpd.GeoSeries([input_zips_coverage_union], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
                
                schools_within_indices = k12_proj.within(coverage_proj)
                filtered_k12_schools = gdf_k12_schools[schools_within_indices]


    # --- Projections ---
    gdf_input_3857    = gdf_input_zips_geo.to_crs(epsg=3857)
    gdf_ad_targets_3857 = gdf_ad_targets_geo.to_crs(epsg=3857) if not gdf_ad_targets_geo.empty else gpd.GeoDataFrame(crs="EPSG:3857")
    gdf_k12_filtered_3857 = filtered_k12_schools.to_crs(epsg=3857) if not filtered_k12_schools.empty else gpd.GeoDataFrame(crs="EPSG:3857")

    # Project buffers for plotting
    if 'buffer_5' in gdf_input_zips_geo.columns: gdf_input_3857['buffer_5_3857']  = gpd.GeoSeries(gdf_input_zips_geo['buffer_5'], crs="EPSG:4326").to_crs(epsg=3857)
    if 'buffer_10' in gdf_input_zips_geo.columns: gdf_input_3857['buffer_10_3857'] = gpd.GeoSeries(gdf_input_zips_geo['buffer_10'], crs="EPSG:4326").to_crs(epsg=3857)


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16,13))
    
    all_geoms_for_bounds = [gdf_input_3857, gdf_ad_targets_3857, gdf_k12_filtered_3857]
    valid_geoms_for_bounds = [g for g in all_geoms_for_bounds if g is not None and not g.empty and g.total_bounds is not None]

    if not valid_geoms_for_bounds: minx, miny, maxx, maxy = -13e6, 2.5e6, -7e6, 6.5e6 # Default US
    else:
        # Calculate combined total_bounds carefully, ensuring all are GeoDataFrames
        bounds_list = [gdf.total_bounds for gdf in valid_geoms_for_bounds if hasattr(gdf, 'total_bounds')]
        if not bounds_list: minx, miny, maxx, maxy = -13e6, 2.5e6, -7e6, 6.5e6
        else:
            minx = min(b[0] for b in bounds_list)
            miny = min(b[1] for b in bounds_list)
            maxx = max(b[2] for b in bounds_list)
            maxy = max(b[3] for b in bounds_list)

    w = maxx - minx if maxx > minx else 1e6; h = maxy - miny if maxy > miny else 1e6
    pad_x, pad_y = 0.15 * w, 0.15 * h # Increased padding for better view

    # 1. Input ZIPs (Primary points of interest) - Plot these first for clarity
    gdf_input_3857.plot(ax=ax, marker='*', color='crimson', markersize=200, label="Input ZIPs", zorder=5, edgecolor='black')

    # 2. Coverage polygons for Input ZIPs (semi-transparent to show overlaps)
    if 'buffer_5_3857' in gdf_input_3857.columns and gdf_input_3857['buffer_5_3857'].notna().any():
        gdf_input_3857[gdf_input_3857['buffer_5_3857'].notna()].plot(ax=ax, facecolor='orangered', edgecolor='orangered', alpha=0.2, linewidth=1.0, zorder=2, linestyle='--')
    if 'buffer_10_3857' in gdf_input_3857.columns and gdf_input_3857['buffer_10_3857'].notna().any():
        gdf_input_3857[gdf_input_3857['buffer_10_3857'].notna()].plot(ax=ax, facecolor='darkorange', edgecolor='darkorange', alpha=0.15, linewidth=1.5, zorder=1, linestyle=':')

    # 3. Ad Target ZIPs (Optional)
    if not gdf_ad_targets_3857.empty:
        gdf_ad_targets_3857.plot(ax=ax, marker='s', color='limegreen', markersize=70, label="Ad Target ZIPs", zorder=4, alpha=0.8, edgecolor='darkgreen')

    # 4. Filtered K-12 School Locations (within 10-mile radius of input ZIPs)
    if not gdf_k12_filtered_3857.empty:
        gdf_k12_filtered_3857.plot(ax=ax, marker='^', color='deepskyblue', markersize=50, label="K-12 Schools (in 10mi radius of Input ZIPs)", zorder=3, alpha=0.9, edgecolor='black')
    elif gdf_k12_schools is not None and not gdf_k12_schools.empty: # If K12 uploaded but none filtered
        st.info("No K-12 schools found within the 10-mile radius of the input ZIP codes.")


    try:
        ctx.add_basemap(ax, crs=gdf_input_3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto', attribution_size=5)
    except Exception as e: st.warning(f"Could not add basemap: {e}")

    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y); ax.axis('off')

    handles, labels = [], []
    handles.append(mlines.Line2D([], [], color='crimson', marker='*', linestyle='None', markersize=12, label='Input ZIPs', markeredgecolor='black')); labels.append(f'Input ZIPs ({len(gdf_input_3857)})')
    if 'buffer_5_3857' in gdf_input_3857: handles.append(mpatches.Patch(facecolor='orangered', alpha=0.3, edgecolor='orangered', linestyle='--', label='5mi Input Coverage')); labels.append('5-mile Input Coverage')
    if 'buffer_10_3857' in gdf_input_3857: handles.append(mpatches.Patch(facecolor='darkorange', alpha=0.2, edgecolor='darkorange', linestyle=':', label='10mi Input Coverage')); labels.append('10-mile Input Coverage')
    if not gdf_ad_targets_3857.empty: handles.append(mlines.Line2D([], [], color='limegreen', marker='s', linestyle='None', markersize=8, label='Ad Target ZIPs', markeredgecolor='darkgreen')); labels.append(f'Ad Target ZIPs ({len(gdf_ad_targets_3857)})')
    if not gdf_k12_filtered_3857.empty: handles.append(mlines.Line2D([], [], color='deepskyblue', marker='^', linestyle='None', markersize=8, label='K-12 Schools (Highlighted)', markeredgecolor='black')); labels.append(f'K-12 Schools ({len(gdf_k12_filtered_3857)} in 10mi radius)')

    if handles: ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small', title="Legend", title_fontsize="medium")
    ax.set_title("Input ZIP Code Coverage & K-12 School Proximity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust for legend
    return fig

###############################################################################
# STREAMLIT UI AND APP LOGIC
###############################################################################
st.sidebar.header("1. Input Primary ZIP Codes")
zip_code_input_text = st.sidebar.text_area("Paste your list of ZIP codes here (comma, space, or newline separated):", height=150, key="zip_input_area_v2",
help="Enter 5-digit ZIP codes. The app will map these and their 5/10-mile coverage.")

st.sidebar.header("2. Upload Supporting Data (CSV)")
uploaded_us_zips_file = st.sidebar.file_uploader("US ZIP Codes (Required: zip, latitude, longitude)", type="csv", key="us_zips_v2")
uploaded_ad_targets_file = st.sidebar.file_uploader("Ad Target ZIPs (Optional: zip)", type="csv", key="ad_targets_v2")
uploaded_k12_schools_file = st.sidebar.file_uploader("K-12 School Locations (Optional: name, latitude, longitude)", type="csv", key="k12_schools_v2")

if uploaded_us_zips_file and zip_code_input_text.strip():
    st.sidebar.success("Core inputs provided!")

    gdf_us_data = load_us_zip_codes(uploaded_us_zips_file)
    df_input_zips_data = parse_input_zips(zip_code_input_text)
    df_ad_targets_data = load_ad_target_zips(uploaded_ad_targets_file) if uploaded_ad_targets_file else pd.DataFrame(columns=['zip'])
    gdf_k12_schools_data = load_k12_schools(uploaded_k12_schools_file) if uploaded_k12_schools_file else gpd.GeoDataFrame()

    data_load_success = True
    if gdf_us_data.empty: st.error("US ZIP codes data is essential and failed to load."); data_load_success = False
    if df_input_zips_data.empty: st.warning("No valid primary ZIP codes parsed from input. Map cannot be generated without them."); data_load_success = False # Make this critical too
    
    if uploaded_ad_targets_file and df_ad_targets_data.empty: st.warning("Problem loading 'Ad Target ZIPs'. It will be excluded.")
    if uploaded_k12_schools_file and gdf_k12_schools_data.empty and uploaded_k12_schools_file is not None: # only warn if file was actually uploaded but failed
        st.warning("Problem loading 'K-12 School Locations'. It will be excluded.")


    if data_load_success:
        st.info("Data loaded. Generating map...")
        try:
            map_figure = generate_map_plot(gdf_us_data, df_input_zips_data, df_ad_targets_data, gdf_k12_schools_data)
            st.pyplot(map_figure)
            st.success("Map generated successfully!")
            fn = 'zip_analysis_map_v2.png'; img = io.BytesIO()
            map_figure.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button(label="Download Map as PNG", data=img, file_name=fn, mime="image/png")
        except Exception as e:
            st.error(f"An error occurred during map generation: {e}"); st.exception(e)
    elif not gdf_us_data.empty and df_input_zips_data.empty : # US Zips loaded but no input zips
         st.warning("Please provide valid primary ZIP codes in the text area to generate the map.")
    else:
        st.warning("Map could not be generated due to critical data loading issues.")
else:
    st.sidebar.info("Please paste primary ZIP codes and upload the 'US ZIP Codes' CSV to generate the map.")
    st.info("Awaiting inputs...")

st.markdown("---")
st.markdown("This tool visualizes your primary ZIP code areas, their coverage, and nearby K-12 schools.")
