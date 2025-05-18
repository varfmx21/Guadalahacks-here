import os
import pandas as pd
import numpy as np
import glob
import re
import random
import joblib
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from shapely.geometry import Point, LineString


# Global variables
tile_size = 512

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def lat_lon_to_tile(lat, lon, zoom):
    """Convert latitude and longitude to tile indices (x, y) at a given zoom level."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    n = 2.0 ** zoom
    x = int((lon_rad - (-math.pi)) / (2 * math.pi) * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return (x, y)

def tile_coords_to_lat_lon(x, y, zoom):
    """Convert tile indices to latitude and longitude."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1-2 * y/n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_bounds(x, y, zoom):
    """Get the bounds of a tile as (lat,lon) pairs for all four corners."""
    lat1, lon1 = tile_coords_to_lat_lon(x, y, zoom)
    lat2, lon2 = tile_coords_to_lat_lon(x+1, y, zoom)
    lat3, lon3 = tile_coords_to_lat_lon(x+1, y+1, zoom)
    lat4, lon4 = tile_coords_to_lat_lon(x, y+1, zoom)
    return (lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4)

def create_wkt_polygon(bounds):
    """Create a WKT polygon string from bounds corners."""
    (lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4) = bounds
    wkt = f"POLYGON(({lon1} {lat1}, {lon2} {lat2}, {lon3} {lat3}, {lon4} {lat4}, {lon1} {lat1}))"
    return wkt

def find_column(df, name_pattern):
    """Find column case-insensitively in DataFrame"""
    pattern = re.compile(name_pattern, re.IGNORECASE)
    matches = [col for col in df.columns if pattern.search(col)]
    return matches[0] if matches else None

def get_and_mark_satellite_tile(lat, lon, zoom, shape_points, reference_node, non_reference_node,
                               tile_format, api_key, output_dir="tiles", identifier=None,
                               pois=None):
    """
    Download a satellite tile and immediately draw road elements and POIs on it.
    Only saves the marked version.
    """
    x, y = lat_lon_to_tile(lat, lon, zoom)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the URL for the map tile API
    url = f'https://maps.hereapi.com/v3/base/mc/{zoom}/{x}/{y}/{tile_format}?style=satellite.day&size={tile_size}&apiKey={api_key}'

    # Define filename for the marked image
    if identifier:
        marked_filename = f'{output_dir}/tile_{identifier}_marked.{tile_format}'
    else:
        marked_filename = f'{output_dir}/tile_{lat:.5f}_{lon:.5f}_z{zoom}_marked.{tile_format}'

    # Make the request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the tile bounds
        bounds = get_tile_bounds(x, y, zoom)
        wkt_polygon = create_wkt_polygon(bounds)

        # Create image from response content
        img = Image.open(io.BytesIO(response.content))
        draw = ImageDraw.Draw(img)

        # Draw road elements on the image
        try:
            # Get tile dimensions
            (lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4) = bounds
            min_lon = min(lon1, lon2, lon3, lon4)
            max_lon = max(lon1, lon2, lon3, lon4)
            min_lat = min(lat1, lat2, lat3, lat4)
            max_lat = max(lat1, lat2, lat3, lat4)

            # Function to convert geo coordinates to pixel coordinates
            def geo_to_pixel(lon, lat):
                # Convert longitude/latitude to pixel coordinates on the image
                x = int((lon - min_lon) / (max_lon - min_lon) * img.width)
                y = int((max_lat - lat) / (max_lat - min_lat) * img.height)
                return x, y

            # Convert all shape points to pixel coordinates
            pixel_points = [geo_to_pixel(point[0], point[1]) for point in shape_points]

            # Draw the blue line connecting all shape points in sequence
            if len(pixel_points) > 1:
                for i in range(len(pixel_points) - 1):
                    draw.line((pixel_points[i][0], pixel_points[i][1],
                              pixel_points[i+1][0], pixel_points[i+1][1]),
                              fill='blue', width=3)

            # Draw all shape points
            for point in shape_points:
                lon, lat = point
                x, y = geo_to_pixel(lon, lat)
                # Draw small yellow dots for shape points
                draw.ellipse((x-2, y-2, x+2, y+2), fill='yellow')

            # Draw reference node (green)
            ref_lon, ref_lat = reference_node
            ref_x, ref_y = geo_to_pixel(ref_lon, ref_lat)
            draw.ellipse((ref_x-8, ref_y-8, ref_x+8, ref_y+8), outline='green', width=2)
            draw.text((ref_x+10, ref_y-10), "REF", fill='green')

            # Draw non-reference node (red)
            non_ref_lon, non_ref_lat = non_reference_node
            non_ref_x, non_ref_y = geo_to_pixel(non_ref_lon, non_ref_lat)
            draw.ellipse((non_ref_x-8, non_ref_y-8, non_ref_x+8, non_ref_y+8), outline='red', width=2)
            draw.text((non_ref_x+10, non_ref_y-10), "NON-REF", fill='red')

            # Draw POIs if provided
            if pois:
                print(f"Drawing {len(pois)} POIs on tile")
                for poi in pois:
                    try:
                        # Get POI properties with case-insensitive fallback
                        # Use PERCFRREF instead of PERCFFREF which was incorrect
                        perc_from_ref = float(poi.get('PERCFRREF', poi.get('PERCFFREF', 0.5)))
                        poi_side = poi.get('POI_ST_SD', 'R')

                        # Calculate POI position based on percentage and link geometry
                        poi_position = calculate_poi_position(
                            shape_points,
                            perc_from_ref,
                            reference_node,
                            non_reference_node,
                            poi_side
                        )

                        # Now extract the longitude and latitude
                        if isinstance(poi_position, list) and len(poi_position) == 2:
                            poi_lon, poi_lat = poi_position
                            poi_x, poi_y = geo_to_pixel(poi_lon, poi_lat)

                            # Draw POI with larger purple marker for better visibility
                            draw.ellipse((poi_x-8, poi_y-8, poi_x+8, poi_y+8), fill='purple')

                            # Add POI name/ID if available
                            poi_label = poi.get('POI_NAME', str(poi.get('POI_ID', 'POI')))
                            draw.text((poi_x+10, poi_y-10), poi_label, fill='purple')
                            print(f"  - Drew POI {poi_label} at position {poi_position}")
                        else:
                            print(f"  - Invalid POI position returned: {poi_position}")
                    except Exception as e:
                        print(f"Error drawing POI: {str(e)}")
                        import traceback
                        traceback.print_exc()

            # Create a legend
            legend_y = 20
            # Reference Node
            draw.ellipse((10, legend_y-8, 26, legend_y+8), outline='green', width=2)
            draw.text((30, legend_y-10), "Reference Node", fill='green')
            # Non-reference Node
            draw.ellipse((10, legend_y+20-8, 26, legend_y+20+8), outline='red', width=2)
            draw.text((30, legend_y+20-10), "Non-reference Node", fill='red')
            # Shape Points
            draw.ellipse((10+2, legend_y+40-2, 26-2, legend_y+40+2), fill='yellow')
            draw.text((30, legend_y+40-10), "Shape Points", fill='yellow')
            # Road Geometry
            draw.line((10, legend_y+60, 26, legend_y+60), fill='blue', width=3)
            draw.text((30, legend_y+60-10), "Road Geometry", fill='blue')
            # POI markers if we have any
            if pois:
                draw.ellipse((10, legend_y+80-6, 26, legend_y+80+6), fill='purple')
                draw.text((30, legend_y+80-10), "POI Locations", fill='purple')

            # Save the marked image
            img.save(marked_filename)
            print(f"Marked tile saved to {marked_filename}")

        except Exception as e:
            print(f"Error drawing road elements: {str(e)}")
            # Save the original image in case of error
            img.save(marked_filename)
            print(f"Saved original tile to {marked_filename} due to error")

        return wkt_polygon, marked_filename
    else:
        print(f'Failed to retrieve tile for {lat}, {lon}. Status code: {response.status_code}')
        return None, None

def calculate_poi_position(shape_points, perc_from_ref, reference_node, non_reference_node, side=''):
    """
    Calculate the geographic position of a POI based on:
    - The link shape points
    - Percentage from reference node (PERCFRREF)
    - Which side of the road it's on (POI_ST_SD)
    """
    # Ensure perc_from_ref is a valid float between 0 and 1
    try:
        perc_from_ref = float(perc_from_ref)
        perc_from_ref = max(0.0, min(1.0, perc_from_ref))
    except (ValueError, TypeError):
        print(f"Warning: Invalid percentage value: {perc_from_ref}, using default 0.5")
        perc_from_ref = 0.5

    # Handle empty shape points
    if not shape_points or len(shape_points) < 2:
        return reference_node

    # Check if reference node is at start using coordinate comparison
    ref_is_start = False
    ref_point = shape_points[0]
    if (abs(ref_point[0] - reference_node[0]) < 0.000001 and
        abs(ref_point[1] - reference_node[1]) < 0.000001):
        ref_is_start = True

    # If reference is not at start, reverse order
    working_points = list(shape_points)  # Make a copy to avoid modifying original
    if not ref_is_start:
        working_points.reverse()

    # Calculate total length of link
    total_length = 0
    segment_lengths = []

    for i in range(len(working_points) - 1):
        p1 = working_points[i]
        p2 = working_points[i + 1]
        # Calculate distance between adjacent points
        dx = p2[0] - p1[0]  # lon difference
        dy = p2[1] - p1[1]  # lat difference
        segment_length = math.sqrt(dx*dx + dy*dy)
        segment_lengths.append(segment_length)
        total_length += segment_length

    # Calculate target distance from reference node
    target_distance = perc_from_ref * total_length

    # Find the segment containing the target point
    current_distance = 0
    for i, segment_length in enumerate(segment_lengths):
        if current_distance + segment_length >= target_distance:
            # Found the segment containing our target
            segment_perc = (target_distance - current_distance) / segment_length
            p1 = working_points[i]
            p2 = working_points[i + 1]

            # Interpolate position on the segment
            poi_lon = p1[0] + segment_perc * (p2[0] - p1[0])
            poi_lat = p1[1] + segment_perc * (p2[1] - p1[1])

            # Apply side offset (perpendicular to segment direction)
            if side in ['R', 'L', 'r', 'l']:
                # Calculate perpendicular vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # Normalize the perpendicular vector
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Perpendicular vector (-dy, dx) for right side, (dy, -dx) for left side
                    # Adjust the offset magnitude based on the map scale
                    offset_magnitude = 0.00008  # Increased for better visibility

                    if side.upper() == 'R':
                        perp_x = -dy / length * offset_magnitude
                        perp_y = dx / length * offset_magnitude
                    else:  # 'L'
                        perp_x = dy / length * offset_magnitude
                        perp_y = -dx / length * offset_magnitude

                    poi_lon += perp_x
                    poi_lat += perp_y

            return [poi_lon, poi_lat]

        current_distance += segment_length

    # If we get here, something went wrong, return the last point
    if working_points:
        return working_points[-1]
    else:
        return reference_node

def determine_reference_node(shape_points):
    """
    Determine which end node is the reference node according to NAVSTREETS rules.

    Reference Node rules:
    1. The node with the lower latitude
    2. If latitudes are equal, the node with the lower longitude
    3. If lat and lon are equal, the node with the lower Z-Level (not implemented)

    Args:
        shape_points: List of [lon, lat] coordinates

    Returns:
        A tuple (reference_node, non_reference_node, is_reference_at_start)
    """
    # Get end nodes [lon, lat]
    start_node = shape_points[0]
    end_node = shape_points[-1]

    # Convert to [lat, lon] for easier comparison
    start_node_lat_lon = (start_node[1], start_node[0])
    end_node_lat_lon = (end_node[1], end_node[0])

    # Determine reference node based on latitude
    if start_node_lat_lon[0] < end_node_lat_lon[0]:  # Compare latitudes
        reference_node = start_node
        non_reference_node = end_node
        is_reference_at_start = True
    elif start_node_lat_lon[0] > end_node_lat_lon[0]:
        reference_node = end_node
        non_reference_node = start_node
        is_reference_at_start = False
    else:  # Equal latitudes, compare longitudes
        if start_node_lat_lon[1] < end_node_lat_lon[1]:  # Compare longitudes
            reference_node = start_node
            non_reference_node = end_node
            is_reference_at_start = True
        else:
            reference_node = end_node
            non_reference_node = start_node
            is_reference_at_start = False

    return reference_node, non_reference_node, is_reference_at_start

def get_satellite_tile(lat, lon, zoom, tile_format, api_key, output_dir="tiles", identifier=None, use_cache=True):
    """Get a satellite tile with caching and timeout support."""
    x, y = lat_lon_to_tile(lat, lon, zoom)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define filename for caching
    if identifier:
        filename = f'{output_dir}/tile_{identifier}.{tile_format}'
    else:
        filename = f'{output_dir}/tile_{lat:.5f}_{lon:.5f}_z{zoom}.{tile_format}'
    
    # Check cache first if enabled
    if use_cache and os.path.exists(filename):
        print(f"Using cached tile: {filename}")
        bounds = get_tile_bounds(x, y, zoom)
        wkt_polygon = create_wkt_polygon(bounds)
        return wkt_polygon, filename
    
    # Construct URL for map tile API
    url = f'https://maps.hereapi.com/v3/base/mc/{zoom}/{x}/{y}/{tile_format}?style=satellite.day&size={tile_size}&apiKey={api_key}'
    
    # Make the request with timeout
    try:
        response = requests.get(url, timeout=15)  # 15 second timeout
        
        # Check if the request was successful
        if response.status_code == 200:
            bounds = get_tile_bounds(x, y, zoom)
            wkt_polygon = create_wkt_polygon(bounds)
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Tile saved to {filename}")
            
            return wkt_polygon, filename
        else:
            print(f'Failed to retrieve tile for {lat}, {lon}. Status code: {response.status_code}')
            return None, None
    except requests.exceptions.Timeout:
        print(f"Request timed out for tile at {lat}, {lon}")
        return None, None
    except Exception as e:
        print(f"Error retrieving tile: {str(e)}")
        return None, None

def is_exception_type(exception, exception_types):
    """
    Check if the given exception is of any of the specified exception types.
    
    Args:
        exception: The exception to check
        exception_types: A single exception type or tuple of exception types to check against
        
    Returns:
        bool: True if the exception is of one of the specified types, False otherwise
    """
    return isinstance(exception, exception_types)

def generate_visualization(poi_row, roads_gdf, output_folder, poi_id=None):
    """Generate visualization with explicit API key"""
    try:
        # Define API key explicitly here to fix the scope issue
        api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get POI ID for filename
        if poi_id is None:
            if 'POI_ID' in poi_row:
                poi_id = poi_row['POI_ID']
            else:
                # Generate a random ID if none available
                poi_id = f"poi_{random.randint(10000, 99999)}"
        
        # Case-insensitive column lookups
        # 1. Find the LINK_ID column in the POI data
        link_id = None
        for col in poi_row.index:
            if col.upper() == 'LINK_ID':
                link_id = str(poi_row[col])
                break
        
        if link_id is None or pd.isna(link_id):
            print(f"Warning: No LINK_ID found for POI {poi_id}")
            
            # Plot all roads for context
            if isinstance(roads_gdf, gpd.GeoDataFrame):
                roads_gdf.plot(ax=ax, color='lightgray', linewidth=1)
            
            # Add text annotation for the POI
            plt.text(0.5, 0.5, f"POI: {poi_id}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            plt.text(0.5, 0.4, "No matching road segment found", 
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=10, color='darkred')
            
            # Save figure
            plt.title(f"POI {poi_id} - No Matching Road Found")
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(f"{output_folder}/poi_{poi_id}_visualization.png")
            plt.close(fig)
            return None
            
        # 2. Find the link_id column in the road data
        road_link_col = None
        for col in roads_gdf.columns:
            if col.upper() == 'LINK_ID':
                road_link_col = col
                break
        
        # If we don't find it, try lowercase version which is common in GeoJSON
        if road_link_col is None and 'link_id' in roads_gdf.columns:
            road_link_col = 'link_id'
            
        # Match road using road_link_col
        matching_road = None
        if road_link_col is not None:
            # Try both string and numeric comparison
            matching_road = roads_gdf[roads_gdf[road_link_col].astype(str) == link_id]
            
        # If we found a matching road, visualize it
        if matching_road is not None and len(matching_road) > 0:
            road = matching_road.iloc[0]
            
            # Plot all roads in light gray for context
            roads_gdf.plot(ax=ax, color='lightgray', linewidth=1)
            
            # Highlight the matching road in blue
            matching_road.plot(ax=ax, color='blue', linewidth=2)
            
            # Get road shape points
            shape_points = list(road.geometry.coords) if hasattr(road.geometry, 'coords') else []
            
            if shape_points and len(shape_points) >= 2:
                # Determine reference node
                reference_node, non_reference_node, _ = determine_reference_node(shape_points)
                
                # Try to calculate POI position
                poi_side = poi_row.get('POI_ST_SD', 'R')
                perc_from_ref = poi_row.get('PERCFRREF', 0.5)
                
                try:
                    poi_position = calculate_poi_position(
                        shape_points,
                        float(perc_from_ref) if pd.notna(perc_from_ref) else 0.5,
                        reference_node,
                        non_reference_node,
                        poi_side
                    )
                    
                    # Plot POI as a purple dot
                    if isinstance(poi_position, list) and len(poi_position) == 2:
                        poi_point = Point(poi_position[0], poi_position[1])
                        gpd.GeoSeries([poi_point]).plot(ax=ax, color='purple', markersize=50)
                except Exception as e:
                    print(f"Could not calculate POI position: {str(e)}")
        else:
            # No matching road found, just show roads for context
            roads_gdf.plot(ax=ax, color='lightgray', linewidth=1)
            plt.text(0.5, 0.5, f"No matching road for POI {poi_id}\nLINK_ID={link_id}", 
                    ha='center', va='center', transform=ax.transAxes)
                
        # Add POI metadata as text
        metadata = f"POI ID: {poi_id}\nLINK_ID: {link_id}\n"
        for key in ['FAC_TYPE', 'POI_NAME', 'POI_ST_SD', 'PERCFRREF', 'NAT_IMPORT']:
            if key in poi_row and pd.notna(poi_row[key]):
                metadata += f"{key}: {poi_row[key]}\n"
                
        plt.figtext(0.02, 0.02, metadata, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the visualization
        plt.title(f"POI {poi_id} Validation")
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(f"{output_folder}/poi_{poi_id}_visualization.png")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"Error generating visualization for POI {poi_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# POI VALIDATION SYSTEM
# ==========================================

class POIValidationSystem:
    def __init__(self, model_path=None):
        """Initialize the POI validation system."""
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Visualization settings
        self.api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
        self.zoom_level = 17
        self.tile_size = 512
        self.tile_format = 'png'
        self.output_dir = "poi_visualizations"


    def load_model(self, model_path):
        """Load a pre-trained model."""
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

    def extract_features(self, poi_row, roads_gdf):
        """Extract features for POI classification."""
        features = {}

        # Get POI core data
        link_id = poi_row.get('link_id')
        poi_side = poi_row.get('POI_ST_SD', 'R')
        perc_from_ref = poi_row.get('PERCFRREF', 0.5)
        facility_type = poi_row.get('FAC_TYPE')

        # 1. Road relationship features - IMPROVED MATCHING
        road_match = None

        if link_id is not None:
            # Try multiple matching strategies
            if 'link_id' in roads_gdf.columns:
                # Strategy 1: Direct match
                road_match = roads_gdf[roads_gdf['link_id'] == link_id]

                # Strategy 2: String comparison if needed
                if len(road_match) == 0:
                    str_link_id = str(link_id)
                    roads_gdf['link_id_STR'] = roads_gdf['link_id'].astype(str)
                    road_match = roads_gdf[roads_gdf['link_id_STR'] == str_link_id]

        # If no match and we want to avoid "delete" classification
        if road_match is None or len(road_match) == 0:
            features.update({
                'no_matching_road': True,
                'dist_to_road': 999,
                'is_multi_dig': False,
                'road_side_match': True,
                'road_side': 1 if poi_side == 'R' else 0,
                'perc_from_ref': float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
            })
        else:
            # We found a matching road
            road = road_match.iloc[0]
            shape_points = list(road.geometry.coords) if hasattr(road.geometry, 'coords') else []

            if shape_points and len(shape_points) >= 2:
                # Road with valid shape points
                reference_node, non_reference_node, _ = determine_reference_node(shape_points)

                # Calculate POI position using provided function
                poi_position = calculate_poi_position(
                    shape_points,
                    float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
                    reference_node,
                    non_reference_node,
                    poi_side
                )

                poi_lon, poi_lat = poi_position
                poi_geom = Point(poi_lon, poi_lat)

                # Road attributes
                is_multi_dig = road.get('MULTIDIGIT', 'N') == 'Y'
                calculated_side = self._determine_side_of_road(poi_geom, road.geometry)
                side_match = calculated_side == poi_side

                features.update({
                    'no_matching_road': False,
                    'dist_to_road': 0,  # On the road
                    'is_multi_dig': is_multi_dig,
                    'road_side_match': side_match,
                    'road_side': 1 if poi_side == 'R' else 0,
                    'perc_from_ref': float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
                })
            else:
                # Road exists but no valid shape points
                features.update({
                    'no_matching_road': False,
                    'dist_to_road': 999,
                    'is_multi_dig': False,
                    'road_side_match': True,
                    'road_side': 1 if poi_side == 'R' else 0,
                    'perc_from_ref': float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
                })

        # 2. POI type features
        # Check if facility_type is a valid numeric value, otherwise convert
        if isinstance(facility_type, str) and facility_type.isdigit():
            facility_type = int(facility_type)
        else:
            facility_type = -1

        exception_types = [3538, 4013, 1100, 4170, 4444, 4482, 4493, 4580, 4581, 5000]

        # Handle integer parsing for binary flags
        def parse_binary(value):
            if pd.isna(value):
                return 0
            if isinstance(value, (int, float, bool)):
                return 1 if value else 0
            if isinstance(value, str):
                return 1 if value.lower() in ('1', 'y', 'yes', 'true') else 0
            return 0

        features.update({
            'facility_type': facility_type,
            'is_exception_type': 1 if facility_type in exception_types else 0,
            'nat_import': parse_binary(poi_row.get('NAT_IMPORT')),
            'in_vicinity': parse_binary(poi_row.get('IN_VICIN')),
            'private': parse_binary(poi_row.get('PRIVATE')),
            'has_chain_id': pd.notna(poi_row.get('CHAIN_ID')),
            'has_phone': pd.notna(poi_row.get('PH_NUMBER')),
        })

        # 3. Special type indicators
        features.update({
            'is_airport': pd.notna(poi_row.get('AIRPT_TYPE')),
            'is_entrance': pd.notna(poi_row.get('ENTR_TYPE')),
            'is_restaurant': pd.notna(poi_row.get('REST_TYPE')),
            'is_24hour': parse_binary(poi_row.get('OPEN_24')),
        })

        return features

    def extract_enhanced_features(poi_row, roads_gdf):
        """Extrae características más discriminativas para cada clase"""
        features = {}
        
        # Características básicas existentes
        features.update(extract_poi_features(poi_row, roads_gdf))
        
        # Características adicionales para mejorar discriminación
        
        # 1. Detectar patrones en IDs para verificar existencia
        link_id = str(poi_row.get('link_id', ''))
        features['valid_id_pattern'] = 1 if re.match(r'^\d{5,8}$', link_id) else 0
        
        # 2. Verificar coherencia entre PERCFRREF y ubicación real
        # (Requiere análisis geométrico - simplificado aquí)
        
        # 3. Características específicas para MULTIDIGIT
        if features.get('is_multi_dig') == 1:
            # Comprobar si el tipo de instalación debería estar en una vía MULTIDIGIT
            features['appropriate_for_multidigit'] = 1 if features.get('is_exception_type') else 0
        
        # 4. Características específicas para excepciones legítimas
        features['exception_confidence'] = 1.0 if features.get('is_exception_type') and features.get('nat_import') else 0.0
        
        return features

    def _determine_side_of_road(self, point, line_geom):
        """Determine if POI is on left or right side of the road."""
        try:
            # Get the coordinates of the line
            if not hasattr(line_geom, 'coords'):
                return 'R'  # Default if geometry doesn't have coords

            coords = list(line_geom.coords)
            if len(coords) < 2:
                return 'R'  # Default if not enough points

            # Get the nearest point on the line
            nearest_point = line_geom.interpolate(line_geom.project(point))

            # Find the segment containing the nearest point
            min_dist = float('inf')
            segment_idx = 0

            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i+1]
                segment = LineString([p1, p2])
                dist = nearest_point.distance(segment)

                if dist < min_dist:
                    min_dist = dist
                    segment_idx = i

            # Get the segment direction
            p1 = coords[segment_idx]
            p2 = coords[segment_idx+1]

            # Calculate direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # Calculate vector from nearest point to POI
            nx = point.x - nearest_point.x
            ny = point.y - nearest_point.y

            # Cross product to determine side
            cross_product = dx * ny - dy * nx

            # If cross product > 0, point is on the right
            return 'R' if cross_product > 0 else 'L'

        except Exception as e:
            print(f"Error determining side of road: {str(e)}")
            return 'R'  # Default if calculation fails

    def train_model(self, labeled_data, roads_data):
        """
        Train the POI validation model.

        Args:
            labeled_data: DataFrame with POIs and their correct classifications
            roads_data: GeoDataFrame with road network
        """
        # Extract features for training
        features = []
        labels = []

        for idx, poi in labeled_data.iterrows():
            try:
                feature_dict = self.extract_features(poi, roads_data)
                features.append(feature_dict)
                labels.append(poi['classification'])  # Should be 0-3 for the four categories
            except Exception as e:
                print(f"Error extracting features for POI {poi.get('POI_ID', 'unknown')}: {str(e)}")

        # Convert to DataFrame for sklearn
        X = pd.DataFrame(features)
        y = np.array(labels)

        # Handle missing values
        X = X.fillna(-1)

        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        # Save the model
        joblib.dump(self.model, 'poi_validation_model.joblib')
        print("Model trained and saved to poi_validation_model.joblib")

        # Report feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = sorted(zip(X.columns, self.model.feature_importances_), key=lambda x: x[1], reverse=True)
            print("\nFeature Importance:")
            for feature, importance in feature_importance[:10]:  # Top 10
                print(f"  {feature}: {importance:.4f}")

    def classify_poi(self, poi, roads_data):
        """
        Classify a POI into one of the four categories.

        Args:
            poi: Dict or Series with POI data
            roads_data: GeoDataFrame with road network

        Returns:
            class_id: 0-3 classification result
            confidence: Model confidence
            action: Recommended action
        """
        # Extract features
        features = self.extract_features(poi, roads_data)

        # Direct rule-based classification for key cases
        # 1. No POI in reality - IMPROVED LOGIC
        if features['no_matching_road']:
            # Instead of automatic deletion, check for special facility types
            if features['is_exception_type'] or features['nat_import']:
                class_id = 3  # Mark as legitimate exception
                confidence = 0.7
            else:
                # If road ID looks valid but not found, mark for investigation
                link_id = poi.get('link_id')
                if link_id and isinstance(link_id, (int, str)) and str(link_id).isdigit():
                    class_id = 1  # Mark as wrong side for review
                    confidence = 0.6
                else:
                    class_id = 0  # Mark for deletion only if clearly invalid
                    confidence = 0.8
            rule_based = True

        # 2. Incorrect road matching (wrong side)
        elif not features['road_side_match'] and not features['is_multi_dig']:
            class_id = 1
            confidence = 0.85
            rule_based = True

        # 3. Incorrect MULTIDIGIT attribution
        elif features['is_multi_dig'] and not features['is_exception_type'] and not features['nat_import']:
            class_id = 2
            confidence = 0.8
            rule_based = True

        # 4. Legitimate exception
        elif features['is_exception_type'] or (features['is_multi_dig'] and features['nat_import']):
            class_id = 3
            confidence = 0.95
            rule_based = True

        # Use ML model for less clear cases
        else:
            rule_based = False
            # Convert features to DataFrame and handle missing values
            features_df = pd.DataFrame([features]).fillna(-1)

            if self.model:
                class_id = int(self.model.predict(features_df)[0])
                probabilities = self.model.predict_proba(features_df)[0]
                confidence = float(probabilities[class_id])
            else:
                # More balanced default classification if no model is available
                class_id = 1  # Changed from 0 to 1 (wrong side) as a safer default
                confidence = 0.5

        # Map class to action
        actions = [
            "Mark the feature for deletion",  # 0: POI doesn't exist
            "Update POI side of street",      # 1: Wrong side of road
            "Update the Link feature's MULTIDIGIT attribute to 'N'",  # 2: Incorrect MULTIDIGIT
            "Mark violation as 'Legitimate Exception'"  # 3: Valid exception
        ]

        return {
            'class_id': int(class_id),
            'confidence': float(confidence),
            'action': actions[class_id],
            'features': features,
            'rule_based': rule_based
        }

def train_advanced_model(training_data, roads_data):
    """Entrena un modelo más robusto con validación cruzada y ajuste de hiperparámetros"""
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Preparar datos
    features = []
    labels = []
    for _, poi in training_data.iterrows():
        feature_dict = extract_enhanced_features(poi, roads_data)
        features.append(feature_dict)
        labels.append(poi['classification'])
    
    X = pd.DataFrame(features).fillna(-1)
    y = np.array(labels)
    
    # Definir grid de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    # Buscar mejores hiperparámetros
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Entrenar modelo final con mejores parámetros
    best_model = grid_search.best_estimator_
    
    # Guardar el modelo
    joblib.dump(best_model, 'advanced_poi_model.joblib')
    print(f"Mejor modelo: {grid_search.best_params_}")
    
    return best_model

def train_specialized_classifier():
    """Crea un clasificador especializado con modelos específicos para cada clase"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Modelo principal (multiclase) para primera evaluación
    main_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    # Modelos binarios específicos para cada clase
    binary_models = {
        'nonexistent_detector': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight={0: 1, 1: 2},  # Penalizar más los falsos positivos (preservar POIs válidos)
            random_state=42
        ),
        'wrong_side_detector': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight='balanced',
            random_state=42
        ),
        'multidigit_detector': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        ),
        'exception_detector': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight={0: 1, 1: 3},  # Mayor peso a excepciones legítimas para evitar eliminaciones incorrectas
            random_state=42
        )
    }
    
    # Preprocesadores para cada modelo
    preprocessors = {
        'main': StandardScaler(),
        'nonexistent': StandardScaler(),
        'wrong_side': StandardScaler(),
        'multidigit': StandardScaler(),
        'exception': StandardScaler()
    }
    
    return main_model, binary_models, preprocessors

def visualize_poi(poi, nav_data, output_dir=None, zoom_level=17, tile_format="png", api_key=None):
    """Create a visual representation of a POI on satellite imagery"""
    # Define the API key explicitly if not provided
    if api_key is None:
        api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
    
    # Default output directory
    if output_dir is None:
        output_dir = "tiles"
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get POI ID and link ID
        if isinstance(poi, pd.Series):
            poi_id = poi.get('POI_ID', 'unknown')
            link_id = None
            # Try different case variations for LINK_ID
            for col in ['LINK_ID', 'link_id', 'LinkId']:
                if col in poi.index:
                    link_id = poi[col]
                    break
        else:
            poi_id = getattr(poi, 'POI_ID', 'unknown')
            link_id = getattr(poi, 'link_id', None)
        
        # Find matching road using case-insensitive matching
        road_match = None
        if link_id is not None and nav_data is not None:
            link_col = None
            for col in nav_data.columns:
                if col.lower() == 'link_id':
                    link_col = col
                    break
            
            if link_col:
                road_match = nav_data[nav_data[link_col].astype(str) == str(link_id)]
        
        # If road match found, visualize with road
        if road_match is not None and len(road_match) > 0:
            road = road_match.iloc[0]
            
            if hasattr(road, 'geometry') and road.geometry is not None:
                shape_points = list(road.geometry.coords)
                
                if len(shape_points) >= 2:
                    # Get reference nodes
                    reference_node, non_reference_node, _ = determine_reference_node(shape_points)
                    
                    # Calculate center
                    center_lon = sum(p[0] for p in shape_points) / len(shape_points)
                    center_lat = sum(p[1] for p in shape_points) / len(shape_points)
                    
                    # Get satellite tile with markers - pass API key explicitly!
                    wkt_bounds, filename = get_and_mark_satellite_tile(
                        center_lat, center_lon, zoom_level,
                        shape_points, reference_node, non_reference_node,
                        tile_format, api_key,  # Pass API key here
                        output_dir=output_dir,
                        identifier=f"poi_{poi_id}",
                        pois=[poi]
                    )
                    
                    return filename
        
        # Default to a simple tile in Mexico City center
        # Pass API key explicitly here too
        wkt_bounds, filename = get_satellite_tile(
            19.432, -99.133, zoom_level, tile_format, api_key,  # Pass API key here
            output_dir=output_dir,
            identifier=f"poi_{poi_id}_default"
        )
        return filename
        
    except Exception as e:
        print(f"Error in visualize_poi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def visualize_poi_classification(poi, roads, class_id, confidence, output_dir, poi_id=None):
    """Genera visualización especializada por clase"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Asegurar que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Obtener POI ID para el nombre del archivo
    if poi_id is None:
        poi_id = poi.get('POI_ID', f"unknown_{random.randint(1000, 9999)}")
    
    # Encontrar el link_id en POI
    link_id = None
    for col in poi.index:
        if col.upper() == 'LINK_ID':
            link_id = str(poi[col])
            break
    
    # Dibujar todas las calles como contexto
    if not roads.empty:
        roads.plot(ax=ax, color='lightgray', linewidth=1)
    
    # Buscar camino correspondiente al POI
    matching_road = None
    if link_id:
        # Buscar en diferentes variaciones de columnas de ID
        for id_col in ['link_id', 'LINK_ID', 'linkId']:
            if id_col in roads.columns:
                matching_road = roads[roads[id_col].astype(str) == link_id]
                if not matching_road.empty:
                    break
    
    # Dibujar camino coincidente o punto POI
    if matching_road is not None and not matching_road.empty:
        matching_road.plot(ax=ax, color='blue', linewidth=2)
        
        # Extraer coordenadas del POI
        poi_coords = get_poi_coordinates(poi)
        if poi_coords:
            ax.scatter(poi_coords[0], poi_coords[1], color='red', s=100, zorder=5)
    else:
        # Si no hay camino coincidente, solo mostrar el POI
        poi_coords = get_poi_coordinates(poi)
        if poi_coords:
            ax.scatter(poi_coords[0], poi_coords[1], color='red', s=100, zorder=5)
            ax.text(poi_coords[0], poi_coords[1], "POI (No matching road)", 
                   fontsize=10, ha='right')
    
    # Determinar color según clase
    class_colors = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}
    class_names = {
        0: 'Non-existent POI', 
        1: 'Wrong Side', 
        2: 'Incorrect MULTIDIGIT', 
        3: 'Legitimate Exception'
    }
    
    # Título y leyenda según clasificación
    plt.title(f"POI Classification: {class_names.get(class_id, 'Unknown')} (Conf: {confidence:.2f})")
    
    # Agregar metadatos del POI
    metadata = []
    for key, label in [
        ('POI_ID', 'ID'), ('POI_NAME', 'Name'), ('LINK_ID', 'Link ID'),
        ('POI_ST_SD', 'Side'), ('NAT_IMPORT', 'National Importance')
    ]:
        if key in poi and not pd.isna(poi.get(key)):
            metadata.append(f"{label}: {poi.get(key)}")
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    plt.figtext(0.02, 0.02, "\n".join(metadata), fontsize=8, bbox=bbox_props)
    
    # Leyenda con categoría
    legend_elements = [
        Patch(facecolor=class_colors.get(class_id, 'gray'), 
              label=f"Class {class_id}: {class_names.get(class_id, 'Unknown')}")
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Guardar figura
    filename = f"{output_dir}/poi_{poi_id}_class{class_id}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    
    return filename

# ==========================================
# MAIN PROCESSING FUNCTIONS
# ==========================================

def process_validation_dataset(poi_file, roads_file, model_file=None, output_dir="validation_results"):
    """Process a complete dataset for POI validation."""
    # Initialize system
    system = POIValidationSystem(model_path=model_file)

    # Add the visualize_poi method if it doesn't exist
    if not hasattr(system, 'visualize_poi'):
        system.visualize_poi = visualize_poi.__get__(system, POIValidationSystem)

    print(f"Loading POI data from {poi_file}")
    print(f"Loading road data from {roads_file}")

    # Load data
    try:
        pois = pd.read_csv(poi_file, header=0)
        print(f"Loaded {len(pois)} POIs")
        print(f"POI columns: {', '.join(pois.columns)}")
    except Exception as e:
        print(f"Error loading POI file: {str(e)}")
        return None

    try:
        roads = gpd.read_file(roads_file)
        print(f"Loaded {len(roads)} road segments")
        print(f"Road columns: {', '.join(roads.columns)}")
    except Exception as e:
        print(f"Error loading road file: {str(e)}")
        return None

    results = []

    # Process each POI
    print(f"\nProcessing {len(pois)} POIs...")
    for idx, poi in pois.iterrows():
        if idx % 10 == 0:
            print(f"Processing POI {idx+1}/{len(pois)}")

        try:
            # Classify the POI
            classification = system.classify_poi(poi, roads)

            # Generate visualization
            try:
                vis_image = system.visualize_poi(poi, roads, output_dir=f"{output_dir}/images")
            except Exception as e:
                print(f"Visualization error for POI {idx}: {str(e)}")
                vis_image = None

            # Store results
            result = {
                'poi_id': poi.get('POI_ID', idx),
                'link_id': poi.get('link_id'),
                'poi_name': poi.get('POI_NAME', ''),
                'class_id': classification.get('class_id', 1),  # Default to 1 if missing
                'confidence': classification.get('confidence', 0.5),
                'action': classification.get('action', 'Review needed'),
                'visualization': vis_image,
                'rule_based': classification.get('rule_based', False)
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing POI {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Add a placeholder result for this POI
            results.append({
                'poi_id': poi.get('POI_ID', idx),
                'link_id': poi.get('link_id'),
                'poi_name': poi.get('POI_NAME', ''),
                'class_id': 1,  # Default to wrong side (safer than delete)
                'confidence': 0.0,
                'action': 'Error during processing - review needed',
                'visualization': None,
                'rule_based': False
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/validation_results.csv", index=False)

    # Generate summary statistics with safeguards
    if 'class_id' in results_df.columns:
        summary = results_df['class_id'].value_counts().to_dict()
        print("\n=== POI Validation Summary ===")
        print(f"Total POIs processed: {len(results_df)}")
        print(f"POIs to delete: {summary.get(0, 0)}")
        print(f"POIs with wrong side of road: {summary.get(1, 0)}")
        print(f"POIs with incorrect MULTIDIGIT attribute: {summary.get(2, 0)}")
        print(f"Legitimate exceptions: {summary.get(3, 0)}")

        # Generate HTML report
        try:
            generate_html_report(results_df, output_dir)
            print(f"HTML report generated: {output_dir}/validation_report.html")
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
    else:
        print("Could not generate summary - 'class_id' column missing from results")

    return results_df

def generate_synthetic_training_data(roads_file, output_file, num_samples=1000):
    """
    Generate synthetic training data for model training that closely resembles
    the structure and patterns of real POI data.
    
    Args:
        roads_file: Path to the roads GeoDataFrame file
        output_file: Path to save the synthetic data
        num_samples: Total number of synthetic samples to generate
    """
    try:
        # Load roads
        roads = gpd.read_file(roads_file)
        
        # Load sample POI data to understand distributions
        sample_pois = pd.read_csv("data/POIs/POI_4815075.csv")
        
        # Facility Type mapping
        facility_types = {
            9587: "Agricultural Product Market",
            4581: "Airport",
            7996: "Amusement Park",
            9718: "Animal Park",
            3578: "ATM",
            5512: "Auto Dealership-Used Cars",
            7538: "Auto Service & Maintenance",
            8699: "Automobile Club",
            5511: "Auto Dealerships",
            6000: "Bank",
            9532: "Bar or Pub",
            9051: "Bicycle Parking",
            9059: "Bicycle Service",
            9050: "Bicycle Sharing Location",
            9058: "Bike Park",
            9057: "BMX Track",
            9995: "Bookstore",
            9999: "Border Crossing",
            7933: "Bowling Centre",
            4170: "Bus Station",
            5000: "Business Facility",
            9517: "Campground",
            9056: "Campsite",
            9714: "Cargo Centre",
            7985: "Casino",
            9591: "Cemetery",
            7832: "Cinema",
            9121: "City Hall",
            7994: "Civic/Community Centre",
            9537: "Clothing Store",
            9996: "Coffee Shop",
            4100: "Commuter Rail Station",
            9987: "Consumer Electronics Store",
            9535: "Convenience Store",
            7990: "Convention/Exhibition Centre",
            9994: "County Council",
            9211: "Court House",
            9722: "Delivery Entrance",
            9545: "Department Store",
            9723: "Dock",
            9993: "Embassy",
            9598: "EV Charging Station",
            4482: "Ferry Terminal",
            9527: "Fire Department",
            7992: "Golf Course",
            9573: "Golf Practice Range",
            9525: "Government Office",
            5400: "Grocery Store",
            9998: "Hamlet",
            8200: "Higher Education",
            9592: "Highway Exit",
            5999: "Historical Monument",
            9986: "Home Improvement & Hardware Store",
            9560: "Home Specialty Store",
            8060: "Hospital",
            7011: "Hotel",
            7998: "Ice Skating Rink",
            9991: "Industrial Zone",
            8231: "Library",
            9724: "Loading Zone",
            9594: "Lottery Booth",
            4493: "Marina",
            9583: "Medical Service",
            9725: "Meeting Point",
            9715: "Military Base",
            5571: "Motorcycle Dealership",
            8410: "Museum",
            9730: "Named Intersection",
            4444: "Named Place",
            9709: "Neighbourhood",
            5813: "Nightlife",
            9988: "Office Supply & Service Store",
            7013: "Other Accommodation",
            9053: "Outdoor Service",
            7522: "Park & Ride",
            7947: "Park/Recreation Area",
            7521: "Parking Garage/House",
            7520: "Parking Lot",
            7929: "Performing Arts",
            5540: "Petrol/Gasoline Station",
            9565: "Pharmacy",
            9992: "Place of Worship",
            9221: "Police Station",
            9530: "Post Office",
            9589: "Public Restroom",
            4580: "Public Sports Airport",
            9054: "Ranger Station",
            7510: "Rental Car Agency",
            9595: "Repair Services",
            9590: "Residential Area/Building",
            7897: "Rest Area",
            5800: "Restaurant",
            9055: "Running Track",
            8211: "School",
            6512: "Shopping",
            7014: "Ski Lift",
            7012: "Ski Resort",
            9567: "Specialty Store",
            9568: "Sporting Goods Store",
            7997: "Sports Centre",
            7940: "Sports Complex",
            9989: "Taxi Stand",
            9597: "Tea House",
            9717: "Tollbooth",
            7999: "Tourist Attraction",
            7389: "Tourist Information",
            9052: "Trailhead",
            4013: "Train Station",
            9596: "Training Centre/Institute",
            9593: "Transportation Service",
            9719: "Truck Dealership",
            9720: "Truck Parking",
            9522: "Truck Stop/Plaza",
            9710: "Weigh Station",
            2084: "Winery",
            3538: "Hotel", # Adding some missing exception types with descriptions
            1100: "Restaurant",
            4444: "Named Place",
            4493: "Marina"
        }
        
        # List of facility type codes
        facility_codes = list(facility_types.keys())
        
        # Exception facility types (those that can be on MULTIDIGIT roads)
        exception_fac_types = [3538, 4013, 1100, 4170, 4444, 4482, 4493, 4580, 4581, 5000]
        
        # Analyze facility types in sample data
        sample_fac_types = sample_pois['FAC_TYPE'].dropna().astype(int).tolist()
        
        # Create empty list for synthetic POIs
        synthetic_data = []
        
        # Helper function to generate realistic POI names
        def generate_poi_name(fac_type):
            # Use the facility type description as a base
            if fac_type in facility_types:
                base_name = facility_types[fac_type]
                
                # Common name patterns
                prefixes = ["The", "City", "Town", "Local", "Central", "Downtown", "Express", "Royal", "Golden", "Premier"]
                suffixes = ["Center", "Express", "Plus", "Services", "Direct", "Prime", "Pro"]
                
                name_pattern = random.randint(1, 5)
                
                if name_pattern == 1:
                    # Simple name: Base name
                    return base_name
                elif name_pattern == 2:
                    # Prefix + Base name
                    return f"{random.choice(prefixes)} {base_name}"
                elif name_pattern == 3:
                    # Base name + Suffix
                    return f"{base_name} {random.choice(suffixes)}"
                elif name_pattern == 4:
                    # Person's name + Base name
                    first_names = ["Joe's", "Mary's", "John's", "David's", "Linda's", "Susan's", "Michael's", "Robert's"]
                    return f"{random.choice(first_names)} {base_name}"
                else:
                    # Location + Base name
                    locations = ["Riverside", "Highland", "Lakeside", "Mountain", "Valley", "Plaza", "Metro", "Urban"]
                    return f"{random.choice(locations)} {base_name}"
            else:
                # Default for unknown facility types
                return f"Business {fac_type}"
        
        # Helper function to determine reference node
        def determine_reference_node(coords):
            # Simple implementation - first point is ref, last is non-ref
            ref_node = coords[0]
            non_ref_node = coords[-1]
            return ref_node, non_ref_node, "FROM_NODE"  # Default to FROM_NODE as ref type
        
        # 1. Generate valid POIs (class 3 - legitimate exceptions)
        num_legitimate = num_samples // 4
        valid_roads = roads[roads['MULTIDIGIT'] == 'Y'].sample(min(num_legitimate, len(roads[roads['MULTIDIGIT'] == 'Y'])), replace=True)
        
        for idx, road in valid_roads.iterrows():
            if hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                if len(coords) >= 2:
                    # Calculate random position along road
                    perc_from_ref = round(random.random(), 2)
                    ref_node, non_ref_node, _ = determine_reference_node(coords)

                    # Use actual exception facility types
                    facility_type = random.choice(exception_fac_types)
                    
                    # Generate POI name based on facility type
                    poi_name = generate_poi_name(facility_type)

                    poi = {
                        'POI_ID': f"SYN{str(idx).zfill(7)}",
                        'link_id': road.get('link_id', f"L{str(idx).zfill(6)}"),
                        'SEQ_NUM': 1,
                        'FAC_TYPE': facility_type,
                        'POI_NAME': poi_name,
                        'POI_LANGCD': 'SPA',
                        'POI_NMTYPE': 'B',
                        'POI_ST_SD': random.choice(['R', 'L']),
                        'PERCFRREF': perc_from_ref,
                        'NAT_IMPORT': random.choices(['Y', 'N'], [0.2, 0.8])[0],  # 20% nationally important
                        'PRIVATE': random.choices(['Y', 'N'], [0.3, 0.7])[0],
                        'IN_VICIN': random.choices(['Y', 'N'], [0.1, 0.9])[0],
                        'classification': 3  # Legitimate exception
                    }
                    synthetic_data.append(poi)

        # 2. Generate POIs with incorrect MULTIDIGIT (class 2)
        num_incorrect_multidig = num_samples // 4
        incorrect_roads = roads[roads['MULTIDIGIT'] == 'Y'].sample(min(num_incorrect_multidig, len(roads[roads['MULTIDIGIT'] == 'Y'])), replace=True)
        
        non_exception_fac_types = [ft for ft in facility_codes if ft not in exception_fac_types]
        
        for idx, road in incorrect_roads.iterrows():
            if hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                if len(coords) >= 2:
                    # Use non-exception facility type
                    facility_type = random.choice(non_exception_fac_types)
                    
                    # Generate POI name based on facility type
                    poi_name = generate_poi_name(facility_type)
                    
                    poi = {
                        'POI_ID': f"SYN{str(idx+10000).zfill(7)}",
                        'link_id': road.get('link_id', f"L{str(idx).zfill(6)}"),
                        'SEQ_NUM': 1,
                        'FAC_TYPE': facility_type,
                        'POI_NAME': poi_name,
                        'POI_LANGCD': 'SPA',
                        'POI_NMTYPE': 'B',
                        'POI_ST_SD': random.choice(['R', 'L']),
                        'PERCFRREF': round(random.random(), 2),
                        'NAT_IMPORT': random.choices(['Y', 'N'], [0.1, 0.9])[0],  # 10% nationally important
                        'PRIVATE': random.choices(['Y', 'N'], [0.3, 0.7])[0],
                        'IN_VICIN': random.choices(['Y', 'N'], [0.1, 0.9])[0],
                        'classification': 2  # Incorrect MULTIDIGIT
                    }
                    synthetic_data.append(poi)

        # 3. Generate wrong side of road POIs (class 1)
        num_wrong_side = num_samples // 4
        side_roads = roads.sample(min(num_wrong_side, len(roads)), replace=True)
        
        for idx, road in side_roads.iterrows():
            if hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                if len(coords) >= 2:
                    # Use random facility type from comprehensive list
                    facility_type = random.choice(facility_codes)
                    
                    # Generate POI name based on facility type
                    poi_name = generate_poi_name(facility_type)
                    
                    poi = {
                        'POI_ID': f"SYN{str(idx+20000).zfill(7)}",
                        'link_id': road.get('link_id', f"L{str(idx).zfill(6)}"),
                        'SEQ_NUM': 1,
                        'FAC_TYPE': facility_type,
                        'POI_NAME': poi_name,
                        'POI_LANGCD': 'SPA',
                        'POI_NMTYPE': 'B',
                        'POI_ST_SD': random.choice(['R', 'L']),  # Will be treated as wrong in training
                        'PERCFRREF': round(random.random(), 2),
                        'NAT_IMPORT': random.choices(['Y', 'N'], [0.1, 0.9])[0],
                        'PRIVATE': random.choices(['Y', 'N'], [0.3, 0.7])[0],
                        'IN_VICIN': random.choices(['Y', 'N'], [0.1, 0.9])[0],
                        'classification': 1  # Wrong side
                    }
                    synthetic_data.append(poi)

        # 4. Generate non-existent POIs (class 0)
        num_nonexistent = num_samples // 4
        for i in range(num_nonexistent):
            # Use random facility type from comprehensive list
            facility_type = random.choice(facility_codes)
            
            # Generate POI name based on facility type
            poi_name = generate_poi_name(facility_type)
            
            poi = {
                'POI_ID': f"SYN{str(i+30000).zfill(7)}",
                'link_id': f"INVALID_{str(i).zfill(5)}",
                'SEQ_NUM': 1,
                'FAC_TYPE': facility_type,
                'POI_NAME': poi_name,
                'POI_LANGCD': 'SPA',
                'POI_NMTYPE': 'B',
                'POI_ST_SD': random.choice(['R', 'L']),
                'PERCFRREF': round(random.random(), 2),
                'NAT_IMPORT': random.choices(['Y', 'N'], [0.05, 0.95])[0],  # 5% nationally important
                'PRIVATE': random.choices(['Y', 'N'], [0.3, 0.7])[0],
                'IN_VICIN': random.choices(['Y', 'N'], [0.1, 0.9])[0],
                'classification': 0  # Does not exist
            }
            synthetic_data.append(poi)

        # Save to CSV
        df = pd.DataFrame(synthetic_data)
        
        # Add any missing columns from the original dataset
        for col in sample_pois.columns:
            if col not in df.columns and col != 'classification':
                if col in ['LATITUDE', 'LONGITUDE']:
                    df[col] = np.random.uniform(-90 if col == 'LATITUDE' else -180, 
                                               90 if col == 'LATITUDE' else 180, 
                                               size=len(df))
                else:
                    # Use empty values for other columns
                    df[col] = ""
        
        df.to_csv(output_file, index=True)  # Include index to match sample CSV format
        print(f"Generated {len(df)} synthetic training samples and saved to {output_file}")
        print(f"Class distribution: Class 0: {len(df[df['classification']==0])}, " 
              f"Class 1: {len(df[df['classification']==1])}, "
              f"Class 2: {len(df[df['classification']==2])}, "
              f"Class 3: {len(df[df['classification']==3])}")

        return df

    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_enhanced_training_data(road_files, output_file, samples_per_class=500):
    """Genera datos sintéticos mejorados con características más realistas"""
    import random
    from shapely.geometry import Point, LineString
    
    # Cargar datos de calles para referencia
    all_roads = []
    for file in road_files[:3]:  # Limitar a 3 archivos para no sobrecargar la memoria
        try:
            roads = gpd.read_file(file)
            all_roads.append(roads)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_roads:
        raise ValueError("No se pudieron cargar datos de calles")
    
    roads = pd.concat(all_roads, ignore_index=True)
    
    # Lista para almacenar datos sintéticos
    data = []
    
    # ----- CLASE 0: POIs inexistentes -----
    for i in range(samples_per_class):
        # Variedad de problemas que causan POIs inexistentes
        subtype = random.choice(['invalid_id', 'no_match', 'far_away'])
        
        if subtype == 'invalid_id':
            # ID de enlace completamente inválido
            link_id = f"INV{random.randint(10000, 99999)}"
        elif subtype == 'no_match':
            # ID de enlace con formato válido pero que no existe
            link_id = str(random.randint(10000000, 99999999))
        else:  # 'far_away'
            # POI muy alejado de cualquier calle
            if len(roads) > 0:
                random_road = roads.sample(1).iloc[0]
                if hasattr(random_road.geometry, 'coords'):
                    coords = list(random_road.geometry.coords)
                    if coords:
                        base_x, base_y = coords[0]
                        # Desplazar el POI lejos de la calle
                        offset = 0.01  # ~1km en grados
                        link_id = str(random_road.get('link_id', random.randint(10000, 99999)))
                    else:
                        link_id = str(random.randint(10000, 99999))
                else:
                    link_id = str(random.randint(10000, 99999))
            else:
                link_id = str(random.randint(10000, 99999))
        
        # Crear POI sintético inexistente
        data.append({
            'POI_ID': f"SYN_NON_{i}",
            'LINK_ID': link_id,
            'FAC_TYPE': random.choice([1000, 2000, 3000, 4000, 5000]),
            'POI_NAME': f"Nonexistent Place {i}",
            'POI_ST_SD': random.choice(['R', 'L']),
            'PERCFRREF': random.random(),
            'NAT_IMPORT': random.choice([0, 0, 0, 1]),  # Mayoría no importantes
            'class_id': 0
        })
    
    # ----- CLASE 1: POIs en lado incorrecto -----
    # Códigos similares para generar datos de entrenamiento para Clase 1, 2 y 3
    # ...
    
    # Convertir a DataFrame y guardar
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(data)} training samples ({samples_per_class} per class)")
    return df

def create_improved_training_data(roads_file, output_file, samples_per_class=500):
    """Genera datos sintéticos mejorados con características más discriminativas"""
    roads = gpd.read_file(roads_file)
    
    # Para cada clase, crea ejemplos que reflejen las características específicas
    data = []
    
    # Clase 0: POIs que no existen (link_ids inválidos)
    for i in range(samples_per_class):
        data.append({
            'POI_ID': f"nonexistent_{i}",
            'link_id': f"invalid_{i}",  # IDs claramente inválidos
            'FAC_TYPE': random.randint(1000, 9000),
            'POI_NAME': f"Nonexistent POI {i}",
            'POI_ST_SD': random.choice(['R', 'L']),
            'PERCFRREF': random.random(),
            'NAT_IMPORT': 0,  # Mayormente no son importantes nacionalmente
            'IN_VICIN': 'N',
            'classification': 0
        })
    
    # Clase 1: POIs en lado incorrecto
    multi_roads = roads.sample(min(samples_per_class, len(roads)))
    for i, road in enumerate(multi_roads.iterrows()):
        if i >= samples_per_class:
            break
        _, road_data = road
        data.append({
            'POI_ID': f"wrong_side_{i}",
            'link_id': road_data.get('link_id', f"link_{i}"),
            'FAC_TYPE': random.randint(1000, 9000),
            'POI_NAME': f"Wrong Side POI {i}",
            'POI_ST_SD': 'R',  # Forzar lado incorrecto
            'PERCFRREF': random.random(),
            'NAT_IMPORT': random.choice([0, 1]),
            'classification': 1
        })
    
    # Añadir clases 2 y 3 con características distintivas...
    
    # Guardar a CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return df

def generate_balanced_poi_dataset(roads_file, output_file, num_per_class=250):
    """Generate a balanced dataset with equal POIs per class"""
    print(f"Generating balanced POI dataset with {num_per_class} POIs per class...")
    
    # Load roads
    roads = gpd.read_file(roads_file)
    print(f"Loaded {len(roads)} road segments")
    
    # Create empty list for POIs
    pois = []
    
    # 1. Generate class 3: legitimate exceptions
    print(f"Generating {num_per_class} legitimate exceptions (class 3)...")
    multi_roads = roads[roads['MULTIDIGIT'] == 'Y']
    
    count = 0
    road_pool = list(multi_roads.iterrows())
    if len(road_pool) < num_per_class:
        # Duplicate if not enough roads
        while len(road_pool) < num_per_class:
            road_pool.extend(multi_roads.iterrows())
    
    # Sample randomly from the road pool
    for _, road in random.sample(road_pool, num_per_class):
        if hasattr(road.geometry, 'coords'):
            coords = list(road.geometry.coords)
            if len(coords) >= 2:
                # Generate a legitimate exception
                poi = {
                    'POI_ID': f"poi_c3_{count}",
                    'link_id': road.get('link_id', f"link_{count}"),
                    'FAC_TYPE': random.choice([3538, 4013, 1100, 4170, 4444, 4482, 4493, 4580, 4581, 5000]),
                    'POI_NAME': f"Legitimate Exception POI {count}",
                    'POI_ST_SD': random.choice(['R', 'L']),
                    'PERCFRREF': random.random(),
                    'NAT_IMPORT': 1,  # Use integer instead of 'Y'
                    'classification': 3
                }
                pois.append(poi)
                count += 1
    
    print(f"Generated {count} class 3 POIs")
    
    # 2. Generate class 2: incorrect MULTIDIGIT
    print(f"Generating {num_per_class} incorrect MULTIDIGIT POIs (class 2)...")
    count = 0
    
    # Use random roads
    road_pool = []
    for _, road in roads.iterrows():
        if hasattr(road.geometry, 'coords') and len(list(road.geometry.coords)) >= 2:
            road_pool.append((_, road))
            if len(road_pool) >= num_per_class * 2:
                break
    
    random.shuffle(road_pool)
    
    for _, road in road_pool[:num_per_class]:
        # Generate a POI with incorrect MULTIDIGIT attribution
        poi = {
            'POI_ID': f"poi_c2_{count}",
            'link_id': road.get('link_id', f"link_c2_{count}"),
            'FAC_TYPE': random.randint(1000, 3000),
            'POI_NAME': f"Incorrect MULTIDIGIT POI {count}",
            'POI_ST_SD': random.choice(['R', 'L']),
            'PERCFRREF': random.random(),
            'NAT_IMPORT': 0,  # Use integer instead of 'N'
            'classification': 2
        }
        pois.append(poi)
        count += 1
    
    print(f"Generated {count} class 2 POIs")
    
    # 3. Generate class 1: wrong side of road
    print(f"Generating {num_per_class} wrong side POIs (class 1)...")
    count = 0
    
    # Reuse road pool
    random.shuffle(road_pool)
    
    for _, road in road_pool[:num_per_class]:
        # Generate a POI with wrong side attribution
        poi = {
            'POI_ID': f"poi_c1_{count}",
            'link_id': road.get('link_id', f"link_c1_{count}"),
            'FAC_TYPE': random.randint(1000, 7000),
            'POI_NAME': f"Wrong Side POI {count}",
            'POI_ST_SD': 'R',  # Force 'wrong' side
            'PERCFRREF': random.random(),
            'NAT_IMPORT': 0,  # Use integer instead of 'N'
            'classification': 1
        }
        pois.append(poi)
        count += 1
    
    print(f"Generated {count} class 1 POIs")
    
    # 4. Generate class 0: non-existent POIs
    print(f"Generating {num_per_class} non-existent POIs (class 0)...")
    
    for i in range(num_per_class):
        # Generate a POI with invalid link ID
        poi = {
            'POI_ID': f"poi_c0_{i}",
            'link_id': f"definitely_invalid_{i}",
            'FAC_TYPE': random.randint(1000, 7000),
            'POI_NAME': f"Non-existent POI {i}",
            'POI_ST_SD': random.choice(['R', 'L']),
            'PERCFRREF': random.random(),
            'NAT_IMPORT': 0,  # Use integer instead of 'N'
            'classification': 0
        }
        pois.append(poi)
    
    print(f"Generated {num_per_class} class 0 POIs")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(pois)
    df.to_csv(output_file, index=False)
    print(f"Generated total of {len(df)} POIs with balanced classes")
    print(f"Saved to {output_file}")
    
    return df

def generate_html_report(results_df, output_dir):
    """Generate an HTML report with the validation results."""
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>POI Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #2c3e50; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .high-confidence { color: green; }
            .medium-confidence { color: orange; }
            .low-confidence { color: red; }
            .class-0 { background-color: #ffcccc; } /* Red for deletion */
            .class-1 { background-color: #ffffcc; } /* Yellow for wrong side */
            .class-2 { background-color: #ccffff; } /* Blue for incorrect MULTIDIGIT */
            .class-3 { background-color: #ccffcc; } /* Green for legitimate exception */
        </style>
    </head>
    <body>
        <h1>POI Validation Report</h1>
    """

    # Add summary statistics
    class_id_col = 'predicted_label' if 'predicted_label' in results_df.columns else 'class_id'
    summary = results_df[class_id_col].value_counts().to_dict()
    html_content += """
        <div class="summary">
            <h2>Summary</h2>
            <p>Total POIs processed: {}</p>
            <p>POIs to delete: {}</p>
            <p>POIs with wrong side of road: {}</p>
            <p>POIs with incorrect MULTIDIGIT attribute: {}</p>
            <p>Legitimate exceptions: {}</p>
        </div>
    """.format(
        len(results_df),
        summary.get(0, 0),
        summary.get(1, 0),
        summary.get(2, 0),
        summary.get(3, 0)
    )

    # Add results table
    html_content += """
        <h2>Validation Results</h2>
        <table>
            <tr>
                <th>POI ID</th>
                <th>Name</th>
                <th>Link ID</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Action</th>
            </tr>
    """

    # Class names for display
    class_names = [
        "Non-existent POI",
        "Wrong Side of Road",
        "Incorrect MULTIDIGIT",
        "Legitimate Exception"
    ]

    # Add a row for each POI
    for _, row in results_df.iterrows():
        # Set confidence color class
        confidence = row.get('confidence', 0)
        if confidence > 0.8:
            confidence_class = "high-confidence"
        elif confidence > 0.6:
            confidence_class = "medium-confidence"
        else:
            confidence_class = "low-confidence"
            
        # Get class ID
        class_id = row[class_id_col]
        class_style = f"class-{class_id}" if 0 <= class_id <= 3 else ""

        # Add row
        html_content += f"""
            <tr class="{class_style}">
                <td>{row.get('poi_id', '')}</td>
                <td>{row.get('poi_name', '')}</td>
                <td>{row.get('link_id', '')}</td>
                <td>{class_names[class_id] if 0 <= class_id < len(class_names) else f"Unknown ({class_id})"}</td>
                <td class="{confidence_class}">{confidence:.2f}</td>
                <td>{row.get('action', '')}</td>
            </tr>
        """

    # Close table and HTML
    html_content += """
        </table>
    </body>
    </html>
    """

    # Write HTML to file
    report_file = f"{output_dir}/validation_report.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"HTML report generated: {report_file}")

def generate_html_report_with_images(results_df, output_dir):
    """Generate an HTML report with the validation results and visualizations."""
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>POI Validation Report with Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #2c3e50; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .high-confidence { color: green; }
            .medium-confidence { color: orange; }
            .low-confidence { color: red; }
            .visualization { max-width: 300px; max-height: 300px; border: 1px solid #ddd; }
            .class-0 { background-color: #ffcccc; } /* Red for deletion */
            .class-1 { background-color: #ffffcc; } /* Yellow for wrong side */
            .class-2 { background-color: #ccffff; } /* Blue for incorrect MULTIDIGIT */
            .class-3 { background-color: #ccffcc; } /* Green for legitimate exception */
        </style>
    </head>
    <body>
        <h1>POI Validation Report with Visualizations</h1>
    """

    # Add summary statistics
    summary = results_df['predicted_label'].value_counts().to_dict()
    html_content += """
        <div class="summary">
            <h2>Summary</h2>
            <p>Total POIs processed: {}</p>
            <p>POIs to delete: {}</p>
            <p>POIs with wrong side of road: {}</p>
            <p>POIs with incorrect MULTIDIGIT attribute: {}</p>
            <p>Legitimate exceptions: {}</p>
        </div>
    """.format(
        len(results_df),
        summary.get(0, 0),
        summary.get(1, 0),
        summary.get(2, 0),
        summary.get(3, 0)
    )

    # Add results table
    html_content += """
        <h2>Validation Results</h2>
        <table>
            <tr>
                <th>POI ID</th>
                <th>Name</th>
                <th>Link ID</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Action</th>
                <th>Visualization</th>
            </tr>
    """

    # Class names for display
    class_names = [
        "Non-existent POI",
        "Wrong Side of Road",
        "Incorrect MULTIDIGIT",
        "Legitimate Exception"
    ]

    # Add a row for each POI
    for _, row in results_df.iterrows():
        # Set confidence color class
        confidence = row['confidence']
        if confidence > 0.8:
            confidence_class = "high-confidence"
        elif confidence > 0.6:
            confidence_class = "medium-confidence"
        else:
            confidence_class = "low-confidence"
            
        # Get class ID
        class_id = row['predicted_label']
        class_style = f"class-{class_id}" if 0 <= class_id <= 3 else ""

        # Get visualization path
        vis_path = row.get('visualization_path', None)
        vis_html = ""
        if pd.notna(vis_path) and os.path.exists(vis_path):
            # Convert to relative path
            rel_path = os.path.relpath(vis_path, output_dir)
            vis_html = f'<img src="{rel_path}" class="visualization" alt="POI Visualization">'
        else:
            vis_html = "No visualization available"

        # Add row
        html_content += f"""
            <tr class="{class_style}">
                <td>{row['poi_id']}</td>
                <td>{row.get('poi_name', '')}</td>
                <td>{row.get('link_id', '')}</td>
                <td>{class_names[class_id] if 0 <= class_id < len(class_names) else f"Unknown ({class_id})"}</td>
                <td class="{confidence_class}">{row['confidence']:.2f}</td>
                <td>{row['action']}</td>
                <td>{vis_html}</td>
            </tr>
        """

    # Close table and HTML
    html_content += """
        </table>
    </body>
    </html>
    """

    # Write HTML to file
    report_file = f"{output_dir}/validation_report_with_images.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"HTML report with images generated: {report_file}")

# ==========================================
# NOTEBOOK-FRIENDLY EXECUTION FUNCTIONS
# ==========================================

def extract_poi_features(poi_row, roads_gdf):
    """Extract features with case-insensitive column handling"""
    features = {}
    
    try:
        # Get POI core data with case-insensitive column lookup
        link_id = None
        if isinstance(poi_row, pd.Series):
            # Try different case variations of link_id
            for link_col in ['link_id', 'link_id', 'LinkId', 'linkid']:
                if link_col in poi_row.index:
                    link_id = poi_row[link_col]
                    break
        
        # Get other POI attributes with fallbacks
        poi_side = 'R'  # Default right side
        if 'POI_ST_SD' in poi_row.index:
            poi_side = poi_row['POI_ST_SD']
        
        perc_from_ref = 0.5  # Default middle
        if 'PERCFRREF' in poi_row.index:
            try:
                perc_from_ref = float(poi_row['PERCFRREF'])
            except:
                pass
                
        # Try to find facility type
        facility_type = -1
        if 'FAC_TYPE' in poi_row.index:
            facility_type = poi_row['FAC_TYPE']
            
        # 1. Road relationship - CASE INSENSITIVE MATCHING
        road_match = None
        
        if link_id is not None and roads_gdf is not None:
            # Get lowercase column names dictionary for case-insensitive lookup
            lower_cols = {col.lower(): col for col in roads_gdf.columns}
            
            # Find the actual link_id column name in the roads data
            link_id_col = None
            for possible_col in ['link_id', 'linkid', 'link_id']:
                if possible_col.lower() in lower_cols:
                    link_id_col = lower_cols[possible_col.lower()]
                    break
            
            # Try to match if we found the column
            if link_id_col:
                # Convert both to strings for comparison
                road_match = roads_gdf[roads_gdf[link_id_col].astype(str) == str(link_id)]
                
                # If no match, try checking if there's a numeric difference
                if len(road_match) == 0 and str(link_id).isdigit():
                    # Try looking for partial matches
                    for _, road in roads_gdf.iterrows():
                        road_link_id = str(road[link_id_col])
                        if road_link_id.isdigit() and (road_link_id in str(link_id) or str(link_id) in road_link_id):
                            print(f"Found partial match: POI {link_id} ~ Road {road_link_id}")
                            # Create a single-row GeoDataFrame for this match
                            road_match = gpd.GeoDataFrame([road])
                            break
        
        # Check if we found a matching road
        if road_match is None or len(road_match) == 0:
            features.update({
                'no_matching_road': 1,
                'dist_to_road': 999,
                'is_multi_dig': 0,
                'road_side_match': 0,
                'road_side': 1 if poi_side == 'R' else 0,
                'perc_from_ref': float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
            })
        else:
            # We found a matching road
            features.update({
                'no_matching_road': 0,
                'dist_to_road': 0,
                'is_multi_dig': 0,  # Default if no MULTIDIGIT column
                'road_side_match': 1,  # Default
                'road_side': 1 if poi_side == 'R' else 0,
                'perc_from_ref': float(perc_from_ref) if isinstance(perc_from_ref, (int, float, str)) else 0.5,
            })
            
            # Check if MULTIDIGIT column exists in any form
            multi_dig_col = None
            for possible_col in ['MULTIDIGIT', 'multidigit', 'MultiDigit']:
                for col in roads_gdf.columns:
                    if col.lower() == possible_col.lower():
                        multi_dig_col = col
                        break
                if multi_dig_col:
                    break
            
            # Set multi_dig if column exists
            if multi_dig_col and multi_dig_col in road_match.columns:
                road = road_match.iloc[0]
                multi_dig_val = road[multi_dig_col]
                features['is_multi_dig'] = 1 if (isinstance(multi_dig_val, str) and multi_dig_val.upper() == 'Y') else 0
        
        # 2. POI type features
        exception_types = [3538, 4013, 1100, 4170, 4444, 4482, 4493, 4580, 4581, 5000]
        
        # Make sure facility_type is numeric
        if isinstance(facility_type, str) and facility_type.isdigit():
            facility_type = int(facility_type)
            
        features.update({
            'facility_type': facility_type if isinstance(facility_type, int) else -1,
            'is_exception_type': 1 if facility_type in exception_types else 0,
        })
        
        # Other POI attributes
        nat_import = 0
        if 'NAT_IMPORT' in poi_row.index:
            nat_val = poi_row['NAT_IMPORT']
            if pd.notna(nat_val):
                if isinstance(nat_val, (int, float)):
                    nat_import = 1 if nat_val else 0
                elif isinstance(nat_val, str):
                    nat_import = 1 if nat_val.upper() == 'Y' else 0
        features['nat_import'] = nat_import
        
        # Add defaults for any missing features the model expects
        expected_features = [
            'in_vicinity', 'has_chain_id', 'has_phone',
            'is_24hour', 'is_airport', 'is_entrance',
            'is_restaurant', 'private'
        ]
        
        for feat in expected_features:
            features[feat] = 0  # Default to 0 for all these flags
            
        # Try to populate them where data exists
        if 'IN_VICIN' in poi_row.index:
            features['in_vicinity'] = 1 if str(poi_row['IN_VICIN']).upper() == 'Y' else 0
            
        if 'CHAIN_ID' in poi_row.index and pd.notna(poi_row['CHAIN_ID']) and poi_row['CHAIN_ID'] != 0:
            features['has_chain_id'] = 1
            
        if 'PH_NUMBER' in poi_row.index and pd.notna(poi_row['PH_NUMBER']) and poi_row['PH_NUMBER'] != '':
            features['has_phone'] = 1
            
        if 'PRIVATE' in poi_row.index:
            features['private'] = 1 if str(poi_row['PRIVATE']).upper() == 'Y' else 0
            
        # Check restaurant and airport types
        if 'REST_TYPE' in poi_row.index and pd.notna(poi_row['REST_TYPE']):
            features['is_restaurant'] = 1
            
        if 'AIRPT_TYPE' in poi_row.index and pd.notna(poi_row['AIRPT_TYPE']):
            features['is_airport'] = 1
            
        return features
        
    except Exception as e:
        print(f"Error in extract_poi_features: {str(e)}")
        
        # Return default features that will classify as probable deletion
        return {
            'no_matching_road': 1,
            'dist_to_road': 999,
            'is_multi_dig': 0,
            'road_side_match': 0,
            'road_side': 1, 
            'perc_from_ref': 0.5,
            'facility_type': -1,
            'is_exception_type': 0,
            'nat_import': 0,
            'in_vicinity': 0,
            'has_chain_id': 0,
            'has_phone': 0,
            'is_24hour': 0,
            'is_airport': 0,
            'is_entrance': 0,
            'is_restaurant': 0,
            'private': 0
        }

def extract_class_specific_features(poi_row, roads_gdf):
    """Extrae características específicas para cada clase de problema"""
    features = {}
    
    # --- CARACTERÍSTICAS PARA CLASE 0 (POIs inexistentes) ---
    link_id = str(poi_row.get('LINK_ID', ''))
    # Verificar si el link_id existe en roads_gdf
    features['link_id_exists'] = 1 if link_id in roads_gdf['link_id'].astype(str).values else 0
    # Verificar formato de link_id
    features['valid_link_format'] = 1 if re.match(r'^\d+$', link_id) else 0
    # Distancia al segmento de calle más cercano si no hay coincidencia exacta
    if features['link_id_exists'] == 0 and not roads_gdf.empty:
        try:
            poi_point = get_poi_coordinates(poi_row)
            if poi_point:
                min_dist = float('inf')
                for _, road in roads_gdf.iterrows():
                    if hasattr(road.geometry, 'distance'):
                        dist = road.geometry.distance(Point(poi_point))
                        min_dist = min(min_dist, dist)
                features['min_road_distance'] = min_dist
                # POIs muy alejados de cualquier calle tienen mayor probabilidad de ser erróneos
                features['far_from_roads'] = 1 if min_dist > 0.001 else 0  # ~100m en grados decimales
        except Exception:
            features['min_road_distance'] = -1
            features['far_from_roads'] = 0
    
    # --- CARACTERÍSTICAS PARA CLASE 1 (Lado incorrecto) ---
    # Extraer lado actual de la calle
    features['poi_side'] = 1 if poi_row.get('POI_ST_SD') == 'R' else 0
    # Calcular lado correcto según geometría (si podemos)
    try:
        if link_id in roads_gdf['link_id'].astype(str).values:
            road = roads_gdf[roads_gdf['link_id'].astype(str) == link_id].iloc[0]
            poi_point = get_poi_coordinates(poi_row)
            if poi_point and hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                if len(coords) >= 2:
                    # Calcular lado real según geometría
                    actual_side = calculate_point_side(coords, poi_point)
                    features['calculated_side'] = 1 if actual_side == 'R' else 0
                    # ¿Coincide lado declarado con calculado?
                    features['side_mismatch'] = 1 if features['poi_side'] != features['calculated_side'] else 0
    except Exception:
        features['calculated_side'] = -1
        features['side_mismatch'] = 0
    
    # --- CARACTERÍSTICAS PARA CLASE 2 (MULTIDIGIT incorrecto) ---
    # Verificar si la vía tiene múltiples dígitos de carriles
    features['has_multidigit_attr'] = 1 if 'MULTIDIGIT' in roads_gdf.columns else 0
    if features['has_multidigit_attr'] and link_id in roads_gdf['link_id'].astype(str).values:
        road = roads_gdf[roads_gdf['link_id'].astype(str) == link_id].iloc[0]
        features['is_multidigit'] = 1 if road.get('MULTIDIGIT') == 'Y' else 0
        # Características que sugieren que no debería ser multidigit
        road_width = extract_road_width_if_available(road)
        features['narrow_road'] = 1 if road_width and road_width < 10 else 0  # en metros
        
    # --- CARACTERÍSTICAS PARA CLASE 3 (Excepciones legítimas) ---
    # Tipos de instalaciones que suelen ser excepciones
    exception_types = [7311, 7510, 7520, 7521, 7522, 7538, 7999]  # Aeropuertos, estaciones, etc.
    fac_type = poi_row.get('FAC_TYPE')
    features['is_exception_type'] = 1 if fac_type in exception_types else 0
    features['national_importance'] = int(poi_row.get('NAT_IMPORT', 0))
    
    # Agrega el resto de características básicas
    features.update(extract_basic_features(poi_row, roads_gdf))
    
    return features

# Función auxiliar para extraer coordenadas del POI
def get_poi_coordinates(poi_row):
    if 'geometry' in poi_row and poi_row['geometry']:
        return [poi_row['geometry'].x, poi_row['geometry'].y]
    elif 'lat' in poi_row and 'lon' in poi_row:
        return [poi_row['lon'], poi_row['lat']]
    return None

# Función para calcular de qué lado de la línea está un punto
def calculate_point_side(line_coords, point):
    if len(line_coords) < 2:
        return 'unknown'
    
    # Usar el primer segmento para determinar dirección
    x1, y1 = line_coords[0]
    x2, y2 = line_coords[1]
    
    # Calcular producto cruz para determinar lado
    cross_product = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
    
    # Positive = izquierda (L), Negative = derecha (R)
    return 'L' if cross_product > 0 else 'R'

def validate_pois(poi_file, roads_file, model_file, output_dir="validation_results", batch_size=10, resume_from=0):
    """Validate POIs using the trained model with proper feature matching"""
    print(f"Loading model from {model_file}")
    model = joblib.load(model_file)
    
    print(f"Loading POI data from {poi_file}")
    poi_data = pd.read_csv(poi_file)
    
    print(f"Loading road data from {roads_file}")
    road_data = gpd.read_file(roads_file)
    
    print(f"Loaded {len(poi_data)} POIs")
    print(f"POI columns: {', '.join(poi_data.columns)}")
    
    # Get expected feature names from model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print(f"Model expects these features: {', '.join(expected_features)}")
    else:
        print("Model doesn't have feature_names_in_ attribute")
        # Try to determine features from first extraction
        test_features = extract_poi_features(poi_data.iloc[0], road_data)
        expected_features = list(test_features.keys())
        print(f"Using extracted features: {', '.join(expected_features)}")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Check for existing results to resume
    results_file = os.path.join(output_dir, "validation_results.csv")
    if os.path.exists(results_file) and resume_from > 0:
        results = pd.read_csv(results_file)
        print(f"Resuming from POI #{resume_from} with existing results")
    else:
        results = pd.DataFrame()
    
    print(f"\nProcessing {len(poi_data)} POIs...")
    
    # Process POIs in batches
    for i in range(resume_from, len(poi_data), batch_size):
        batch_end = min(i + batch_size, len(poi_data))
        print(f"Processing batch {i//batch_size + 1}: POIs {i+1}-{batch_end}/{len(poi_data)}")
        
        batch_results = []
        for j in range(i, batch_end):
            poi = poi_data.iloc[j]
            poi_id = poi.get('POI_ID', f"poi_{j}")
            print(f"Processing POI {j+1}/{len(poi_data)}: {poi_id}")
            
            # Extract features
            features = extract_poi_features(poi, road_data)
            
            # Ensure we have all expected features
            for feature in expected_features:
                if feature not in features:
                    print(f"Warning: Feature '{feature}' missing from extraction, adding with default value 0")
                    features[feature] = 0
            
            # Create DataFrame with only the expected features in the right order
            X = pd.DataFrame([features])[expected_features].fillna(-1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            probas = model.predict_proba(X)[0]
            confidence = probas[prediction]
            
            # Generate action based on prediction
            actions = [
                "Mark for deletion (no POI in reality)",
                "Fix POI side of road",
                "Update MULTIDIGIT attribute to 'N'",
                "Keep as legitimate exception"
            ]
            action = actions[prediction] if prediction < len(actions) else "Unknown action"
            
            # Generate visualization and save as PNG
            visualization_path = None
            try:
                # Generate simple visualization first
                os.makedirs(images_dir, exist_ok=True)
                visualization_path = os.path.join(images_dir, f"poi_{poi_id}_class{prediction}.png")
                
                print(f"Generating visualization for POI {poi_id}")
                if generate_visualization(poi, road_data, images_dir, poi_id=poi_id):
                    print(f"Visualization saved to {visualization_path}")
                
                # If the above simple visualization works, we can also try the satellite visualization
                # But make it optional since it requires API calls
                try:
                    api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'  # Define API key here explicitly
                    
                    # Find matching road for satellite visualization
                    link_id = poi.get('LINK_ID')  # Use uppercase LINK_ID for POI data
                    if link_id is None:
                        for col in poi.index:
                            if col.upper() == 'LINK_ID':
                                link_id = str(poi[col])
                                break
                    
                    road_match = None
                    if link_id is not None:
                        road_link_col = 'link_id'  # Use lowercase link_id for road data
                        if road_link_col in road_data.columns:
                            road_match = road_data[road_data[road_link_col].astype(str) == str(link_id)]
                    
                    # If we found a matching road with valid geometry, create satellite visualization
                    if road_match is not None and len(road_match) > 0:
                        road = road_match.iloc[0]
                        if hasattr(road.geometry, 'coords') and len(list(road.geometry.coords)) >= 2:
                            satellite_path = os.path.join(images_dir, f"satellite_poi_{poi_id}.png")
                            print(f"Generating satellite visualization for POI {poi_id}")
                            visualize_poi(poi, road_data, output_dir=images_dir, zoom_level=17, api_key=api_key)
                except Exception as e:
                    print(f"Satellite visualization error for POI {poi_id}: {str(e)}")
                    
            except Exception as e:
                print(f"Error generating visualization for POI {poi_id}: {str(e)}")
            
            # Record result
            result = {
                'poi_id': poi_id,
                'link_id': poi.get('link_id', None),
                'poi_name': poi.get('POI_NAME', ''),
                'predicted_label': prediction,
                'confidence': confidence,
                'action': action,
                'visualization_path': visualization_path
            }
            
            # Get true label if available
            if 'classification' in poi_data.columns:
                true_label = poi.get('classification')
                result['true_label'] = true_label
            
            batch_results.append(result)
            
            # Rest a bit to prevent API rate limiting
            time.sleep(0.1)
        
        # Add batch results to all results
        batch_df = pd.DataFrame(batch_results)
        if results.empty:
            results = batch_df
        else:
            results = pd.concat([results, batch_df], ignore_index=True)
        
        # Save results after each batch
        results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")
        
        # Add longer pause between batches
        if batch_end < len(poi_data):
            print("Waiting before next batch...")
            time.sleep(1)
    
    print("\nValidation complete!")
    print(f"Results saved to {results_file}")
    
    # Update HTML report to include the visualizations
    try:
        generate_html_report_with_images(results, output_dir)
        print(f"Generated validation report with images in {output_dir}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
    
    return results

def enhanced_poi_validation_pipeline(poi_file, road_file, output_dir):
    """Pipeline mejorado para clasificación de POIs"""
    print(f"Procesando {poi_file} con {road_file}")
    
    # 1. Cargar datos
    pois = pd.read_csv(poi_file)
    roads = gpd.read_file(road_file)
    
    # 2. Entrenar modelo si no existe
    model_path = os.path.join(output_dir, "poi_specialized_model.joblib")
    if not os.path.exists(model_path):
        print("Entrenando nuevo modelo especializado...")
        # Generar datos de entrenamiento mejorados
        train_data = generate_enhanced_training_data(
            [road_file], 
            os.path.join(output_dir, "enhanced_training_data.csv"),
            samples_per_class=1000
        )
        
        # Extraer características
        X_train = []
        y_train = train_data['class_id'].values
        
        for _, poi in train_data.iterrows():
            features = extract_class_specific_features(poi, roads)
            X_train.append(features)
        
        X_train_df = pd.DataFrame(X_train).fillna(-999)
        
        # Entrenar los modelos especializados
        main_model, binary_models, preprocessors = train_specialized_classifier()
        
        # Normalizar datos
        X_train_scaled = preprocessors['main'].fit_transform(X_train_df)
        
        # Entrenar modelo principal
        main_model.fit(X_train_scaled, y_train)
        
        # Entrenar modelos binarios especializados
        binary_models['nonexistent_detector'].fit(
            preprocessors['nonexistent'].fit_transform(X_train_df),
            (y_train == 0).astype(int)
        )
        
        binary_models['wrong_side_detector'].fit(
            preprocessors['wrong_side'].fit_transform(X_train_df),
            (y_train == 1).astype(int)
        )
        
        binary_models['multidigit_detector'].fit(
            preprocessors['multidigit'].fit_transform(X_train_df),
            (y_train == 2).astype(int)
        )
        
        binary_models['exception_detector'].fit(
            preprocessors['exception'].fit_transform(X_train_df),
            (y_train == 3).astype(int)
        )
        
        # Guardar modelos
        model_package = {
            'main_model': main_model,
            'binary_models': binary_models,
            'preprocessors': preprocessors,
            'feature_names': list(X_train_df.columns)
        }
        joblib.dump(model_package, model_path)
        print(f"Modelo guardado en {model_path}")
    else:
        print(f"Cargando modelo existente desde {model_path}")
        model_package = joblib.load(model_path)
    
    # 3. Procesar POIs reales
    results = []
    
    for idx, poi in pois.iterrows():
        if idx % 100 == 0:
            print(f"Procesando POI {idx}/{len(pois)}")
            
        # Extraer características específicas para cada clase
        features = extract_class_specific_features(poi, roads)
        X = pd.DataFrame([features])
        
        # Asegurarse de que X tiene todas las columnas necesarias
        for col in model_package['feature_names']:
            if col not in X.columns:
                X[col] = -999
        
        # Reordenar columnas para que coincidan con el entrenamiento
        X = X[model_package['feature_names']]
        
        # Aplicar preprocesamiento
        X_scaled = model_package['preprocessors']['main'].transform(X)
        
        # Predecir con modelo principal
        class_pred = model_package['main_model'].predict(X_scaled)[0]
        class_probs = model_package['main_model'].predict_proba(X_scaled)[0]
        
        # Verificar con modelos específicos para mayor precisión
        specialized_preds = {
            0: model_package['binary_models']['nonexistent_detector'].predict_proba(
                model_package['preprocessors']['nonexistent'].transform(X))[0][1],
            1: model_package['binary_models']['wrong_side_detector'].predict_proba(
                model_package['preprocessors']['wrong_side'].transform(X))[0][1],
            2: model_package['binary_models']['multidigit_detector'].predict_proba(
                model_package['preprocessors']['multidigit'].transform(X))[0][1],
            3: model_package['binary_models']['exception_detector'].predict_proba(
                model_package['preprocessors']['exception'].transform(X))[0][1]
        }
        
        # Determinar clase final basada en todos los modelos
        final_class = class_pred
        max_conf = max(specialized_preds.values())
        if max_conf > 0.8:  # Alta confianza en predicción especializada
            final_class = max(specialized_preds, key=specialized_preds.get)
        
        # Determinar acción basada en la clase
        action = "No Action"
        if final_class == 0:
            action = "DELETE - POI does not exist"
        elif final_class == 1:
            action = f"CHANGE SIDE - from {poi.get('POI_ST_SD', '?')} to {'L' if poi.get('POI_ST_SD') == 'R' else 'R'}"
        elif final_class == 2:
            action = "FIX MULTIDIGIT - set to 'N'"
        elif final_class == 3:
            action = "KEEP - Legitimate exception"
        
        # Almacenar resultado
        results.append({
            'poi_id': poi.get('POI_ID', f"POI_{idx}"),
            'poi_name': poi.get('POI_NAME', ''),
            'link_id': poi.get('LINK_ID', ''),
            'class_id': int(final_class),
            'confidence': float(specialized_preds[final_class]),
            'action': action,
            # Guardar características relevantes para revisión
            'features': {k: v for k, v in features.items() if v != -999}
        })
        
        # Generar visualización para verificación
        try:
            poi_id = poi.get('POI_ID', f"POI_{idx}")
            visualize_poi_classification(poi, roads, final_class, specialized_preds[final_class], 
                                       os.path.join(output_dir, 'images'), poi_id)
        except Exception as e:
            print(f"Error en visualización para POI {idx}: {e}")
    
    # 4. Guardar resultados
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results])
    results_path = os.path.join(output_dir, "enhanced_poi_validation_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # 5. Generar informe
    generate_enhanced_html_report(results_df, results, output_dir)
    
    return results_df

def validate_with_confidence_analysis(poi_file, roads_file, model_file):
    """Validación con análisis de confianza y recomendaciones específicas"""
    model = joblib.load(model_file)
    pois = pd.read_csv(poi_file)
    roads = gpd.read_file(roads_file)
    
    results = []
    
    for idx, poi in pois.iterrows():
        # Extraer características mejoradas
        features = extract_enhanced_features(poi, roads)
        X = pd.DataFrame([features])
        
        # Predecir clase y probabilidades
        class_id = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        confidence = probas[class_id]
        
        # Análisis de confianza
        confidence_level = "alta" if confidence > 0.8 else "media" if confidence > 0.6 else "baja"
        
        # Generar recomendaciones específicas basadas en la clase y confianza
        if class_id == 0:  # No existe
            if confidence > 0.8:
                recommendation = "ELIMINAR: Este POI probablemente no existe"
            else:
                recommendation = "VERIFICAR: Posible POI inexistente (verificar en campo)"
        elif class_id == 1:  # Lado incorrecto
            current_side = poi.get('POI_ST_SD', 'desconocido')
            correct_side = 'L' if current_side == 'R' else 'R'
            recommendation = f"CORREGIR LADO: Cambiar de {current_side} a {correct_side}"
        elif class_id == 2:  # MULTIDIGIT incorrecto
            recommendation = "ACTUALIZAR ATRIBUTO: Cambiar MULTIDIGIT a 'N'"
        else:  # Excepción legítima
            recommendation = "MANTENER: Excepción válida"
        
        results.append({
            'poi_id': poi.get('POI_ID'),
            'class_id': int(class_id),
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'recommendation': recommendation
        })
    
    # Generar informe con estadísticas de confianza
    result_df = pd.DataFrame(results)
    result_df.to_csv("validation_with_confidence.csv", index=False)
    
    # Crear visualizaciones para casos ambiguos (confianza baja)
    ambiguous_cases = result_df[result_df['confidence'] < 0.6]
    print(f"Casos ambiguos para revisión manual: {len(ambiguous_cases)}")
    
    return result_df

def get_action_for_class(class_id):
    """Return the recommended action for a class ID"""
    actions = [
        "Mark for deletion (no POI in reality)",
        "Fix POI side of road",
        "Update MULTIDIGIT attribute to 'N'",
        "Keep as legitimate exception"
    ]
    return actions[class_id] if 0 <= class_id < len(actions) else "Unknown action"

def train_model(data_file, roads_file, output_file='poi_validation_model.joblib'):
    """Train a new POI validation model."""
    print(f"Training model using data from {data_file}")
    system = POIValidationSystem()
    labeled_data = pd.read_csv(data_file)
    roads = gpd.read_file(roads_file)
    system.train_model(labeled_data, roads)
    print(f"Model saved to {output_file}")
    return system.model

def generate_data(roads_file, output_file='synthetic_training_data.csv', count=1000):
    """Generate synthetic training data."""
    print(f"Generating {count} synthetic training samples")
    return generate_synthetic_training_data(
        roads_file=roads_file,
        output_file=output_file,
        num_samples=count
    )

def generate_mexico_city_roads(num_roads=1000, output_file="roads_mexico_city.geojson"):
    """Generate a synthetic Mexico City road network for testing."""
    # Base coordinates for different areas of Mexico City
    base_areas = [
        [-99.133, 19.432],  # Zócalo/Centro Histórico
        [-99.167, 19.420],  # Condesa/Roma
        [-99.204, 19.350],  # Coyoacán
        [-99.184, 19.372],  # San Ángel
        [-99.173, 19.438]   # Chapultepec/Polanco
    ]

    # Initialize FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": []
    }

    # Create a realistic distribution of road types
    # 70% single-digitized (MULTIDIGIT="N"), 30% multiply-digitized (MULTIDIGIT="Y")
    single_digitized_count = int(num_roads * 0.7)
    multi_digitized_count = (num_roads - single_digitized_count) // 2  # Each multiply-digitized road has 2 segments

    # Generate single-digitized roads
    for i in range(single_digitized_count):
        link_id = str(10000 + i)
        road_name = f"Calle {i}"

        # Select base area and generate coordinates
        base_lon, base_lat = random.choice(base_areas)

        # Create a slightly random road direction and length
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        segment_length = random.uniform(0.001, 0.005)  # Approx 100-500m

        # Generate 3-5 points for the LineString
        num_points = random.randint(3, 5)
        coordinates = []

        for j in range(num_points):
            lon = base_lon + j * segment_length * math.cos(angle) + random.uniform(-0.0002, 0.0002)
            lat = base_lat + j * segment_length * math.sin(angle) + random.uniform(-0.0002, 0.0002)
            coordinates.append([round(lon, 6), round(lat, 6)])

        # Create feature
        feature = {
            "type": "Feature",
            "properties": {
                "link_id": link_id,
                "MULTIDIGIT": "N",
                "ROAD_NAME": road_name
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }

        feature_collection["features"].append(feature)

    # Generate multiply-digitized roads (pairs of parallel roads)
    for i in range(multi_digitized_count):
        # Create base link_id for the pair
        base_id = 50000 + i
        forward_id = str(base_id)
        toward_id = str(base_id + 100000)

        road_name = f"Avenida {i}"

        # Select base area and generate coordinates
        base_lon, base_lat = random.choice(base_areas)

        # Create a slightly random road direction and length
        angle = random.uniform(0, 2 * math.pi)
        segment_length = random.uniform(0.002, 0.008)

        num_points = random.randint(3, 7)
        base_coordinates = []

        for j in range(num_points):
            lon = base_lon + j * segment_length * math.cos(angle) + random.uniform(-0.0001, 0.0001)
            lat = base_lat + j * segment_length * math.sin(angle) + random.uniform(-0.0001, 0.0001)
            base_coordinates.append([round(lon, 6), round(lat, 6)])

        # Calculate perpendicular offset for parallel roads
        offset = 0.00015  # Approximately 15 meters
        perp_angle = angle + math.pi/2
        offset_x = offset * math.cos(perp_angle)
        offset_y = offset * math.sin(perp_angle)

        # Create coordinates for forward direction
        forward_coordinates = []
        for j in range(len(base_coordinates)):
            lon = base_coordinates[j][0] + offset_x
            lat = base_coordinates[j][1] + offset_y
            forward_coordinates.append([round(lon, 6), round(lat, 6)])

        # Create coordinates for toward direction
        toward_coordinates = []
        for j in range(len(base_coordinates)):
            lon = base_coordinates[j][0] - offset_x
            lat = base_coordinates[j][1] - offset_y
            toward_coordinates.append([round(lon, 6), round(lat, 6)])

        # Create forward direction feature
        forward_feature = {
            "type": "Feature",
            "properties": {
                "link_id": forward_id,
                "MULTIDIGIT": "Y",
                "ROAD_NAME": road_name,
                "DIR_TRAVEL": "F"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": forward_coordinates
            }
        }

        # Create toward direction feature
        toward_feature = {
            "type": "Feature",
            "properties": {
                "link_id": toward_id,
                "MULTIDIGIT": "Y",
                "ROAD_NAME": road_name,
                "DIR_TRAVEL": "T"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": toward_coordinates
            }
        }

        feature_collection["features"].append(forward_feature)
        feature_collection["features"].append(toward_feature)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(feature_collection, f)

    print(f"Generated {len(feature_collection['features'])} Mexico City road segments in {output_file}")
    return feature_collection

def generate_sample_pois(roads_file, num_pois=1000, output_file="sample_pois.csv"):
    """Generate sample POIs for validation testing based on road network."""
    print(f"Generating {num_pois} sample POIs for validation")

    # Load roads data
    roads = gpd.read_file(roads_file)

    # Initialize sample POIs list
    pois = []

    # Calculate how many POIs to generate in each category
    pois_per_category = num_pois // 4  # Four categories

    # Create a road pool we can sample from repeatedly if needed
    single_dig_roads = roads[roads['MULTIDIGIT'] == 'N']
    multi_dig_roads = roads[roads['MULTIDIGIT'] == 'Y']

    # If not enough roads, allow reusing roads (multiple POIs per road)
    pois_added = 0

    # Keep generating until we reach the target or hit some limit
    max_attempts = min(num_pois * 2, 5000)  # Avoid infinite loops
    attempts = 0

    while pois_added < num_pois and attempts < max_attempts:
        # Sample a road (with replacement if needed)
        if len(pois) % 4 == 0 or len(pois) % 4 == 1:  # 50% on single-dig roads
            if len(single_dig_roads) > 0:
                road = single_dig_roads.sample(1).iloc[0]
            else:
                road = roads.sample(1).iloc[0]
        else:  # 50% on multi-dig roads
            if len(multi_dig_roads) > 0:
                road = multi_dig_roads.sample(1).iloc[0]
            else:
                road = roads.sample(1).iloc[0]

        # Only proceed if road has valid geometry
        if hasattr(road.geometry, 'coords'):
            coords = list(road.geometry.coords)
            if len(coords) >= 2:
                # Create a POI
                perc_from_ref = random.random()
                poi_side = random.choice(['R', 'L'])

                # Add POI to list
                poi = {
                    'POI_ID': f"poi_{pois_added}",
                    'link_id': road.get('link_id', ''),
                    'POI_NAME': f"Sample POI {pois_added}",
                    'FAC_TYPE': random.randint(1000, 7000),
                    'POI_ST_SD': poi_side,
                    'PERCFRREF': perc_from_ref,
                    'NAT_IMPORT': random.choice([0, 1]),
                    'IN_VICIN': random.choice(['N', 'Y']),
                    'PRIVATE': random.choice(['N', 'Y'])
                }

                pois.append(poi)
                pois_added += 1

                # Print progress
                if pois_added % 100 == 0:
                    print(f"Generated {pois_added}/{num_pois} POIs")

        attempts += 1

    # Create DataFrame and save to CSV
    df = pd.DataFrame(pois)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} sample POIs and saved to {output_file}")

    return df

def run_complete_validation_pipeline(data_dir):
    """Ejecuta todo el pipeline: entrenamiento, validación y reporte"""
    # 1. Preparar datos de entrenamiento mejorados
    roads_file = glob.glob(f"{data_dir}/STREETS_*.geojson")[0]
    training_data = create_improved_training_data(
        roads_file=roads_file,
        output_file=f"{data_dir}/enhanced_training_data.csv",
        samples_per_class=1000
    )
    
    # 2. Entrenar modelo avanzado
    model = train_advanced_model(training_data, gpd.read_file(roads_file))
    
    # 3. Procesar todos los archivos POI
    poi_files = glob.glob(f"{data_dir}/POI_*.csv")
    all_results = []
    
    for poi_file in poi_files:
        file_id = os.path.basename(poi_file).split('_')[1].split('.')[0]
        matched_road_file = glob.glob(f"{data_dir}/*{file_id}.geojson")[0]
        
        print(f"Procesando {poi_file} con {matched_road_file}")
        results = validate_with_confidence_analysis(poi_file, matched_road_file, "advanced_poi_model.joblib")
        all_results.append(results)
    
    # 4. Generar reporte consolidado con estadísticas
    # ...
    
    return all_results

# Modified if __name__ == "__main__": block with fallback data generation
if __name__ == "__main__":
    import glob
    import os
    import pandas as pd

    # Update patterns to match your actual file paths
    poi_files = glob.glob("**/POI_*.csv", recursive=True)
    street_nav_files = glob.glob("**/SREETS_NAV_*.geojson", recursive=True)  
    street_naming_files = glob.glob("**/SREETS_NAMING_ADDRESSING_*.geojson", recursive=True)
    
    print(f"Found {len(poi_files)} POI files")
    print(f"Found {len(street_nav_files)} STREETS_NAV files")
    print(f"Found {len(street_naming_files)} STREETS_NAMING_ADDRESSING files")

    # Extract the numeric IDs
    poi_ids = [file.split('POI_')[1].split('.csv')[0] for file in poi_files]
    
    # Create map from ID to file path for easier lookup
    poi_map = {file.split('POI_')[1].split('.csv')[0]: file for file in poi_files}
    road_map = {file.split('SREETS_NAMING_ADDRESSING_')[1].split('.geojson')[0]: file for file in street_naming_files}
    nav_map = {file.split('SREETS_NAV_')[1].split('.geojson')[0]: file for file in street_nav_files}

    # Find common IDs between POI and any road type
    all_road_ids = set([*road_map.keys(), *nav_map.keys()])
    common_ids = set(poi_ids).intersection(all_road_ids)
    
    if not common_ids:
        print("No matching POI and road files found. Check your file naming.")
        exit(1)

    print(f"Found {len(common_ids)} matching POI and road file pairs")

    # Select first road file for training data generation and model training
    first_id = list(common_ids)[0]
    first_road_file = road_map.get(first_id, nav_map.get(first_id))
    
    # Define a simplified function to generate training data without relying on MULTIDIGIT column
    def generate_simple_training_data(roads_file, output_file, num_samples=1000):
        """Generate simple training data without relying on specific columns"""
        print(f"Using simplified training data generator for {roads_file}")
        try:
            # Load roads GeoDataFrame
            import geopandas as gpd
            roads = gpd.read_file(roads_file)
            print(f"Loaded {len(roads)} road segments with columns: {list(roads.columns)}")
            
            # Generate synthetic POIs with balanced classes
            import random
            
            data = []
            num_per_class = num_samples // 4
            
            # Sample roads - reuse them if we don't have enough
            road_samples = roads.sample(min(num_samples, len(roads)), replace=True)
            
            # Class 0: POIs marked for deletion (invalid link_id)
            for i in range(num_per_class):
                data.append({
                    'POI_ID': f"SYN_DEL_{i}",
                    'link_id': f"INVALID_{i}",
                    'FAC_TYPE': random.randint(1000, 9000),
                    'POI_NAME': f"Delete POI {i}",
                    'POI_ST_SD': random.choice(['R', 'L']),
                    'PERCFRREF': random.random(),
                    'NAT_IMPORT': 0,
                    'classification': 0  # Delete
                })
            
            # Class 1: Wrong side of road
            for i in range(num_per_class):
                road = road_samples.iloc[i % len(road_samples)]
                data.append({
                    'POI_ID': f"SYN_SIDE_{i}",
                    'link_id': str(road.get('link_id', f"L{i}")),
                    'FAC_TYPE': random.randint(1000, 9000),
                    'POI_NAME': f"Wrong Side POI {i}",
                    'POI_ST_SD': random.choice(['R', 'L']),
                    'PERCFRREF': random.random(),
                    'NAT_IMPORT': 0,
                    'classification': 1  # Wrong side
                })
            
            # Class 2: Incorrect MULTIDIGIT (we simulate based on facility type)
            for i in range(num_per_class):
                road = road_samples.iloc[(i + num_per_class) % len(road_samples)]
                data.append({
                    'POI_ID': f"SYN_MULTI_{i}",
                    'link_id': str(road.get('link_id', f"L{i+1000}")),
                    'FAC_TYPE': random.randint(1000, 3000),  # Non-exception types
                    'POI_NAME': f"Fix MULTIDIGIT POI {i}",
                    'POI_ST_SD': random.choice(['R', 'L']),
                    'PERCFRREF': random.random(),
                    'NAT_IMPORT': 0,
                    'classification': 2  # Fix MULTIDIGIT
                })
            
            # Class 3: Legitimate exceptions (special facility types)
            exception_types = [3538, 4013, 1100, 4170, 4444, 4482, 4493, 4580, 4581, 5000]
            for i in range(num_per_class):
                road = road_samples.iloc[(i + 2*num_per_class) % len(road_samples)]
                data.append({
                    'POI_ID': f"SYN_EXC_{i}",
                    'link_id': str(road.get('link_id', f"L{i+2000}")),
                    'FAC_TYPE': random.choice(exception_types),
                    'POI_NAME': f"Exception POI {i}",
                    'POI_ST_SD': random.choice(['R', 'L']),
                    'PERCFRREF': random.random(),
                    'NAT_IMPORT': 1,
                    'classification': 3  # Legitimate exception
                })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"Created {len(df)} synthetic samples with {num_per_class} POIs per class")
            
            return df
            
        except Exception as e:
            print(f"Error in simplified training data generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create minimal dataset as fallback
            fallback_data = []
            for class_id in range(4):  # 0-3 classes
                for i in range(50):  # 50 samples per class
                    fallback_data.append({
                        'POI_ID': f"FB_{class_id}_{i}",
                        'link_id': f"LINK_{i}" if class_id > 0 else f"INVALID_{i}",
                        'FAC_TYPE': 4580 if class_id == 3 else random.randint(1000, 3000),
                        'POI_NAME': f"Class {class_id} POI {i}",
                        'POI_ST_SD': 'R',
                        'PERCFRREF': 0.5,
                        'NAT_IMPORT': 1 if class_id == 3 else 0,
                        'classification': class_id
                    })
            
            df = pd.DataFrame(fallback_data)
            df.to_csv(output_file, index=False)
            print(f"Created fallback training data with {len(df)} samples")
            return df

    # Use our robust synthetic data generator
    synthetic_data_file = "synthetic_training_data.csv"
    print(f"Generating synthetic training data using roads from {first_road_file}")
    
    try:
        # Use the simplified generator that doesn't rely on MULTIDIGIT column
        synthetic_data = generate_simple_training_data(
            roads_file=first_road_file,
            output_file=synthetic_data_file, 
            num_samples=2000
        )
    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # If we can't create new synthetic data but have existing data, use that
        if os.path.exists(synthetic_data_file):
            print(f"Using existing synthetic data from {synthetic_data_file}")
        else:
            print("No synthetic training data available. Cannot continue.")
            exit(1)
    
    # Train a new model using synthetic data
    print("Training new model with synthetic data...")
    try:
        model = train_model(
            data_file=synthetic_data_file,
            roads_file=first_road_file,
            output_file="poi_validation_model.joblib"
        )
        print("Model trained successfully!")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # If training fails but we have an existing model, use that
        if os.path.exists("poi_validation_model.joblib"):
            print("Using existing model from poi_validation_model.joblib")
        else:
            # Create a simple fallback model as last resort
            print("Creating simple fallback model...")
            try:
                from sklearn.ensemble import RandomForestClassifier
                import joblib
                
                # Create a simple model with minimal features
                X = pd.read_csv(synthetic_data_file)
                if 'classification' in X.columns:
                    y = X['classification']
                    # Use only basic numeric features
                    feature_cols = ['road_side', 'is_exception_type', 'nat_import']
                    for col in feature_cols:
                        if col not in X.columns:
                            X[col] = 0  # Add missing columns with default values
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X[feature_cols], y)
                    joblib.dump(model, "poi_validation_model.joblib")
                    print("Created simple fallback model")
                else:
                    print("Cannot train model: no classification column in synthetic data")
                    exit(1)
            except Exception as e2:
                print(f"Failed to create fallback model: {str(e2)}")
                exit(1)

    # Now validate the real POI data using our model
    print("\n=== Validating Real POI Data ===")
    
    # Create a summary dataframe to track performance across all datasets
    overall_results = pd.DataFrame(columns=['file_id', 'poi_count', 'class_0_count', 
                                           'class_1_count', 'class_2_count', 'class_3_count'])
    
    # Process each pair of files
    for file_id in common_ids:
        print(f"\n===== Processing POI and road files with ID: {file_id} =====")
        
        poi_file = poi_map[file_id]
        # Prefer NAMING_ADDRESSING over NAV if both exist
        road_file = road_map.get(file_id, nav_map.get(file_id))
        
        output_dir = f"validation_results_{file_id}"
        
        # Validate POIs
        print(f"Validating POIs from {poi_file} against roads from {road_file}")
        try:
            results = validate_pois(
                poi_file=poi_file,
                roads_file=road_file,
                model_file="poi_validation_model.joblib",
                output_dir=output_dir,
                batch_size=10  # Adjust batch size as needed
            )
            
            # Save results summary for this dataset
            if results is not None and 'predicted_label' in results.columns:
                summary = results['predicted_label'].value_counts().to_dict()
                summary_row = {
                    'file_id': file_id,
                    'poi_count': len(results),
                    'class_0_count': summary.get(0, 0),  # Deletion
                    'class_1_count': summary.get(1, 0),  # Wrong side
                    'class_2_count': summary.get(2, 0),  # Incorrect MULTIDIGIT
                    'class_3_count': summary.get(3, 0)   # Legitimate exception
                }
                overall_results = pd.concat([overall_results, pd.DataFrame([summary_row])], ignore_index=True)
            
            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save overall summary
    if not overall_results.empty:
        summary_path = "validation_summary.csv"
        overall_results.to_csv(summary_path, index=False)
        
        # Print overall statistics
        total_pois = overall_results['poi_count'].sum()
        if total_pois > 0:
            total_class0 = overall_results['class_0_count'].sum()
            total_class1 = overall_results['class_1_count'].sum() 
            total_class2 = overall_results['class_2_count'].sum()
            total_class3 = overall_results['class_3_count'].sum()
            
            print("\n=== OVERALL VALIDATION SUMMARY ===")
            print(f"Total POIs processed across all datasets: {total_pois}")
            print(f"POIs to delete: {total_class0} ({total_class0/total_pois*100:.1f}%)")
            print(f"POIs with wrong side: {total_class1} ({total_class1/total_pois*100:.1f}%)")
            print(f"POIs with incorrect MULTIDIGIT: {total_class2} ({total_class2/total_pois*100:.1f}%)")
            print(f"Legitimate exceptions: {total_class3} ({total_class3/total_pois*100:.1f}%)")
            print(f"Summary saved to {summary_path}")