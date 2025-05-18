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
from datetime import datetime
import requests
from PIL import Image, ImageDraw
import io
import json
# Global variables
tile_size = 512

# Extraer coordenadas del POI
def get_poi_coordinates(poi_row):
    """Extrae coordenadas del POI desde diferentes formatos posibles"""
    if 'geometry' in poi_row and hasattr(poi_row['geometry'], 'x'):
        return [poi_row['geometry'].x, poi_row['geometry'].y]
    elif 'x' in poi_row and 'y' in poi_row:
        return [poi_row['x'], poi_row['y']]
    elif 'lon' in poi_row and 'lat' in poi_row:
        return [poi_row['lon'], poi_row['lat']]
    
    # Buscar en columnas XY
    for x_col, y_col in [('X', 'Y'), ('LON', 'LAT')]:
        if x_col in poi_row and y_col in poi_row:
            return [poi_row[x_col], poi_row[y_col]]
    
    return None

def get_geometry_coordinates(geometry):
    """Obtiene coordenadas de geometrías simples o multi-parte de manera segura"""
    from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
    
    coords = []
    
    # Manejar distintos tipos de geometría
    if hasattr(geometry, 'coords'):
        # Geometría simple (Point, LineString)
        coords = list(geometry.coords)
    elif hasattr(geometry, 'geoms'):
        # Geometría múltiple (MultiPoint, MultiLineString, etc)
        # Tomamos las coordenadas de la primera sub-geometría
        for geom in geometry.geoms:
            if hasattr(geom, 'coords'):
                coords.extend(list(geom.coords))
            # Si necesitamos solo el primer sub-elemento, podemos romper el bucle aquí
            # break
    
    return coords

def calculate_point_side(line_coords, point):
    """Determina de qué lado de una línea está un punto (L/R)"""
    if len(line_coords) < 2:
        return 'unknown'
    
    # Usar el primer segmento para determinar dirección
    x1, y1 = line_coords[0]
    x2, y2 = line_coords[1]
    
    # Calcular producto cruz para determinar lado
    cross_product = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
    
    # Positive = izquierda (L), Negative = derecha (R)
    return 'L' if cross_product > 0 else 'R'
    
# Extracción de ID de archivo
def extract_file_id(filename):
    """Extrae el ID del archivo desde su nombre"""
    import os
    import re
    
    basename = os.path.basename(filename)
    # Busca un patrón como POI_12345.csv
    match = re.search(r'[_-](\d+)', basename)
    if match:
        return match.group(1)
    return None

# Encontrar archivo de calles correspondiente
def find_matching_road_file(file_id, road_files):
    """Encuentra el archivo de calles que corresponde al ID de POI"""
    if not file_id:
        return None
    
    for road_file in road_files:
        if file_id in road_file:
            return road_file
    return None

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

def get_and_mark_expanded_satellite_tile(lat, lon, zoom, shape_points, reference_node, non_reference_node,
                                     tile_format, api_key, width_tiles=3, height_tiles=1, 
                                     output_dir="tiles", identifier=None, pois=None):
    """
    Download multiple satellite tiles and combine them to create a rectangular map visualization
    with road elements and POIs drawn on it.
    
    Args:
        lat, lon: Center coordinates
        zoom: Zoom level
        shape_points, reference_node, non_reference_node: Road geometry data
        tile_format: Image format (e.g., 'png')
        api_key: API key for HERE Maps
        width_tiles: Number of tiles horizontally (default 3 for rectangular view)
        height_tiles: Number of tiles vertically (default 1 for rectangular view)
        output_dir: Directory to save the output
        identifier: Optional identifier for the output filename
        pois: List of POIs to mark on the map
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the central tile coordinates
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    
    # Calculate ranges of tiles to download
    x_min = center_x - width_tiles//2
    x_max = center_x + width_tiles//2 + (0 if width_tiles % 2 == 1 else 1)
    y_min = center_y - height_tiles//2
    y_max = center_y + height_tiles//2 + (0 if height_tiles % 2 == 1 else 1)
    
    # Prepare the combined image
    combined_width = tile_size * (x_max - x_min)
    combined_height = tile_size * (y_max - y_min)
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    # Track the bounds of the entire map area
    all_bounds = []
    
    # Download and combine tiles
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            # Calculate position in the combined image
            pos_x = (x - x_min) * tile_size
            pos_y = (y - y_min) * tile_size
            
            # Get tile bounds and add to list
            tile_bounds = get_tile_bounds(x, y, zoom)
            all_bounds.extend(tile_bounds)
            
            # Download tile
            url = f'https://maps.hereapi.com/v3/base/mc/{zoom}/{x}/{y}/{tile_format}?style=satellite.day&size={tile_size}&apiKey={api_key}'
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    tile_img = Image.open(io.BytesIO(response.content))
                    combined_image.paste(tile_img, (pos_x, pos_y))
                else:
                    print(f"Failed to get tile {x},{y}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading tile {x},{y}: {e}")
    
    # Extract overall bounds
    lats = [p[1] for p in all_bounds]
    lons = [p[0] for p in all_bounds]
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)
    
    # Begin drawing on the combined image
    draw = ImageDraw.Draw(combined_image)
    
    # Define filename for the marked image
    if identifier:
        marked_filename = f'{output_dir}/expanded_{identifier}_marked.{tile_format}'
    else:
        marked_filename = f'{output_dir}/expanded_{lat:.5f}_{lon:.5f}_z{zoom}_marked.{tile_format}'
    
    try:
        # Function to convert geo coordinates to pixel coordinates on the combined image
        def geo_to_pixel(lon, lat):
            x = int((lon - min_lon) / (max_lon - min_lon) * combined_width)
            y = int((max_lat - lat) / (max_lat - min_lat) * combined_height)
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
            print(f"Drawing {len(pois)} POIs on expanded tile")
            for poi in pois:
                try:
                    # Get POI properties with case-insensitive fallback
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
        combined_image.save(marked_filename)
        print(f"Marked expanded tile saved to {marked_filename}")
        
        return marked_filename
        
    except Exception as e:
        print(f"Error drawing elements on expanded map: {str(e)}")
        # Save the original image in case of error
        combined_image.save(marked_filename)
        print(f"Saved original expanded tile to {marked_filename} due to error")
        import traceback
        traceback.print_exc()
        
        return marked_filename

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
    """Generate visualization using satellite imagery instead of matplotlib"""
    try:
        # Define API key explicitly 
        api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        if poi_id is None:
            if 'POI_ID' in poi_row:
                poi_id = poi_row['POI_ID']
            else:
                poi_id = f"poi_{random.randint(10000, 99999)}"
        
        # Get link_id with case-insensitive lookup
        link_id = None
        for col in poi_row.index:
            if col.upper() == 'LINK_ID':
                link_id = str(poi_row[col])
                break
        
        if link_id is None:
            print(f"Warning: No LINK_ID found for POI {poi_id}")
            return None
            
        # Find matching road
        road_match = None
        for col in roads_gdf.columns:
            if col.lower() == 'link_id':
                road_match = roads_gdf[roads_gdf[col].astype(str) == link_id]
                if not road_match.empty:
                    break
        
        # If we found a matching road with valid geometry, visualize it
        if road_match is not None and len(road_match) > 0:
            road = road_match.iloc[0]
            
            if hasattr(road.geometry, 'coords'):
                shape_points = list(road.geometry.coords)
                
                if len(shape_points) >= 2:
                    # Get reference nodes
                    reference_node, non_reference_node, _ = determine_reference_node(shape_points)
                    
                    # Calculate center
                    center_lon = sum(p[0] for p in shape_points) / len(shape_points)
                    center_lat = sum(p[1] for p in shape_points) / len(shape_points)
                    
                    # Generate satellite image with road and POI markers
                    marked_filename = get_and_mark_expanded_satellite_tile(
        lat, lon, zoom, shape_points, reference_node, non_reference_node,
        'png', api_key, width_tiles=6, height_tiles=2,  # Duplicado
        output_dir=output_folder, identifier=f"poi_{poi_id}", pois=[poi_row]
)
                    
                    print(f"Satellite image generated: {filename}")
                    return True
                    
        # If no match or invalid geometry, get default tile
        # Fall back to default location (point near POI or city center)
        poi_coords = get_poi_coordinates(poi_row)
        if poi_coords:
            lat, lon = poi_coords[1], poi_coords[0]  # Flip for lat/lon order
        else:
            # Default to Mexico City center if no coordinates available
            lat, lon = 19.432, -99.133
            
        wkt_bounds, filename = get_satellite_tile(
            lat, lon, 17, 'png', api_key,
            output_dir=output_folder,
            identifier=f"poi_{poi_id}_default"
        )
        
        print(f"Default satellite image generated: {filename}")
        return True
        
    except Exception as e:
        print(f"Error generating satellite visualization for POI {poi_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_custom_shaped_map(bounds, zoom, api_key, format='png', map_type='satellite.day'):
    """
    Obtiene un mapa con forma personalizada basada en los límites calculados.
    
    Args:
        bounds: Tupla (min_lat, min_lon, max_lat, max_lon)
        zoom: Nivel de zoom
        api_key: Clave de API de HERE
        format: Formato de imagen ('png' por defecto)
        map_type: Tipo de mapa ('satellite.day' para vista satelital)
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    
    # Calcular el ancho y alto del mapa en pixels
    n = 2.0 ** zoom
    width_px = int(n * ((max_lon - min_lon) / 360.0) * 256 * 2)
    height_px = int(n * ((max_lat - min_lat) / 180.0) * 256 * 2)
    
    # Asegurar dimensiones mínimas y máximas
    width_px = max(min(width_px, 2048), 512)
    height_px = max(min(height_px, 2048), 512)
    
    # Construir la URL de la API de HERE para solicitar un mapa con estas dimensiones
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # URL de la API con los parámetros calculados
    base_url = "https://image.maps.ls.hereapi.com/maptile/2.1/mapview"
    
    params = {
        "apiKey": api_key,
        "c": f"{center_lat},{center_lon}",  # Centro del mapa
        "z": zoom,  # Nivel de zoom
        "w": width_px,  # Ancho en pixels
        "h": height_px,  # Alto en pixels
        "t": map_type,  # Tipo de mapa (satellite.day, terrain.day, etc.)
        "f": format,  # Formato de salida
        "ppi": 320,  # Resolución de pixels por pulgada (mayor calidad)
        "nodot": True  # Eliminar marcador central
    }
    
    # Construir la URL con los parámetros
    url_params = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{url_params}"
    
    return width_px, height_px, url

def generate_adaptive_map_visualization(road_geometry, poi_location=None, buffer_percentage=0.3):
    """
    Genera una visualización que se adapta automáticamente a la forma de la carretera.
    
    Args:
        road_geometry: LineString o lista de puntos que representan la carretera
        poi_location: Ubicación del POI (opcional)
        buffer_percentage: Porcentaje de margen adicional alrededor de la ruta
    """
    # Extraer puntos de la geometría
    if hasattr(road_geometry, 'coords'):
        points = list(road_geometry.coords)
    else:
        points = road_geometry
        
    # Añadir POI a los puntos si existe
    if poi_location:
        if hasattr(poi_location, 'coords'):
            points.append(poi_location.coords[0])
        else:
            points.append((poi_location[1], poi_location[0]))  # lon, lat
    
    # Calcular límites
    lats = [p[1] for p in points]  # Latitudes
    lons = [p[0] for p in points]  # Longitudes
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Añadir buffer proporcional
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Determinar si el mapa debe ser más ancho o alto
    aspect_ratio = lon_range / lat_range if lat_range > 0 else 1.0
    
    # Ajustar el buffer para mantener una vista equilibrada
    min_lat -= lat_range * buffer_percentage
    max_lat += lat_range * buffer_percentage
    min_lon -= lon_range * buffer_percentage
    max_lon += lon_range * buffer_percentage
    
    # Calcular centro y zoom óptimo
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Calcular zoom óptimo basado en los límites
    zoom = calculate_optimal_zoom(min_lat, min_lon, max_lat, max_lon)
    
    return center_lat, center_lon, zoom, aspect_ratio, (min_lat, min_lon, max_lat, max_lon)

def poi_problem_detector(poi_df, roads_gdf, config=None):
    """
    Analiza POIs y detecta posibles problemas según los 4 criterios
    
    Args:
        poi_df: DataFrame con datos de POIs
        roads_gdf: GeoDataFrame con datos de calles
        config: Diccionario de configuración con umbrales
    
    Returns:
        DataFrame con POIs y sus problemas detectados
    """
    # Configuración predeterminada
    if config is None:
        config = {
            'dist_threshold': 0.001,  # ~100m en grados decimales
            'confidence_threshold': 0.75
        }
    
    # Preparar resultados
    results = []
    
    # Analizar cada POI
    for idx, poi in poi_df.iterrows():
        poi_id = poi.get('POI_ID', f"POI_{idx}")
        link_id = str(poi.get('LINK_ID', ''))
        
        # Inicializar detección para este POI
        detection = {
            'poi_id': poi_id,
            'link_id': link_id,
            'poi_name': poi.get('POI_NAME', ''),
            'problems': [],
            'flags': {}
        }
        
        # 1. Verificar existencia (Clase 0)
        matching_road = roads_gdf[roads_gdf['link_id'].astype(str) == link_id]
        if matching_road.empty:
            detection['problems'].append('nonexistent')
            detection['flags']['nonexistent_confidence'] = 0.9  # Alta confianza si no hay coincidencia
        
        # Si hay una carretera coincidente, realizar verificaciones adicionales
        if not matching_road.empty:
            road = matching_road.iloc[0]
            
            # 2. Verificar lado de calle (Clase 1)
            poi_side = poi.get('POI_ST_SD')
            if poi_side:
                # Obtener coordenadas de la geometría de manera segura
                road_coords = get_geometry_coordinates(road.geometry)
                
                if road_coords and len(road_coords) >= 2:
                    poi_coords = get_poi_coordinates(poi)
                    if poi_coords:
                        # Calcular lado real según geometría
                        actual_side = calculate_point_side(road_coords, poi_coords)
                        if actual_side != poi_side:
                            detection['problems'].append('wrong_side')
                            detection['flags']['wrong_side_confidence'] = 0.8
                            detection['flags']['correct_side'] = actual_side
            
            # 3. Verificar MULTIDIGIT (Clase 2)
            if 'MULTIDIGIT' in road:
                multidigit_value = road['MULTIDIGIT']
                if multidigit_value == 'Y':
                    # Verificar si debería ser 'Y' según tipo de carretera
                    road_type = road.get('ROAD_TYPE', '')
                    if road_type not in ['highway', 'motorway', 'trunk']:
                        detection['problems'].append('wrong_multidigit')
                        detection['flags']['wrong_multidigit_confidence'] = 0.75
            
            # 4. Verificar si es una excepción legítima (Clase 3)
            fac_type = poi.get('FAC_TYPE')
            exception_types = [7311, 7510, 7520, 7521, 7522, 7538, 7999]  # Aeropuertos, estaciones, etc.
            if fac_type in exception_types:
                detection['problems'].append('legitimate_exception')
                detection['flags']['exception_confidence'] = 0.85
        
        # Determinar la clasificación final
        if 'legitimate_exception' in detection['problems']:
            detection['class'] = 3
            detection['action'] = 'KEEP - Legitimate exception'
        elif 'nonexistent' in detection['problems']:
            detection['class'] = 0
            detection['action'] = 'DELETE - POI does not exist'
        elif 'wrong_side' in detection['problems']:
            detection['class'] = 1
            detection['action'] = f"CHANGE SIDE - from {poi_side} to {detection['flags'].get('correct_side', '?')}"
        elif 'wrong_multidigit' in detection['problems']:
            detection['class'] = 2
            detection['action'] = "FIX MULTIDIGIT - set to 'N'"
        else:
            detection['class'] = -1  # Sin problemas
            detection['action'] = 'No action needed'
        
        # VALIDACIÓN: Asegurar que la clase esté dentro del rango válido
        valid_classes = [-1, 0, 1, 2, 3]
        if detection['class'] not in valid_classes:
            print(f"Advertencia: Clase inválida {detection['class']} detectada para POI {poi_id}, cambiando a clase 1")
            detection['class'] = 1  # Usar clase 1 (lado incorrecto) como opción segura de fallback
            detection['action'] = "REVIEW - Clase inválida detectada"
        
        results.append(detection)
    
    # Convertir resultados a DataFrame y devolver
    return pd.DataFrame(results)

def poi_auto_corrector(poi_df, detection_df, roads_gdf, auto_fix=True):
    """
    Corrige automáticamente los problemas detectados en POIs
    
    Args:
        poi_df: DataFrame con datos de POIs originales
        detection_df: DataFrame con detecciones de problemas
        roads_gdf: GeoDataFrame con datos de calles
        auto_fix: Si True, aplica correcciones automáticas
    
    Returns:
        DataFrame con POIs corregidos
    """
    # Crear copia para no modificar los originales
    corrected_df = poi_df.copy()
    
    # Registrar cambios para reporte
    changes_log = []
    
    # Procesar cada POI con problemas detectados
    for _, detection in detection_df.iterrows():
        poi_id = detection['poi_id']
        problems = detection['problems']
        
        # Encontrar el POI en el DataFrame original
        poi_mask = corrected_df['POI_ID'] == poi_id
        if not any(poi_mask):
            continue
        
        # Registrar para eliminación (Clase 0)
        if 'nonexistent' in problems and detection['class'] == 0:
            if auto_fix:
                # Marcar para eliminación (no eliminar directamente)
                corrected_df.loc[poi_mask, 'TO_DELETE'] = True
            changes_log.append({
                'poi_id': poi_id,
                'action': 'Mark for deletion',
                'reason': 'POI does not exist',
                'confidence': detection['flags'].get('nonexistent_confidence', 0)
            })
        
        # Corregir lado de calle (Clase 1)
        if 'wrong_side' in problems and detection['class'] == 1:
            correct_side = detection['flags'].get('correct_side')
            if correct_side and auto_fix:
                old_side = corrected_df.loc[poi_mask, 'POI_ST_SD'].values[0]
                corrected_df.loc[poi_mask, 'POI_ST_SD'] = correct_side
                changes_log.append({
                    'poi_id': poi_id,
                    'action': f'Change side from {old_side} to {correct_side}',
                    'reason': 'Wrong side of street',
                    'confidence': detection['flags'].get('wrong_side_confidence', 0)
                })
        
        # Corregir MULTIDIGIT (Clase 2)
        if 'wrong_multidigit' in problems and detection['class'] == 2:
            # Necesitamos encontrar el segmento de calle correspondiente y corregirlo
            link_id = detection['link_id']
            road_mask = roads_gdf['link_id'].astype(str) == link_id
            if any(road_mask) and auto_fix:
                # No modificamos directamente roads_gdf para evitar efectos secundarios
                # En su lugar, registramos los cambios necesarios
                changes_log.append({
                    'poi_id': poi_id,
                    'action': 'Change MULTIDIGIT to N',
                    'reason': 'Incorrect MULTIDIGIT attribute',
                    'confidence': detection['flags'].get('wrong_multidigit_confidence', 0),
                    'road_id': link_id
                })
    
    # Agregar columna de estado para seguimiento
    corrected_df['CORRECTION_STATUS'] = 'Unchanged'
    for _, change in pd.DataFrame(changes_log).iterrows():
        poi_id = change['poi_id']
        poi_mask = corrected_df['POI_ID'] == poi_id
        if any(poi_mask):
            corrected_df.loc[poi_mask, 'CORRECTION_STATUS'] = change['action']
    
    return corrected_df, pd.DataFrame(changes_log)

def automated_poi_validation_pipeline(poi_files, roads_files, output_dir, auto_fix=True):
    """
    Pipeline completo de validación y corrección automática de POIs
    
    Args:
        poi_files: Lista de archivos CSV con datos de POIs
        roads_files: Lista de archivos GeoJSON con datos de calles
        output_dir: Directorio para guardar resultados
        auto_fix: Si True, aplica correcciones automáticas
        
    Returns:
        DataFrame con resumen de resultados
    """
    import os
    import pandas as pd
    import geopandas as gpd
    from datetime import datetime
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Registrar inicio
    start_time = datetime.now()
    print(f"Iniciando pipeline de validación: {start_time}")
    
    # Resultados consolidados
    all_detections = []
    all_corrections = []
    all_logs = []
    
    # Procesar archivos
    for i, poi_file in enumerate(poi_files):
        print(f"Procesando archivo {i+1}/{len(poi_files)}: {poi_file}")
        
        try:
            # Cargar datos
            poi_df = pd.read_csv(poi_file, nrows=5)
            
            # Encontrar archivo de calles correspondiente por ID
            file_id = extract_file_id(poi_file)
            matching_road_file = find_matching_road_file(file_id, roads_files)
            
            if matching_road_file:
                roads_gdf = gpd.read_file(matching_road_file)
                
                # Crear subdirectorio para este archivo
                file_output_dir = os.path.join(output_dir, f"results_{file_id}")
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Detectar problemas
                detections = poi_problem_detector(poi_df, roads_gdf)
                all_detections.append(detections)
                
                # Guardar detecciones
                detections.to_csv(f"{file_output_dir}/detections.csv", index=False)
                
                # Corregir problemas
                corrected_df, changes = poi_auto_corrector(poi_df, detections, roads_gdf, auto_fix)
                all_corrections.append(corrected_df)
                all_logs.append(changes)
                
                # Guardar correcciones
                corrected_df.to_csv(f"{file_output_dir}/corrected_pois.csv", index=False)
                changes.to_csv(f"{file_output_dir}/change_log.csv", index=False)
                
                # Generar visualizaciones
                generate_validation_visualizations(poi_df, roads_gdf, detections, file_output_dir)
                
                print(f"✓ Procesado {len(poi_df)} POIs, detectados {len(detections[detections['class'] != -1])} problemas")
            else:
                print(f"⚠ No se encontró archivo de calles correspondiente para {poi_file}")
        
        except Exception as e:
            print(f"❌ Error procesando {poi_file}: {str(e)}")
    
    # Consolidar resultados
    if all_detections:
        all_detections_df = pd.concat(all_detections, ignore_index=True)
        all_detections_df.to_csv(f"{output_dir}/all_detections.csv", index=False)
    
    if all_corrections:
        all_corrections_df = pd.concat(all_corrections, ignore_index=True)
        all_corrections_df.to_csv(f"{output_dir}/all_corrected_pois.csv", index=False)
    
    if all_logs:
        all_logs_df = pd.concat(all_logs, ignore_index=True)
        all_logs_df.to_csv(f"{output_dir}/all_changes.csv", index=False)
    
    # Generar informe consolidado
    summary = generate_summary_report(all_detections, all_logs, output_dir)
    
    # Registrar finalización
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"Pipeline completado en {duration:.2f} minutos")
    
    return summary

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

def classify_poi_rule_based(features):
    """More strict rule-based classification"""
    
    # 1. Check for non-existent roads (Class 0)
    if features['no_matching_road'] == 1 or 'link_id_exists' in features and features['link_id_exists'] == 0:
        if features['is_exception_type'] == 1 and features['nat_import'] == 1:
            return {'class_id': 3, 'confidence': 0.80, 'action': "Keep as legitimate exception"}
        return {'class_id': 0, 'confidence': 0.90, 'action': "Mark for deletion (no POI in reality)"}
    
    # 2. Force detection of wrong side (Class 1) - MORE STRICT
    road_side = features.get('road_side', -1)
    road_side_match = features.get('road_side_match', -1)
    
    # Increase strictness: assume side issues unless explicitly confirmed correct
    if road_side_match == 0 or ('calculated_side' in features and features['calculated_side'] != road_side):
        return {'class_id': 1, 'confidence': 0.80, 'action': "Fix POI side of road"}
    
    # 3. Force detection of MULTIDIGIT issues (Class 2) - MORE STRICT
    if features.get('is_multi_dig', 0) == 1:
        # More strict: flag all multi-dig unless explicitly exception type
        if features.get('is_exception_type', 0) == 0:
            return {'class_id': 2, 'confidence': 0.75, 'action': "Update MULTIDIGIT attribute to 'N'"}
    
    # 4. Check for legitimate exceptions (Class 3)
    if features.get('is_exception_type', 0) == 1:
        return {'class_id': 3, 'confidence': 0.90, 'action': "Keep as legitimate exception"}
    
    # Default case - no issues
    return {'class_id': -1, 'confidence': 0.70, 'action': "No issues detected"}
def extract_enhanced_poi_features(poi_row, roads_gdf):
    """Extract features with improved detection sensitivity"""
    try:
        features = {}
        
        # Start with basic extraction
        base_features = extract_poi_features(poi_row, roads_gdf)
        features.update(base_features)
        
        # 1. More sensitive road existence check
        link_id = poi_row.get('link_id', '')
        link_id_exists = False
        
        # Check for EXACT match (more strict)
        if 'link_id' in roads_gdf.columns and str(link_id) in roads_gdf['link_id'].astype(str).values:
            link_id_exists = True
        
        features['link_id_exists'] = 1 if link_id_exists else 0
        features['no_matching_road'] = 0 if link_id_exists else 1  # Invert for consistency
        
        # 2. More sensitive side of road detection
        poi_side = poi_row.get('POI_ST_SD', 'R')
        
        # If we have geometry, use it to compute actual side
        if link_id_exists:
            road_row = roads_gdf[roads_gdf['link_id'].astype(str) == str(link_id)].iloc[0]
            if hasattr(road_row.geometry, 'coords'):
                coords = list(road_row.geometry.coords)
                poi_coords = get_poi_coordinates(poi_row)
                
                if coords and poi_coords:
                    calculated_side = calculate_point_side(coords, poi_coords)
                    features['calculated_side'] = 1 if calculated_side == 'R' else 0
                    features['road_side'] = 1 if poi_side == 'R' else 0
                    features['road_side_match'] = 1 if calculated_side == poi_side else 0
        
        # 3. More sensitive MULTIDIGIT check
        for col_name in roads_gdf.columns:
            if 'multi' in col_name.lower() or 'digit' in col_name.lower():
                if link_id_exists:
                    road_row = roads_gdf[roads_gdf['link_id'].astype(str) == str(link_id)].iloc[0]
                    multi_val = road_row.get(col_name, 'N')
                    features['is_multi_dig'] = 1 if str(multi_val).upper() == 'Y' else 0
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Return default features
        return {
            'no_matching_road': 1,
            'dist_to_road': 999,
            'is_multi_dig': 0,
            'road_side_match': 0,
            'road_side': 1,
            'facility_type': -1,
            'is_exception_type': 0,
            'nat_import': 0,
            'in_vicinity': 0,
        }
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
    fig, ax = plt.subplots(figsize=(20, 16))
    
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

def generate_validation_visualizations(poi_df, roads_gdf, detections, output_dir):
    """Generate satellite visualizations for POIs with detected problems"""
    import os
    
    # Create directory for images
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Define API key explicitly
    api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
    
    # Visualize only POIs with problems (class != -1)
    problem_detections = detections[detections['class'] != -1]
    
    print(f"Generating satellite visualizations for {len(problem_detections)} POIs with issues...")
    
    for _, detection in problem_detections.iterrows():
        poi_id = detection['poi_id']
        class_id = detection['class']
        
        # Find the POI complete data
        poi = poi_df[poi_df['POI_ID'] == poi_id]
        if len(poi) == 0:
            continue
        poi = poi.iloc[0]
        
        # Get link_id and find corresponding road
        link_id = detection['link_id']
        road_match = roads_gdf[roads_gdf['link_id'].astype(str) == link_id]
        
        # Generate satellite visualization
        try:
            if not road_match.empty:
                road = road_match.iloc[0]
                
                if hasattr(road.geometry, 'coords'):
                    shape_points = list(road.geometry.coords)
                    
                    if len(shape_points) >= 2:
                        # Get reference nodes
                        reference_node, non_reference_node, _ = determine_reference_node(shape_points)
                        
                        # Calculate center point
                        center_lon = sum(p[0] for p in shape_points) / len(shape_points)
                        center_lat = sum(p[1] for p in shape_points) / len(shape_points)
                        
                        # Get satellite tile with markers
                        wkt_bounds, filename = get_and_mark_satellite_tile(
                            center_lat, center_lon, 17,
                            shape_points, reference_node, non_reference_node,
                            'png', api_key,
                            output_dir=images_dir,
                            identifier=f"poi_{poi_id}_class{class_id}",
                            pois=[poi]
                        )
                        
                        print(f"Generated satellite visualization for POI {poi_id}")
                        continue
            
            # If no match or road geometry issues, get a basic satellite image
            poi_coords = get_poi_coordinates(poi)
            if poi_coords:
                lat, lon = poi_coords[1], poi_coords[0]  # Flip coordinates for lat/lon order
                
                # Get satellite tile centered on POI
                wkt_bounds, filename = get_satellite_tile(
                    lat, lon, 17, 'png', api_key,
                    output_dir=images_dir,
                    identifier=f"poi_{poi_id}_class{class_id}_default"
                )
                print(f"Generated default satellite image for POI {poi_id}")
                
        except Exception as e:
            print(f"Error generating satellite visualization for POI {poi_id}: {str(e)}")
def generate_summary_report(all_detections, all_changes, output_dir):
    """Genera informe HTML con resumen de la validación y correcciones"""
    import pandas as pd
    import os
    
    # Consolidar detecciones
    if all_detections:
        detections = pd.concat(all_detections, ignore_index=True)
    else:
        detections = pd.DataFrame()
    
    # Consolidar cambios
    if all_changes:
        changes = pd.concat(all_changes, ignore_index=True)
    else:
        changes = pd.DataFrame()
    
    # Estadísticas de detección
    class_counts = {
        0: len(detections[detections['class'] == 0]),
        1: len(detections[detections['class'] == 1]),
        2: len(detections[detections['class'] == 2]),
        3: len(detections[detections['class'] == 3]),
        -1: len(detections[detections['class'] == -1])
    }
    
    # Estadísticas de cambios
    total_changes = len(changes)
    
    # Crear HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>POI Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .class-0 {{ background-color: #ffcccc; }} /* Red for deletion */
            .class-1 {{ background-color: #ffffcc; }} /* Yellow for wrong side */
            .class-2 {{ background-color: #ccffff; }} /* Blue for incorrect MULTIDIGIT */
            .class-3 {{ background-color: #ccffcc; }} /* Green for legitimate exception */
            .chart {{ height: 300px; }}
        </style>
    </head>
    <body>
        <h1>POI Validation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total POIs analyzed: {len(detections)}</p>
            <p>POIs with issues: {len(detections) - class_counts[-1]}</p>
            <p>Changes applied: {total_changes}</p>
            <ul>
                <li>Non-existent POIs (to delete): {class_counts[0]}</li>
                <li>Wrong Side of Road: {class_counts[1]}</li>
                <li>Incorrect MULTIDIGIT: {class_counts[2]}</li>
                <li>Legitimate Exceptions: {class_counts[3]}</li>
                <li>No issues detected: {class_counts[-1]}</li>
            </ul>
        </div>
        
        <h2>Distribution of Issues</h2>
        <div class="chart">
            <!-- Insertar gráfico aquí si es posible -->
        </div>
        
        <h2>Top POIs with Issues</h2>
        <table>
            <tr>
                <th>POI ID</th>
                <th>Name</th>
                <th>Issue Type</th>
                <th>Action</th>
            </tr>
    """
    
    # Agregar filas para los primeros 100 POIs con problemas
    problem_pois = detections[detections['class'] != -1].head(100)
    for _, poi in problem_pois.iterrows():
        class_id = poi['class']
        html += f"""
            <tr class="class-{class_id}">
                <td>{poi['poi_id']}</td>
                <td>{poi['poi_name']}</td>
                <td>{['Non-existent POI', 'Wrong Side of Road', 'Incorrect MULTIDIGIT', 'Legitimate Exception'][class_id]}</td>
                <td>{poi['action']}</td>
            </tr>
        """
    
    # Cerrar tabla y documento
    html += """
        </table>
    </body>
    </html>
    """
    
    # Guardar HTML
    report_path = os.path.join(output_dir, 'validation_report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated at {report_path}")
    
    # Devolver resumen
    summary = {
        'total_pois': len(detections),
        'total_issues': len(detections) - class_counts[-1],
        'class_distribution': class_counts,
        'total_changes': total_changes
    }
    
    return summary

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
        pois = pd.read_csv(poi_file, header=0, nrows=5)
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

def generate_balanced_poi_dataset(roads_file, output_file="balanced_training_data.csv", num_per_class=250):
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
        if pd.isna(vis_path) or not os.path.exists(vis_path):
            possible_paths = [
                os.path.join(output_dir, 'images', f"poi_{row['poi_id']}.png"),
                os.path.join(output_dir, 'images', f"poi_{row['poi_id']}_class{row['predicted_label']}.png"),
                os.path.join(output_dir, 'images', f"satellite_poi_{row['poi_id']}.png")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    vis_path = path
                    break
        vis_html = ""
        if pd.notna(vis_path) and os.path.exists(vis_path):
            # Convert to relative path
            rel_path = os.path.relpath(vis_path, output_dir)
            vis_html = f'<img src="{rel_path}" class="visualization" alt="POI Satellite Image">'
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

def is_poi_on_median(poi_geometry, road_geometry):
    """Detecta si un POI está ubicado en un camellón o separador central"""
    if not hasattr(road_geometry, 'coords'):
        return False
        
    # Obtener coordenadas
    road_coords = list(road_geometry.coords)
    if len(road_coords) < 2:
        return False
        
    # Encontrar segmento más cercano
    min_dist = float('inf')
    closest_segment = None
    
    for i in range(len(road_coords)-1):
        segment = LineString([road_coords[i], road_coords[i+1]])
        dist = segment.distance(poi_geometry)
        
        if dist < min_dist:
            min_dist = dist
            closest_segment = segment
    
    if closest_segment is None:
        return False
        
    # Verificar si la distancia es muy pequeña (probablemente en camellón)
    # La distancia se mide en grados, aproximadamente 0.0001° ≈ 10m
    return min_dist < 0.0001

def verify_poi_side(poi_row, road_geometry):
    """Verifica si el POI está en el lado declarado de la calle"""
    declared_side = poi_row.get('POI_ST_SD', 'R')
    
    poi_coords = get_poi_coordinates(poi_row)
    if not poi_coords:
        return True  # No podemos verificar
        
    poi_point = Point(poi_coords)
    actual_side = calculate_point_side(road_geometry, poi_coords)
    
    return declared_side == actual_side

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

def safe_extract_coordinates(geometry):
    """
    Safely extract coordinates from geometries, handling both simple and multi-part geometries.
    
    Args:
        geometry: A Shapely geometry object
        
    Returns:
        list: A list of coordinate tuples
    """
    from shapely.geometry import Point, LineString, MultiPoint, MultiLineString, Polygon, MultiPolygon
    
    # Handle None case
    if geometry is None:
        return []
    
    # For Point type
    if isinstance(geometry, Point):
        return [(geometry.x, geometry.y)]
    
    # For LineString and similar types with coords attribute
    if isinstance(geometry, LineString) or hasattr(geometry, 'coords'):
        try:
            return list(geometry.coords)
        except Exception as e:
            print(f"Error extracting coords: {e}")
            return []
    
    # For Polygon
    if isinstance(geometry, Polygon):
        try:
            return list(geometry.exterior.coords)
        except Exception as e:
            print(f"Error extracting polygon coords: {e}")
            return []
    
    # For multi-part geometries
    if hasattr(geometry, 'geoms'):
        try:
            all_coords = []
            for geom in geometry.geoms:
                # Recursively extract coordinates from each sub-geometry
                all_coords.extend(safe_extract_coordinates(geom))
            return all_coords
        except Exception as e:
            print(f"Error processing multi-geometry: {e}")
            return []
    
    print(f"Warning: Unsupported geometry type: {type(geometry)}")
    return []
# Función auxiliar para extraer coordenadas del POI
def get_poi_coordinates(poi_row):
    """Extrae coordenadas del POI desde diferentes formatos posibles"""
    if 'geometry' in poi_row:
        # Usar nuestra función segura si hay geometría
        coords = safe_extract_coordinates(poi_row['geometry'])
        if coords:
            return [coords[0][0], coords[0][1]]
    
    # Resto de los métodos para extraer coordenadas cuando no hay geometría
    for x_col, y_col in [('x', 'y'), ('X', 'Y'), ('lon', 'lat'), ('LON', 'LAT')]:
        if x_col in poi_row and y_col in poi_row:
            return [float(poi_row[x_col]), float(poi_row[y_col])]
    
    return None

# Función para calcular de qué lado de la línea está un punto
def calculate_point_side(geometry, point):
    """Determina de qué lado de una geometría está un punto (L/R)"""
    # Extraer coordenadas de manera segura
    coords = safe_extract_coordinates(geometry)
    
    if len(coords) < 2:
        return 'unknown'
    
    # Usar el primer segmento para determinar dirección
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    
    # Calcular producto cruz para determinar lado
    cross_product = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
    
    # Positive = izquierda (L), Negative = derecha (R)
    return 'L' if cross_product > 0 else 'R'

def validate_pois(poi_file, roads_file, model_file, output_dir="validation_results", batch_size=10, resume_from=0):
    """Validate POIs using the trained model with proper feature matching"""
    print(f"Loading model from {model_file}")
    model = joblib.load(model_file)
    
    print(f"Loading POI data from {poi_file}")
    poi_data = pd.read_csv(poi_file, nrows=5)
    
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

def validate_pois_strict(poi_file, roads_file, output_dir="strict_validation"):
    """Validate POIs using stricter rule-based approach without ML model"""
    pois = pd.read_csv(poi_file)
    roads = gpd.read_file(roads_file)
    
    results = []
    
    for idx, poi in pois.iterrows():
        # Extract link_id
        link_id = poi.get('link_id', '')
        
        # RULE 1: Check if road exists (CLASS 0)
        road_exists = str(link_id) in roads['link_id'].astype(str).values
        if not road_exists:
            class_id = 0
            action = "Mark for deletion (no POI in reality)"
            
        # RULE 2: Check side of road (CLASS 1)
        elif hasattr(poi, 'POI_ST_SD'):
            # This is a simplified rule - in reality you would check geometry
            poi_side = poi['POI_ST_SD']
            # Assume 30% of POIs have wrong side for demonstration
            if random.random() < 0.3:  # Replace with actual geometric calculation
                class_id = 1
                action = f"Fix POI side of road (from {poi_side} to {'L' if poi_side=='R' else 'R'})"
            
        # RULE 3: Check MULTIDIGIT (CLASS 2)
        # Assume roads with certain IDs should not be MULTIDIGIT
        elif road_exists and int(link_id) % 10 == 0:  # Simplified rule
            class_id = 2
            action = "Update MULTIDIGIT attribute to 'N'"
            
        # Default: No issues
        else:
            class_id = -1
            action = "No issues detected"
            
        results.append({
            'poi_id': poi.get('POI_ID', idx),
            'link_id': link_id,
            'class_id': class_id,
            'action': action
        })
    
    # Save results
    pd.DataFrame(results).to_csv(f"{output_dir}/strict_results.csv", index=False)
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

if __name__ == "__main__":
    import glob
    import os
    import pandas as pd
    import argparse
    import sys

    # First option: completely clean arguments in Jupyter/Colab environments
    if 'ipykernel' in sys.modules or any('jupyter' in arg for arg in sys.argv):
        sys.argv = [sys.argv[0]]  # Keep only the script name

    # Create parser and define ALL arguments BEFORE parsing
    parser = argparse.ArgumentParser(description='POI Validation and Auto-correction System')
    parser.add_argument('--no-auto-fix', action='store_true', help='Disable automatic corrections')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training and use existing model')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    
    # Now analyze the arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

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

    # Full pipeline execution wrapped in try-except
    try:
        # Only train the model if not skipping training
        if not args.skip_training:
            # Training code here...
            pass

        # Now validate the real POI data using our automated pipeline
        print("\n=== Running Automated POI Validation and Correction ===")
        
        # Prepare files for automated validation
        active_poi_files = []
        active_road_files = []
        
        for file_id in common_ids:
            poi_file = poi_map[file_id]
            # Prefer NAMING_ADDRESSING over NAV if both exist
            road_file = road_map.get(file_id, nav_map.get(file_id))
            
            active_poi_files.append(poi_file)
            active_road_files.append(road_file)
        
        # Run the automated pipeline with all files
        output_dir = "automated_validation_results"
        
        # Call our automated pipeline
        summary = automated_poi_validation_pipeline(
            poi_files=active_poi_files,
            roads_files=active_road_files,
            output_dir=output_dir,
            auto_fix=not args.no_auto_fix
        )
        
        # Display summary results
        print("\n=== AUTOMATED VALIDATION SUMMARY ===")
        print(f"Total POIs processed: {summary['total_pois']}")
        print(f"POIs with issues detected: {summary['total_issues']}")
        print(f"Changes applied: {summary['total_changes']}")
        
        # Print class distribution
        class_names = {
            0: "Non-existent POIs (to delete)",
            1: "Wrong Side of Road",
            2: "Incorrect MULTIDIGIT", 
            3: "Legitimate Exceptions",
            -1: "No issues detected"
        }
        
        for class_id, count in summary['class_distribution'].items():
            if class_id in class_names:
                percent = (count / summary['total_pois'] * 100) if summary['total_pois'] > 0 else 0
                print(f"{class_names[class_id]}: {count} ({percent:.1f}%)")
        
        print(f"\nDetailed report available at: {output_dir}/validation_report.html")
        
    except Exception as e:
        print(f"Error in automated validation pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fall back to original validation as backup
        print("\nFalling back to basic validation...")
        
        # Create a summary dataframe to track performance across all datasets
        overall_results = pd.DataFrame(columns=['file_id', 'poi_count', 'class_0_count', 
                                              'class_1_count', 'class_2_count', 'class_3_count'])
        
        # Process each pair of files with the original validate_pois function
        for file_id in common_ids:
            try:
                print(f"\n===== Processing POI and road files with ID: {file_id} =====")
                
                poi_file = poi_map[file_id]
                road_file = road_map.get(file_id, nav_map.get(file_id))
                
                output_dir = f"validation_results_{file_id}"
                
                # Validate POIs with original function
                print(f"Validating POIs from {poi_file} against roads from {road_file}")
                
                results = validate_pois(
                    poi_file=poi_file,
                    roads_file=road_file,
                    model_file="poi_validation_model.joblib",
                    output_dir=output_dir,
                    batch_size=args.batch_size
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