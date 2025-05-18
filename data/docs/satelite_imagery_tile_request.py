import requests
import math
import json
import os
import pandas as pd
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import io

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
    lat_def = math.degrees(lat_rad)
    return (lat_def, lon_deg)

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

def extract_id_from_filename(filename):
    """Extract the numeric ID from a filename."""
    match = re.search(r'_(\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def find_files_by_id(id_value):
    """Find all related files with the given ID."""
    files = {
        'poi': glob.glob(f"**/POI_{id_value}*.csv", recursive=True),
        'streets_naming': glob.glob(f"**/SREETS_NAMING_ADDRESSING_{id_value}*.geojson", recursive=True),
        'streets_nav': glob.glob(f"**/SREETS_NAV_{id_value}*.geojson", recursive=True)
    }
    return files

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

def get_geometry_center_and_zoom(shape_points, padding_percent=20):
    """
    Calculate the center point of a geometry and determine appropriate zoom level.
    Includes padding to ensure all points are visible.
    Always returns zoom level 17 as requested.
    
    Args:
        shape_points: List of [lon, lat] coordinates
        padding_percent: Additional padding as percentage of the extent
        
    Returns:
        center_lon, center_lat: The center coordinates
        recommended_zoom: Always 17
    """
    if not shape_points:
        return 0, 0, 17  # Default values if no points with fixed zoom 17
        
    # Extract lon and lat lists
    lons = [point[0] for point in shape_points]
    lats = [point[1] for point in shape_points]
    
    # Calculate the geometry extent
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Apply padding
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    # Add padding (more padding for smaller geometries)
    padding_factor = padding_percent / 100.0
    min_lon -= lon_span * padding_factor
    max_lon += lon_span * padding_factor
    min_lat -= lat_span * padding_factor
    max_lat += lat_span * padding_factor
    
    # Recalculate spans with padding
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    # Calculate the center of the bounding box
    center_lon = min_lon + lon_span / 2
    center_lat = min_lat + lat_span / 2
    
    # Always use zoom level 17 as requested
    recommended_zoom = 17
    
    print(f"Geometry extent: Lon [{min_lon:.6f}, {max_lon:.6f}], Lat [{min_lat:.6f}, {max_lat:.6f}]")
    print(f"Calculated center: {center_lat:.6f}, {center_lon:.6f} with zoom {recommended_zoom}")
    
    return center_lon, center_lat, recommended_zoom

def get_satellite_tile(lat, lon, zoom, tile_format, api_key, output_dir="tiles", identifier=None):
    """Get a satellite tile without marking it (used for POIs)."""
    x, y = lat_lon_to_tile(lat, lon, zoom)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Construct the URL for the map tile API
    url = f'https://maps.hereapi.com/v3/base/mc/{zoom}/{x}/{y}/{tile_format}?style=satellite.day&size={tile_size}&apiKey={api_key}'
    
    # Define filename
    if identifier:
        filename = f'{output_dir}/tile_{identifier}.{tile_format}'
    else:
        filename = f'{output_dir}/tile_{lat:.5f}_{lon:.5f}_z{zoom}.{tile_format}'
    
    # Make the request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the tile bounds
        bounds = get_tile_bounds(x, y, zoom)
        wkt_polygon = create_wkt_polygon(bounds)
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Tile saved to {filename}")
        
        return wkt_polygon, filename
    else:
        print(f'Failed to retrieve tile for {lat}, {lon}. Status code: {response.status_code}')
        return None, None

def process_area(area_id, api_key, zoom_level=16, tile_format='png', max_tiles=10, adaptive_zoom=True):
    """Process a complete area using all three data sources."""
    print(f"\n==== Processing Area {area_id} ====")

    # Find all related files
    files = find_files_by_id(area_id)

    # Create output directory
    output_dir = f"tiles_area_{area_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    area_data = {
        'poi': None,
        'streets_naming': None,
        'streets_nav': None
    }

    # Load POI data if available
    if files['poi']:
        poi_file = files['poi'][0]
        try:
            area_data['poi'] = pd.read_csv(poi_file)
            print(f"Loaded {len(area_data['poi'])} POIs from {poi_file}")
            
            # Display POI column names to verify data structure
            print(f"POI columns: {area_data['poi'].columns.tolist()}")
            
            # Check for required columns (case-insensitive)
            poi_df = area_data['poi']
            link_id_col = find_column(poi_df, r'LINK_ID')
            perc_col = find_column(poi_df, r'PERC[FR]+REF')
            side_col = find_column(poi_df, r'POI.*S[TD].*S[TD]')
            
            print(f"Found POI key columns: LINK_ID={link_id_col}, PERCFFREF={perc_col}, POI_ST_SD={side_col}")
            
            if not all([link_id_col, perc_col, side_col]):
                print("WARNING: Missing required POI columns. POIs may not be properly visualized.")
        except Exception as e:
            print(f"Error loading POI file {poi_file}: {str(e)}")
    else:
        print("No POI file found for this area")

    # Load STREETS_NAMING_ADDRESSING data if available
    if files['streets_naming']:
        streets_naming_file = files['streets_naming'][0]
        try:
            with open(streets_naming_file, 'r') as f:
                area_data['streets_naming'] = json.load(f)
            print(f"Loaded {len(area_data['streets_naming']['features'])} street naming records from {streets_naming_file}")
        except Exception as e:
            print(f"Error loading streets naming file {streets_naming_file}: {str(e)}")
    else:
        print("No STREETS_NAMING_ADDRESSING file found for this area")

    # Load STREETS_NAV data if available
    if files['streets_nav']:
        streets_nav_file = files['streets_nav'][0]
        try:
            with open(streets_nav_file, 'r') as f:
                area_data['streets_nav'] = json.load(f)
            print(f"Loaded {len(area_data['streets_nav']['features'])} street navigation records from {streets_nav_file}")
        except Exception as e:
            print(f"Error loading streets nav file {streets_nav_file}: {str(e)}")
    else:
        print("No STREETS_NAV file found for this area")

    # Process data and fetch tiles
    results = []
    count = 0
    
    # Create link_id to shape_points mapping for POIs
    link_map = {}
    links_with_pois = set()

    # First, identify which links have POIs
    if area_data['poi'] is not None:
        poi_df = area_data['poi']
        link_id_col = find_column(poi_df, r'LINK_ID')
        if link_id_col:
            # Get all unique link IDs that have POIs
            links_with_pois = set(poi_df[link_id_col].astype(str).unique())
            print(f"Found {len(links_with_pois)} unique links that have POIs")
    
    # Then process streets from STREETS_NAV and build link mapping, prioritizing those with POIs
    if area_data['streets_nav']:
        # First pass: Process links that have POIs
        for feature in area_data['streets_nav']['features']:
            if count >= max_tiles:
                break  # Stop if we've reached max_tiles
                
            if feature['geometry']['type'] == 'LineString' and len(feature['geometry']['coordinates']) > 0:
                # Get link_id
                link_id = str(feature['properties'].get('link_id', None))
                
                # Skip if this link doesn't have POIs
                if link_id not in links_with_pois:
                    continue
                    
                # Store shape points and reference node info
                shape_points = feature['geometry']['coordinates']
                reference_node, non_reference_node, is_reference_at_start = determine_reference_node(shape_points)
                
                link_map[link_id] = {
                    'shape_points': shape_points,
                    'reference_node': reference_node, 
                    'non_reference_node': non_reference_node,
                    'is_reference_at_start': is_reference_at_start
                }
                
                # Calculate the center of the road geometry and zoom level
                center_longitude, center_latitude, adjusted_zoom = get_geometry_center_and_zoom(shape_points)
                actual_zoom = adjusted_zoom if adaptive_zoom else zoom_level
                print(f"  - Using zoom level: {actual_zoom} (center: {center_latitude}, {center_longitude})")

                # Find POIs associated with this link
                link_pois = []
                poi_df = area_data['poi']
                link_id_col = find_column(poi_df, r'LINK_ID')
                perc_col = find_column(poi_df, r'PERC[FR]+REF')
                side_col = find_column(poi_df, r'POI.*S[TD].*S[TD]')
                
                if all([link_id_col, perc_col, side_col]):
                    for idx, poi in poi_df.iterrows():
                        if str(poi.get(link_id_col)) == link_id:
                            # Create POI dictionary with actual data from CSV
                            poi_dict = {
                                'POI_ID': poi.get('POI_ID', idx),
                                'POI_NAME': poi.get('POI_NAME', f"POI_{idx}"),
                                'LINK_ID': link_id,
                                'PERCFRREF': float(poi.get(perc_col, 0.5)),
                                'POI_ST_SD': poi.get(side_col, 'R')
                            }
                            link_pois.append(poi_dict)
                            print(f"  - Found POI {poi_dict['POI_ID']} on link {link_id} at {poi_dict['PERCFRREF']} from ref node, side {poi_dict['POI_ST_SD']}")
                
                # Only process if we found POIs
                if link_pois:
                    # Additional properties
                    properties = {
                        'link_id': link_id,
                        'source': 'streets_nav',
                        'func_class': feature['properties'].get('FUNC_CLASS', ''),
                        'speed_cat': feature['properties'].get('SPEED_CAT', ''),
                        'dir_travel': feature['properties'].get('DIR_TRAVEL', ''),
                        'shape_point_count': len(shape_points),
                        'reference_node_position': 'start' if is_reference_at_start else 'end',
                        'reference_node': reference_node,
                        'non_reference_node': non_reference_node,
                        'poi_count': len(link_pois)
                    }

                    print(f"Processing street navigation segment {link_id} with {len(link_pois)} POIs")
                    print(f"  - Shape points: {len(shape_points)}")
                    print(f"  - Reference node: {reference_node} ({'start' if is_reference_at_start else 'end'})")
                    print(f"  - Non-reference node: {non_reference_node}")

                    # Get and mark satellite tile with POIs
                    wkt_bounds, filename = get_and_mark_satellite_tile(
                        center_latitude, center_longitude,
                        actual_zoom,
                        shape_points,
                        reference_node,
                        non_reference_node,
                        tile_format,
                        api_key,
                        output_dir=output_dir,
                        identifier=f"nav_{link_id}",
                        pois=link_pois  # Pass POIs to be drawn
                    )

                    # Store the results
                    if wkt_bounds and filename:
                        result = {
                            'link_id': link_id,
                            'shape_points': shape_points,
                            'reference_node': reference_node,
                            'non_reference_node': non_reference_node,
                            'is_reference_at_start': is_reference_at_start,
                            'wkt_bounds': wkt_bounds,
                            'filename': filename,
                            'properties': properties,
                            'pois': link_pois
                        }
                        results.append(result)
                        count += 1
        
        # If we still have room for more tiles, process links without POIs
        if count < max_tiles and len(area_data['streets_nav']['features']) > count:
            print(f"Processed {count} links with POIs. Not generating any test POIs.")

    # Process standalone POIs if available and we're under max_tiles limit
    if area_data['poi'] is not None and count < max_tiles:
        # Check if POI data has coordinates
        coord_columns = []
        for col in area_data['poi'].columns:
            if 'LAT' in col.upper() or 'LATITUDE' in col.upper():
                coord_columns.append(('lat', col))
            elif 'LON' in col.upper() or 'LONGITUDE' in col.upper():
                coord_columns.append(('lon', col))

        if len(coord_columns) >= 2:
            # Find one pair of lat/lon columns
            lat_col = next((col[1] for col in coord_columns if col[0] == 'lat'), None)
            lon_col = next((col[1] for col in coord_columns if col[0] == 'lon'), None)

            if lat_col and lon_col:
                # Process POIs that aren't tied to links
                standalone_pois = []
                
                for idx, row in area_data['poi'].iterrows():
                    # Skip POIs already associated with links if LINK_ID is present
                    link_id_col = find_column(area_data['poi'], r'LINK_ID')
                    if link_id_col and pd.notna(row[link_id_col]) and str(row[link_id_col]) in link_map:
                        continue
                        
                    try:
                        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                            latitude = float(row[lat_col])
                            longitude = float(row[lon_col])
                            poi_id = row.get('POI_ID', idx)
                            poi_name = row.get('POI_NAME', f"POI_{poi_id}")

                            standalone_pois.append({
                                'poi_id': poi_id,
                                'poi_name': poi_name,
                                'latitude': latitude,
                                'longitude': longitude
                            })
                    except (ValueError, TypeError) as e:
                        print(f"Error processing POI {idx}: {str(e)}")
                
                # If we have standalone POIs, process them
                if standalone_pois and count < max_tiles:
                    for poi in standalone_pois[:max_tiles - count]:
                        print(f"Processing standalone POI {poi['poi_id']} ({poi['poi_name']}) at {poi['latitude']}, {poi['longitude']}")
                        
                        # For standalone POIs, we just use original function
                        wkt_bounds, filename = get_satellite_tile(
                            poi['latitude'], poi['longitude'], zoom_level, tile_format, api_key,
                            output_dir=output_dir,
                            identifier=f"poi_{poi['poi_id']}"
                        )

                        # Store the results
                        result = {
                            'poi_id': poi['poi_id'],
                            'poi_name': poi['poi_name'],
                            'latitude': poi['latitude'],
                            'longitude': poi['longitude'],
                            'wkt_bounds': wkt_bounds,
                            'filename': filename,
                            'properties': {
                                'source': 'poi',
                                'poi_name': poi['poi_name']
                            }
                        }
                        results.append(result)
                        count += 1

    # Save metadata about all tiles
    metadata_file = f"{output_dir}/metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} tiles for area {area_id}. Metadata saved to {metadata_file}")
    return results

# Define global parameters
api_key = 'soLfEc1KZmaeWBIBSdfhCEytfR6S6qnLsP45R8JWoYA'
zoom_level = 17  # Default zoom level if not using adaptive zoom
tile_size = 512  # Tile size in pixels
tile_format = 'png'  # Tile format
adaptive_zoom = True  # Set to True to automatically adjust zoom based on road length

# Find all unique area IDs by searching for STREETS_NAV files first
nav_files = glob.glob("**/SREETS_NAV_*.geojson", recursive=True)
area_ids = []

for file_path in nav_files:
    area_id = extract_id_from_filename(file_path)
    if area_id and area_id not in area_ids:
        area_ids.append(area_id)

if not area_ids:
    print("No area files found. Please check file naming patterns.")
else:
    print(f"Found {len(area_ids)} areas to process: {', '.join(area_ids)}")

    # Process each area
    for area_id in area_ids:
        process_area(
            area_id,
            api_key,
            zoom_level=zoom_level,
            tile_format=tile_format,
            max_tiles=5,  # Increased from 1 to 5 to process more links
            adaptive_zoom=adaptive_zoom
        )