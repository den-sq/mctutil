
'''
### Implementing the Changes

I've updated the script below to incorporate all these features. Here’s a summary of the changes:

1.  **GUI:** Added two new checkboxes to the "Synthetic Segmentation" panel:
    *   `[x] Treat Voxel Size as Multiplier`
    *   `[x] Use [0,0,0] as Segmentation Origin`
2.  **Logic:**
    *   The `run_processing` function now checks for the multiplier setting. If active, it calculates the new voxel size based on the input JSON's resolution.
    *   The `generate_segmentation_slices` function now accepts a `use_zero_origin` flag.
    *   The `compute_global_bounding_box` function was updated to handle the `use_zero_origin` flag by forcing the minimum coordinates to `[0, 0, 0]`.
    *   The segmentation output folder name is now more descriptive, including the volume size in voxels (`..._size_WxHxD_...`).

Here is the fully updated script. You can replace your existing file with this one.

--- START OF UPDATED FILE NG_axon_synth_gen_gui_v4.3.py ---
'''

# --- IMPORTS ---
import json
import uuid
import numpy as np
import re
import os
import sys
import gzip
import struct
import math
from scipy.interpolate import splprep, splev
from concurrent.futures import ProcessPoolExecutor, as_completed
from tifffile import imwrite
# import tkinter as tk  # <-- Remove direct tk import for widgets
# from tkinter import ttk, filedialog, messagebox # <-- Remove ttk import
import tkinter as tk # Keep for variables
from tkinter import filedialog, messagebox # Keep standard dialogs
import customtkinter # <--- ADD CUSTOMTKINTER IMPORT
import networkx as nx
from collections import defaultdict
import json, os, urllib.request, urllib.parse


# --- ALL YOUR UTILITY, PROCESSING, and HELPER FUNCTIONS GO HERE ---
# (find_segments_in_graph, process_segment_group, process_state_grouped,
# remove_duplicate_points, parse_radius_from_layer_name, ... etc ...
# compute_tangents, get_frenet_frame, generate_offset_splines_*,
# create_annotations_from_curve, create_skeleton_info_json, write_skeleton_file,
# curves_to_vertices_edges, compute_global_bounding_box, draw_disk,
# process_slice_and_save_2, generate_segmentation_slices, process_annotation_layer,
# process_state, save_new_json, export_splines_for_threejs)
# --- PASTE ALL YOUR EXISTING FUNCTIONS HERE ---
# --- UTILITY FUNCTIONS (PARSING, GEOMETRY, ETC.) ---


def save_new_json(state, original_path, suffix):
    """ Save the modified state to a new JSON file. """
    if not original_path: return None # Cannot save without original path

    base, ext = os.path.splitext(original_path)
    # Ensure suffix is reasonable
    if not isinstance(suffix, str) or not suffix:
         suffix = "_processed" # Default suffix
    if not suffix.startswith(('_', '.')):
         suffix = "_" + suffix
    # Ensure suffix ends with .json only once
    if suffix.lower().endswith('.json'):
        suffix = suffix[:-5] # Remove existing .json
    new_path = base + suffix + ".json" # Add .json back

    try:
        with open(new_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Saved updated JSON state to: {new_path}")
        return new_path
    except Exception as e:
        print(f"Error saving JSON to {new_path}: {e}")
        return None

def export_splines_for_threejs(spline_collection_processed, output_path):
    """ Exports processed splines (using structure with 'curves' list of (points, radii)) to JSON for three.js viewer. """
    if not spline_collection_processed:
        print("No processed splines to export for visualization.")
        return

    export_data = {"segments": []}
    # spline_collection_processed is list of dicts: {'segment_id': id, 'curves': [(points, radii), ...]}
    for segment_data in spline_collection_processed:
        segment_id = segment_data.get('segment_id', 'unknown')
        curves_data = segment_data.get('curves', []) # List of (points, radii)

        curves_points_only = []
        for curve_entry in curves_data:
            # Expecting (points, radii) tuple
            if isinstance(curve_entry, (list, tuple)) and len(curve_entry) == 2:
                points = curve_entry[0]
                # Check if points is a list and not empty
                if isinstance(points, list) and points:
                    # Ensure points are lists of floats for JSON compatibility
                    points_list = [[float(c) for c in p] for p in points]
                    curves_points_only.append(points_list)
            # else: # Less verbose
                # print(f"Warning: Skipping unexpected curve format during viz export for segment {segment_id}.")


        if curves_points_only: # Only add segment if it has valid curves
            export_data["segments"].append({
                "id": segment_id,
                "curves": curves_points_only # Store only point coordinates
            })

    # Write the export data to JSON file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported spline data for visualization to: {output_path}")
    except Exception as e:
        print(f"Error exporting visualization data to {output_path}: {e}")


# --- TKINTER GUI APPLICATION ---
def find_segments_in_graph(graph):
    """
    Identifies paths (segments) in the graph between nodes of degree != 2.

    Args:
        graph (nx.Graph): A graph where nodes are point indices and edges represent connectivity.

    Returns:
        list: A list of segments, where each segment is a list of node indices in order.
    """
    segments = []
    # Find endpoints (degree 1) and branch points (degree > 2)
    special_nodes = [n for n, deg in graph.degree() if deg != 2]
    if not special_nodes: # Handle simple loops or single straight paths
        if graph.number_of_nodes() > 1:
             # Attempt to find a path if it's just a line
             try:
                 start_node = list(graph.nodes())[0]
                 # Find the longest simple path starting from an arbitrary node
                 # This is heuristic for unbranched structures
                 longest_path = []
                 for end_node in graph.nodes():
                      # Find all simple paths - potentially slow for large graphs
                      # Limit path finding if graph is complex?
                      # For simple loops/lines, this should be manageable.
                      paths = list(nx.all_simple_paths(graph, source=start_node, target=end_node))
                      if paths:
                          current_longest = max(paths, key=len)
                          if len(current_longest) > len(longest_path):
                              longest_path = current_longest
                 if longest_path:
                      return [longest_path]
                 else: # Could be disconnected nodes?
                      return [[n] for n in graph.nodes()] # Return individual nodes
             except Exception as e:
                 print(f"Warning: Could not find path in graph with no special nodes: {e}")
                 return [[n] for n in graph.nodes()] # Return individual nodes
        elif graph.number_of_nodes() == 1:
            return [list(graph.nodes())] # Single node segment
        else:
            return [] # Empty graph


    visited_edges = set()

    # Start traversal from endpoints and branch points
    for start_node in special_nodes:
        for neighbor in graph.neighbors(start_node):
            edge = tuple(sorted((start_node, neighbor)))
            if edge in visited_edges:
                continue

            # Start a new segment traversal
            current_path = [start_node]
            prev_node = start_node
            curr_node = neighbor

            while curr_node not in special_nodes:
                current_path.append(curr_node)
                edge = tuple(sorted((prev_node, curr_node)))
                visited_edges.add(edge)

                # Find the next node that isn't the previous one
                next_neighbors = list(graph.neighbors(curr_node))
                if len(next_neighbors) == 1: # Should not happen if not start/end
                    break # Reached end prematurely? Graph error?
                # Should have degree 2 here
                next_node = next_neighbors[0] if next_neighbors[1] == prev_node else next_neighbors[1]
                prev_node = curr_node
                curr_node = next_node

            # Reached another special node (or the start node in a loop segment)
            current_path.append(curr_node)
            edge = tuple(sorted((prev_node, curr_node)))
            visited_edges.add(edge)
            segments.append(current_path)

    # Check if all edges were visited (handles disconnected graphs or loops not connected to special nodes)
    num_graph_edges = graph.number_of_edges()
    if len(visited_edges) < num_graph_edges:
         print(f"Warning: Visited {len(visited_edges)} edges, but graph has {num_graph_edges}. May have missed segments (e.g., isolated loops).")
         # Try to find remaining components/edges - complex, skip for now

    # Filter out potential duplicate paths found by starting at both ends
    unique_segments_by_endpoints = {}
    for seg in segments:
         endpoints = tuple(sorted((seg[0], seg[-1])))
         # If multiple paths between same endpoints, keep longest? Or first? Keep first for now.
         # Check if reverse path already stored
         reverse_endpoints = tuple(sorted((seg[-1], seg[0])))
         if endpoints not in unique_segments_by_endpoints and reverse_endpoints not in unique_segments_by_endpoints:
             unique_segments_by_endpoints[endpoints] = seg
         elif endpoints in unique_segments_by_endpoints:
              # If this path is longer, replace the existing one
              if len(seg) > len(unique_segments_by_endpoints[endpoints]):
                  unique_segments_by_endpoints[endpoints] = seg
         elif reverse_endpoints in unique_segments_by_endpoints:
              # If this path is longer, replace the existing one (checking reverse endpoints)
              if len(seg) > len(unique_segments_by_endpoints[reverse_endpoints]):
                   # Replace potentially reversed segment with this potentially longer one
                   del unique_segments_by_endpoints[reverse_endpoints]
                   unique_segments_by_endpoints[endpoints] = seg


    final_segments = list(unique_segments_by_endpoints.values())
    print(f"  Identified {len(final_segments)} unique segments.")
    return final_segments


def process_segment_group(segment_id, layers_in_group, params):
    """
    Processes a group of layers sharing the same segment_id to handle branching.
    """
    print(f"\nProcessing Segment Group ID: {segment_id} ({len(layers_in_group)} layers)")

    all_points_coords = []
    original_segments = [] # Store pairs of coords defining connections
    effective_radius = params['radius_nm'] # Start with default global radius (from GUI offset params)
    found_radius = False

    # 1. Collect all points and original connectivity
    for layer_idx, layer in enumerate(layers_in_group):
        layer_name = layer.get("name", "")
        # Determine radius for the group (e.g., from first layer with 'r' tag, or any layer)
        # Prioritize radius from any layer in the group, not just the first.
        # If multiple layers have r<num>, the last one processed might win, or the first one found.
        # Let's make it so the first one found sets it for the group.
        if not found_radius:
             r = parse_radius_from_layer_name(layer_name, -1) # Use -1 to signal not found
             if r != -1:
                  effective_radius = r * params["src_voxel_nm"][0]   # assumes isotropic source voxels
                  found_radius = True
                  print(f"  Using radius {effective_radius} from layer '{layer_name}' for group {segment_id}.")

        vox_nm = params['src_voxel_nm']          # e.g. [700,700,700] nm per voxel

        layer_points = []
        last_point   = None
        for ann in layer.get("annotations", []):
            ann_type = ann.get("type")

            # ----------- POINT -------------------------------------------------
            if ann_type == "point":
                pt = ann.get("point")
                if pt and len(pt) == 3:
                    # convert once ✔︎
                    pt_nm = (pt[0]*vox_nm[0], pt[1]*vox_nm[1], pt[2]*vox_nm[2])

                    all_points_coords.append(pt_nm)
                    layer_points.append(pt_nm)

                    if last_point and not params.get('order_by_distance'):
                        original_segments.append((tuple(last_point), tuple(pt_nm)))
                    last_point = pt_nm

            # ----------- LINE --------------------------------------------------
            elif ann_type == "line":
                pA, pB = ann.get("pointA"), ann.get("pointB")
                if pA and len(pA) == 3 and pB and len(pB) == 3:
                    # convert both endpoints ✔︎
                    pA_nm = (pA[0]*vox_nm[0], pA[1]*vox_nm[1], pA[2]*vox_nm[2])
                    pB_nm = (pB[0]*vox_nm[0], pB[1]*vox_nm[1], pB[2]*vox_nm[2])

                    all_points_coords.extend([pA_nm, pB_nm])
                    layer_points.extend([pA_nm, pB_nm])

                    # **only add the edge here when we're NOT going to reorder later**
                    if not params.get('order_by_distance'):
                        original_segments.append((tuple(pA_nm), tuple(pB_nm)))

                last_point = None          # break sequential-point chain
            
            
        if params.get('order_by_distance') and layer_points:
            # nearest-neighbour ordering
            ordered = order_points_by_distance(layer_points)
            original_segments.extend((a, b) for a, b in zip(ordered[:-1], ordered[1:]))
        else:
            # original click order (already stored for point-by-point clicks)
            original_segments.extend((a, b) for a, b in zip(layer_points[:-1], layer_points[1:]))
    if not found_radius:
        print(f"  No 'r<num>' tag found in layer names for group {segment_id}. Using default offset radius: {effective_radius}.")


    if not all_points_coords:
        print(f"  No points found for segment ID {segment_id}. Skipping.")
        return {}, effective_radius # Return empty dict and radius

    # 2. Create unique point list and mapping
    # Use tuples for points to make them hashable for sets/dicts
    unique_points_list = sorted(list(set(all_points_coords))) # Sort for consistent indexing
    point_to_index = {pt_tuple: i for i, pt_tuple in enumerate(unique_points_list)}
    num_unique_points = len(unique_points_list)
    print(f"  Found {num_unique_points} unique points for ID {segment_id}.")
    print(f"  Effective radius for offset generation (if enabled) for this group: {effective_radius}.")


    if num_unique_points < 2:
         print("  Not enough unique points (< 2) to form segments. Treating as isolated points.")
         # Fit "spline" to single point (is just the point)
         # Need to decide if offsets make sense for isolated points. Probably not.
         single_point_curve = [list(unique_points_list[0])] if num_unique_points == 1 else []
         # Return structure compatible with downstream processing
         final_curves_for_id = { "main_segments": [single_point_curve] if single_point_curve else [], "offset_curves": [] }
         return final_curves_for_id, effective_radius


    # 3. Build the graph
    G = nx.Graph()
    G.add_nodes_from(range(num_unique_points))
    edges_added = set()
    original_segments = [(tuple(a), tuple(b)) for a, b in original_segments]
    for pA_tuple, pB_tuple in original_segments:
        if pA_tuple in point_to_index and pB_tuple in point_to_index:
            idxA = point_to_index[pA_tuple]
            idxB = point_to_index[pB_tuple]
            if idxA != idxB: # Avoid self-loops
                 edge = tuple(sorted((idxA, idxB)))
                 if edge not in edges_added:
                     G.add_edge(idxA, idxB)
                     edges_added.add(edge)

    print(f"  Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Handle disconnected graph components if necessary (simplified: use largest component)
    if num_unique_points > 0 and G.number_of_nodes() > 0 and not nx.is_connected(G):
        print(f"  Warning: Graph for ID {segment_id} is not connected. Processing largest component only.")
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy() # Work with the largest component's subgraph
        # Rebuild unique_points_list and point_to_index based on the subgraph? Not strictly necessary
        # if we keep using original indices, but might simplify segment finding logic.
        # Let's stick to original indices for now. Segment finding should work on the subgraph.
        print(f"  Subgraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        # Update num_unique_points to reflect the subgraph? Only matters if used later.
        # num_unique_points = G.number_of_nodes()


    # 4. Find segments (paths between endpoints/branchpoints)
    segment_node_indices = find_segments_in_graph(G)
    if not segment_node_indices:
         print("  Could not identify any segments in the graph.")
         # Fallback: Fit one spline to all *connected* points if possible
         print("  Fallback: Attempting to fit one spline to points in the (largest) component.")
         component_node_indices = list(G.nodes())
         if not component_node_indices:
             print("  No nodes in the component for fallback spline.")
             segment_splines = []
         else:
             component_points_np = np.array([list(unique_points_list[idx]) for idx in component_node_indices])
             if len(component_points_np) > 1:
                 try:
                     # Sort points roughly by projecting onto principal component? Heuristic.
                     mean = np.mean(component_points_np, axis=0)
                     _, _, v = np.linalg.svd(component_points_np - mean)
                     proj = np.dot(component_points_np - mean, v[0])
                     sorted_indices_in_comp = np.argsort(proj)
                     sorted_points = component_points_np[sorted_indices_in_comp].tolist()
                     single_spline = fit_spline_adaptive(sorted_points, n_samples=len(sorted_points)*params['sampling_factor'], smoothing=params['smoothing']) # High sampling?
                     segment_splines = [single_spline] if single_spline else []
                 except Exception as e:
                      print(f"  Error during fallback single spline fit: {e}")
                      segment_splines = [] # Failed entirely
             elif len(component_points_np) == 1:
                 segment_splines = [component_points_np.tolist()]
             else:
                 segment_splines = []

    else:
        # 5. Fit spline to each segment
        segment_splines = []
        print(f"  Fitting splines to {len(segment_node_indices)} segments...")
        for i, node_indices in enumerate(segment_node_indices):
            # Ensure we use the original coordinates based on indices
            segment_points = [list(unique_points_list[idx]) for idx in node_indices if idx < len(unique_points_list)] # Safety check idx
            if len(segment_points) < 2:
                print(f"    Skipping segment {i}: only {len(segment_points)} point(s).")
                continue # Cannot fit spline to single point here

            # Determine n_samples based on original segment length/points
            num_orig_points_in_segment = len(segment_points) # Rough estimate
            n_samples_seg = max(2, int(round(num_orig_points_in_segment * params['sampling_factor'])))

            print(f"    Fitting segment {i} ({len(segment_points)} pts) with {n_samples_seg} samples, smoothing={params['smoothing']}")
            try:
                 fitted_spline = fit_spline_adaptive(segment_points, n_samples=n_samples_seg, smoothing=params['smoothing'])
                 if fitted_spline: # Check if fitting was successful
                      segment_splines.append(fitted_spline)
                 else:
                      print(f"    Warning: Spline fitting failed for segment {i}. Skipping.")
            except Exception as e:
                 print(f"    Error fitting spline for segment {i}: {e}. Skipping.")


    # 6. Generate offsets (if requested) for EACH fitted segment spline
    all_offset_curves_for_id = [] # Collect only offsets separately for structure

    if params['generate_offsets'] and effective_radius > 0 and segment_splines: # Check segment_splines not empty
        print(f"  Generating offsets for {len(segment_splines)} fitted segments (Radius={effective_radius})...")
        # Compute n_offsets once based on the group's effective radius
        n_offsets_computed = compute_n_offsets_from_radius(
            effective_radius,
            base_factor=params['n_offsets_factor'],
            random_variance=params['n_offsets_variance'],
            min_offsets=params['min_offsets'],
            max_offsets=params['max_offsets']
        )
        print(f"    (Using {n_offsets_computed} offsets per segment)")

        for i, main_segment_spline in enumerate(segment_splines):
             if len(main_segment_spline) < 2: continue # Skip offsets for tiny segments

             # --- Select offset generation method (same as in process_annotation_layer) ---
             # Using random_walk_twist_no_clamp as per original single-layer logic
             offset_curves_seg = generate_offset_splines_random_walk_twist_no_clamp(
                main_segment_spline,
                effective_radius, # This is the radius used for determining the spread of offsets
                n_offsets_computed,
                step_sigma=params['offset_step_sigma'],
                twist_sigma=params['offset_twist_sigma']
             )
             # --- (Other options would go here if selected) ---

             # Apply resampling and smoothing to offsets if requested
             if params['resample_offsets'] and params['offset_resample_factor'] > 1:
                 offset_curves_seg = resample_offset_curves(offset_curves_seg, factor=params['offset_resample_factor'])
             if params['post_offset_smoothing'] > 0:
                 smoothed_offsets_seg = []
                 for curve in offset_curves_seg:
                      # Ensure enough points for smoothing (typically need k+1=4 for cubic)
                      if len(curve) >= 4:
                           smoothed = fit_spline_adaptive(curve, n_samples=len(curve), smoothing=params['post_offset_smoothing'])
                           if smoothed: smoothed_offsets_seg.append( smoothed)
                           else: smoothed_offsets_seg.append(curve) # Keep original if smoothing fails
                      else:
                           smoothed_offsets_seg.append(curve) # Keep short curves
                 offset_curves_seg = smoothed_offsets_seg

             all_offset_curves_for_id.extend(offset_curves_seg) # Add offsets for this segment


    # Structure the final output for this segment ID group
    final_structure = {
         "main_segments": segment_splines, # List of main fitted curves
         "offset_curves": all_offset_curves_for_id # List of all offset curves for this ID
    }

    return final_structure, effective_radius # Return the dict of curves and the radius used for offsets


def fetch_precomputed_info(pre_url):

    """
    Read the JSON at …/info from a local path or http(s) URL.
    Returns (size_vox[3], voxel_nm[3], origin_nm[3]).
    """
    if not pre_url.endswith("/"):               # make sure we have a slash
        pre_url += "/"
    info_url = urllib.parse.urljoin(pre_url, "info")
    with urllib.request.urlopen(info_url) as fh:
        info = json.load(fh)

    # take the *first* scale entry
    s0 = info["scales"][0]
    size_vox  = s0["size"]              # [nx, ny, nz]
    voxel_nm  = s0["voxel_size"] if "voxel_size" in s0 else s0["resolution"]
    T         = s0.get("transform", [1,0,0,0, 0,1,0,0, 0,0,1,0])  # 3×4 row-major
    origin_nm = [T[3], T[7], T[11]]
    return size_vox, voxel_nm, origin_nm


def _normalize_precomputed_url(raw: str) -> str:
    """
    Extract the real precomputed path from all known Neuroglancer
    source-string variants and return it **without** any protocol prefix.

    Examples
    --------
    >>> _normalize_precomputed_url(
    ...   "precomputed://https://site/foo/bar")
    'https://site/foo/bar'

    >>> _normalize_precomputed_url(
    ...   "neuroglancer-precomputed://https://site/foo")
    'https://site/foo'

    >>> _normalize_precomputed_url(
    ...   "https://site/foo|neuroglancer-precomputed:")
    'https://site/foo'
    """
    # 1) trim recognised prefixes
    for prefix in ("precomputed://", "neuroglancer-precomputed://"):
        if raw.startswith(prefix):
            return raw[len(prefix):]

    # 2) strip pipe-suffix form
    if "|neuroglancer-precomputed" in raw:
        return raw.split("|neuroglancer-precomputed", 1)[0]

    # 3) nothing matched → give back unchanged
    return raw


def get_full_volume_info(state_json):
    """
    Returns (origin_nm, size_vox, voxel_nm) for the FIRST image layer.

    • Works with layer-level dimensions.
    • Works with only root-level voxelSize.
    • Falls back to a precomputed info request if necessary.
    """

    # ---------- 1. layer-level dims ----------
    for layer in state_json.get("layers", []):
        if layer.get("type") != "image":
            continue
        dims = layer.get("dimensions")
        if dims and all("size" in dims[ax] for ax in ("x", "y", "z")):
            size   = [dims[ax]["size"]      for ax in ("x", "y", "z")]
            vox_nm = [dims[ax]["voxelSize"] for ax in ("x", "y", "z")]
            T      = layer.get("transform")
            origin = [T[3], T[7], T[11]] if T and len(T) == 12 else [0, 0, 0]
            return origin, size, vox_nm

    # ---------- 2. root-level voxelSize ----------
    root_dims = state_json.get("dimensions")
    voxel_nm  = None
    if root_dims:
        voxel_nm = [root_dims[ax][0] * 1e9 for ax in ("x", "y", "z")]

    # ---------- 3. fetch info from precomputed source ----------
    for layer in state_json.get("layers", []):
        if layer.get("type") != "image":
            continue
        src = layer.get("source")
        url_field = src["url"] if isinstance(src, dict) else src
        if not isinstance(url_field, str):
            continue

        # normalise all viewer variants → bare HTTPS path
        pc_url = _normalize_precomputed_url(url_field)

        # simple heuristic: only ask the server if it looks like an https path
        if pc_url.startswith("http"):
            size_vox, vox_from_srv, origin_nm = fetch_precomputed_info(pc_url)
            # honour root-level voxel override, if any
            voxel_nm = voxel_nm or vox_from_srv
            return origin_nm, size_vox, voxel_nm

    raise ValueError("Could not determine volume size from state or info.")
    
def process_state_grouped(state, params):
    """
    Assign a segment_id to EVERY annotation layer.
    • If a layer name has 'ID###', use that number.
    • Otherwise hand out incremental IDs starting at
      params['default_segment_id'] (100 by default).

    Layers are **never merged**; even layers that share the
    same ID flow through the pipeline separately.
    """
    if "layers" not in state:
        print("No 'layers' key found in JSON state.")
        return state, []

    new_layers   = []
    next_auto_id = params.get("default_segment_id", 100)

    # Keep track of groups for downstream processing
    grouped_layers = defaultdict(list)
    non_ann_layers = []

    for layer in state.get("layers", []):
        if layer.get("type") == "annotation":
            seg_id = parse_segment_id_from_layer_name(layer.get("name", ""))

            if seg_id is None:
                seg_id = next_auto_id
                next_auto_id += 1

            layer["segment_id"] = seg_id             # handy later
            if f"ID{seg_id}" not in layer["name"]:
                layer["name"] = f"{layer['name']} ID{seg_id}".strip()

            grouped_layers[seg_id].append(layer)
            new_layers.append(layer)                 # keep order
        else:
            non_ann_layers.append(layer)
            new_layers.append(layer)

    print(f"Found {len(grouped_layers)} segment groups.")

    # === process each layer individually ===
    processed_spline_collection = []

    for seg_id, layers_for_id in grouped_layers.items():
        for layer in layers_for_id:
            curves_struct, offset_r = process_segment_group(
                seg_id, [layer], params
            )

            main_curves   = curves_struct.get("main_segments", [])
            offset_curves = curves_struct.get("offset_curves", [])
            all_curves    = main_curves + offset_curves

            if not all_curves:
                placeholder = {
                    "type": "annotation",
                    "name": f"Processed_ID_{seg_id}_(No_Geometry)",
                    "annotations": [],
                    "source": layer.get("source")
                }
                new_layers.append(placeholder)
                continue

            processed_spline_collection.append({
                "segment_id": seg_id,
                "curves_structure": curves_struct,
                "all_curves": all_curves,
                "base_radius": offset_r
            })

            # Build NG layer for **this specific annotation layer**
            out_layer = layer.copy()
            out_layer["name"] = f"Processed_ID_{seg_id}_MainSegments"
            anns = []
            for idx, curve in enumerate(main_curves):
                if curve:
                    anns.extend(
                        create_annotations_from_curve(
                            curve, params["annotation_mode"], spline_id=idx
                        )
                    )
            out_layer["annotations"] = anns
            if "source" not in out_layer and "source" in layer:
                out_layer["source"] = layer["source"]
            new_layers.append(out_layer)

    # Write back updated layers list
    state["layers"] = new_layers
    return state, processed_spline_collection

def remove_duplicate_points(points):
    """Remove exact duplicate 3D coordinates."""
    if not points:
        return points
    # Use numpy for efficiency
    arr = np.array(points)
    _, unique_idx = np.unique(arr, axis=0, return_index=True)
    # Keep original order by sorting the unique indices
    unique_pts = arr[np.sort(unique_idx)]
    return unique_pts.tolist()

def parse_radius_from_layer_name(layer_name: str, default_radius: float):
    """
    Return the numeric radius encoded in the layer name or `default_radius`
    if none is found.

    Expected layer tags (case–sensitive):
        • ID###rNN   e.g.  ID100r20
        • _rNN       e.g.  some_name_r15
        • -rNN       e.g.  some-name-r8
        • space-rNN  e.g.  "axon r12"

    The function **ignores** capital “R” so tokens like “R1” used for
    “Round 1 / Replicate 1” no longer trigger a false match.
    """
    # 1.  ID###rNN   (no word-boundary between digits and 'r')
    m = re.search(r"ID\d+r(\d+(?:\.\d+)?)", layer_name)
    if m:
        return float(m.group(1))

    # 2.  spacer + rNN  (underscore, dash or whitespace before the 'r')
    m = re.search(r"(?:[_\-\s])r(\d+(?:\.\d+)?)\b", layer_name)
    if m:
        return float(m.group(1))

    # No valid tag found -> fall back
    return default_radius

def parse_segment_id_from_layer_name(layer_name):
    """
    Look for 'ID###' in the layer name and return the numeric value.
    If no tag is present, return None so the caller can assign an ID.
    """
    m = re.search(r"ID(\d+)", layer_name, re.IGNORECASE)
    return int(m.group(1)) if m else None

def curvature_3d(r1, r2):
    """
    Compute curvature k(t) in 3D given first and second derivatives:
      k(t) = || r'(t) x r''(t) || / ||r'(t)||^3
    """
    cross_prod = np.cross(r1, r2)
    num = np.linalg.norm(cross_prod, axis=1)
    denom = np.linalg.norm(r1, axis=1)**3
    with np.errstate(invalid='ignore', divide='ignore'):
        kappa = np.where(denom > 1e-12, num / denom, 0.0)
    return kappa

def densify_curve(curve_points, max_step=1.0):
    """
    Subdivide each segment so no two consecutive points
    exceed 'max_step' distance.
    """
    arr = np.array(curve_points)
    if len(arr) < 2:
        return arr.tolist()

    new_points = [arr[0].tolist()] # Start with the first point as a list
    for i in range(len(arr) - 1):
        pA = arr[i]
        pB = arr[i+1]
        seg_vec = pB - pA
        seg_len = np.linalg.norm(seg_vec)
        if seg_len > 1e-9: # Avoid division by zero for coincident points
            steps = int(np.ceil(seg_len / max_step))
            if steps > 1:
                for s in range(1, steps):
                    t = s / steps
                    p_sub = pA + t * seg_vec
                    new_points.append(p_sub.tolist())
        new_points.append(pB.tolist()) # Add the end point of the segment

    # Final check for duplicates just in case densification created identical points
    return remove_duplicate_points(new_points)


# --- SPLINE FITTING FUNCTIONS ---

def fit_spline_adaptive(points, n_samples=50, smoothing=0.0,
                        oversample_factor=5, curvature_weight=1.0):
    """
    Fit a 3D spline using splprep, adaptively sample based on curvature.
    """
    arr = np.array(points)
    if arr.shape[0] < 2:
        # print("  Warning: Cannot fit spline, less than 2 points.")
        return arr.tolist() # Return original points if too few

    m = arr.shape[0]
    # Ensure k is valid for the number of points
    k = min(3, m - 1)
    if k < 1:
        # print("  Warning: Cannot fit spline, requires at least 2 points (k>=1).")
        return arr.tolist()

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    try:
        # Reduce smoothing slightly if many points and low smoothing requested
        # This helps prevent rank deficiency issues in splprep with noisy data
        effective_smoothing = smoothing * m if smoothing > 0 else 0
        if m > 10 and smoothing < 0.1: # Heuristic adjustment
             effective_smoothing = max(effective_smoothing, 0.1 * m)

        tck, u = splprep([x, y, z], s=effective_smoothing, k=k, full_output=False) # Don't need full output
    except Exception as e:
         print(f"  Error during splprep: {e}. Returning raw points.")
         # Consider trying with increased smoothing as a fallback?
         # try:
         #     tck, u = splprep([x, y, z], s=m, k=k) # Try smoothing=m
         # except: return arr.tolist()
         return arr.tolist()

    # Ensure n_samples is reasonable
    n_samples = max(2, int(n_samples))
    oversample_count = max(n_samples, int(oversample_factor * n_samples)) # Ensure oversample >= n_samples
    oversample_count = max(4, oversample_count) # Need at least 4 for derivative estimation? More is better.

    t_oversample = np.linspace(0, 1, oversample_count)
    pos_over = np.array(splev(t_oversample, tck, der=0)).T # Transpose to get Nx3

    # Check if derivatives can be calculated (depends on k)
    if k < 1:
        # Cannot calculate curvature if k < 1
        print("  Warning: Spline degree k < 1, cannot calculate curvature. Using uniform sampling.")
        t_final = np.linspace(0, 1, n_samples)
        final_points = np.array(splev(t_final, tck, der=0)).T
        return final_points.tolist()

    d1 = np.array(splev(t_oversample, tck, der=1)).T

    if k < 2:
         # Cannot calculate 2nd derivative if k < 2
         print("  Warning: Spline degree k < 2, cannot calculate curvature accurately. Using uniform sampling.")
         t_final = np.linspace(0, 1, n_samples)
         final_points = np.array(splev(t_final, tck, der=0)).T
         return final_points.tolist()

    d2 = np.array(splev(t_oversample, tck, der=2)).T
    kappa_over = curvature_3d(d1, d2)


    # Adaptive sampling based on curvature
    segment_costs = []
    for i in range(oversample_count - 1):
        pA = pos_over[i]
        pB = pos_over[i + 1]
        seg_len = np.linalg.norm(pB - pA)
        # Use average curvature of the segment ends
        c = 0.5 * (kappa_over[i] + kappa_over[i + 1])
        # Ensure curvature weight is non-negative
        effective_curvature_weight = max(0.0, curvature_weight)
        cost_i = seg_len * (1.0 + effective_curvature_weight * c)
        # Ensure cost is non-negative (can happen if curvature calculation yields negative?)
        segment_costs.append(max(0.0, cost_i))


    segment_costs = np.array(segment_costs)
    total_cost = np.sum(segment_costs)

    # Handle case where total cost is zero (e.g., straight line with zero curvature, or single segment)
    if total_cost < 1e-9:
        # print("  Warning: Zero cost for adaptive sampling (straight line?). Using uniform sampling.")
        t_final = np.linspace(0, 1, n_samples)
        final_points_arr = np.array(splev(t_final, tck, der=0)).T
        return final_points_arr.tolist()

    cumulative_cost = np.cumsum(segment_costs)
    # Normalize cumulative cost
    cumulative_cost_norm = cumulative_cost / total_cost

    # Generate target cumulative costs (normalized)
    target_costs_norm = np.linspace(0, 1, n_samples)

    # Find the corresponding 't' values for these target costs using interpolation
    # We interpolate the original t_oversample values based on the cumulative costs
    # Need to handle segments with zero cost (where cost doesn't increase)
    valid_idx = np.where(np.diff(cumulative_cost_norm, prepend=0) > 1e-12)[0] # Indices where cost increases
    if len(valid_idx) < 2: # Not enough distinct cost points to interpolate
        print("  Warning: Not enough cost variation for adaptive sampling. Using uniform.")
        t_final = np.linspace(0, 1, n_samples)
        final_points_arr = np.array(splev(t_final, tck, der=0)).T
        return final_points_arr.tolist()

    # Interpolate: given target normalized costs, find the corresponding t parameter values
    # We map from cumulative_cost_norm (y) to t_oversample (x)
    # Need to ensure cumulative_cost_norm is strictly increasing for interp
    # Use the indices 'valid_idx' corresponding to non-zero cost segments
    # Include the start point (t=0, cost=0)
    interp_t = np.concatenate(([0.0], t_oversample[1:][valid_idx]))
    interp_cost = np.concatenate(([0.0], cumulative_cost_norm[valid_idx]))

    # Ensure uniqueness in interp_cost for np.interp
    unique_interp_cost, unique_indices = np.unique(interp_cost, return_index=True)
    unique_interp_t = interp_t[unique_indices]

    if len(unique_interp_cost) < 2: # Still not enough points after unique
         print("  Warning: Not enough unique cost points after filtering. Using uniform.")
         t_final = np.linspace(0, 1, n_samples)
         final_points_arr = np.array(splev(t_final, tck, der=0)).T
         return final_points_arr.tolist()


    t_final_adaptive = np.interp(target_costs_norm, unique_interp_cost, unique_interp_t)

    # Evaluate spline at these adaptively spaced 't' values
    final_points_arr = np.array(splev(t_final_adaptive, tck, der=0)).T

    # Ensure exact start and end points from original oversampling
    final_points_arr[0] = pos_over[0]
    final_points_arr[-1] = pos_over[-1]


    return final_points_arr.tolist()


def fit_loose_spline(points, n_samples, smoothing=5.0):
    """
    Fit a 3D spline with potentially high smoothing and resample uniformly.
    """
    arr = np.array(points)
    if arr.shape[0] < 2: return arr.tolist()
    m = arr.shape[0]
    k = min(3, m - 1)
    if k < 1: return arr.tolist()

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    try:
        # Scale smoothing by number of points
        effective_smoothing = smoothing * m if smoothing > 0 else 0
        tck, u = splprep([x, y, z], s=effective_smoothing, k=k)
        u_new = np.linspace(0, 1, max(2, int(n_samples)))
        x_new, y_new, z_new = splev(u_new, tck)
        resampled = np.column_stack((x_new, y_new, z_new))
        return resampled.tolist()
    except Exception as e:
        print(f"  Error during loose spline fit: {e}. Returning raw points.")
        return arr.tolist()

def refit_curve_as_spline(curve_points, desired_num_points, smoothing=0.0):
    """
    Refit discrete curve_points with a 3D spline, sample 'desired_num_points'.
    """
    return fit_loose_spline(curve_points, desired_num_points, smoothing=smoothing)


# --- OFFSET SPLINE GENERATION FUNCTIONS ---

# Helper to compute tangents robustly
def compute_tangents(points_array):
    n_points = points_array.shape[0]
    tangents = np.zeros_like(points_array)
    if n_points == 0:
        return tangents
    if n_points == 1:
        tangents[0] = np.array([1.0, 0.0, 0.0]) # Arbitrary tangent for single point
        return tangents

    # Forward difference for first point
    tangents[0] = points_array[1] - points_array[0]
    # Backward difference for last point
    tangents[-1] = points_array[-1] - points_array[-2]
    # Central difference for intermediate points
    if n_points > 2:
        tangents[1:-1] = points_array[2:] - points_array[:-2]

    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    # Avoid division by zero for zero-length segments or coincident points
    zero_norm_mask = norms.flatten() < 1e-9
    if np.any(zero_norm_mask):
        # For zero-norm tangents, try to propagate neighbor tangent
        for i in np.where(zero_norm_mask)[0]:
            # Try backward first
            if i > 0 and not zero_norm_mask[i-1] and np.linalg.norm(tangents[i-1]) > 1e-9:
                tangents[i] = tangents[i-1]
                norms[i] = norms[i-1]
            # Try forward next
            elif i < n_points - 1 and not zero_norm_mask[i+1]:
                 # Need to compute the next one's tangent if it's based on central diff
                 if i+1 < n_points -1: # If next is not the last point
                      next_tangent = points_array[i+2] - points_array[i]
                 else: # Next is the last point (use backward diff)
                      next_tangent = points_array[i+1] - points_array[i]

                 next_norm = np.linalg.norm(next_tangent)
                 if next_norm > 1e-9:
                     tangents[i] = next_tangent
                     norms[i] = next_norm
                 # If forward also fails, it will remain zero and be defaulted later
            # Else: Isolated zero-norm or start/end zero-norm: leave as zero for now

    # --- final normalisation inside compute_tangents ---
    valid = (norms.squeeze() > 1e-9)          # 1-D boolean mask, shape (N,)

    # divide every good tangent row by its length
    tangents[valid] /= norms[valid].reshape(-1, 1)

    # default any still-zero rows
    tangents[~valid] = np.array([1.0, 0.0, 0.0])

    return tangents

# Helper to get Normal and Binormal vectors (Frenet Frame - simplified)
# This version is simpler and avoids issues with zero curvature, suitable for offsets
def get_frenet_frame(T):
    # Choose an arbitrary vector V not parallel to T
    # Check alignment with Z-axis
    if abs(np.dot(T, np.array([0, 0, 1]))) < 0.999:
        V = np.array([0, 0, 1])
    # Check alignment with Y-axis if close to Z
    elif abs(np.dot(T, np.array([0, 1, 0]))) < 0.999:
        V = np.array([0, 1, 0])
    # Otherwise use X-axis
    else:
        V = np.array([1, 0, 0])

    # Calculate Normal vector (perpendicular to T and V)
    N = np.cross(T, V)
    normN = np.linalg.norm(N)
    # If T and V were parallel (shouldn't happen with checks above unless T is zero)
    if normN < 1e-9:
        # Fallback: if T was degenerate, try cross with X axis
        V = np.array([1, 0, 0])
        N = np.cross(T, V)
        normN = np.linalg.norm(N)
        if normN < 1e-9:
            # If still degenerate, use a default frame (e.g., T=[1,0,0])
            # This implies T was likely [0,0,0] originally, handled by compute_tangents setting it to [1,0,0]
            # So, N should be [0,1,0] and B should be [0,0,1]
            T_fallback = np.array([1.,0.,0.]) # Ensure T is valid if degenerate (Original T might be returned or this fallback)
            N = np.array([0.,1.,0.])
            B = np.array([0.,0.,1.])
            # If T was [0,0,0], compute_tangents sets it to [1,0,0].
            # So N and B will be [0,1,0] and [0,0,1] based on that.
            # The critical part is that T is not zero vector before this.
            # If T is a non-zero vector, N and B will be correctly computed.
            # The only danger is if T itself is [0,0,0], which compute_tangents tries to prevent.
            return N, B # Return potentially default N, B if T was pathological

    N /= normN

    # Calculate Binormal vector (perpendicular to T and N)
    B = np.cross(T, N)
    # B should already be normalized if T and N are orthonormal and unit length
    # normB = np.linalg.norm(B) # Optional check
    # if normB > 1e-9: B /= normB

    return N, B


def generate_offset_splines(main_points, radius, n_offsets=8, wiggle=0.0):
    """
    Original offset generation: Evenly spaced circle + wiggle.
    Wiggle is clamped so offset distance <= radius.
    """
    rng = np.random.default_rng()
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    offset_curves = [[] for _ in range(n_offsets)]
    base_angles = np.linspace(0, 2*np.pi, n_offsets, endpoint=False)

    for i in range(n_points):
        T = tangents[i]
        N, B = get_frenet_frame(T)

        for j in range(n_offsets):
            # Wiggle radius, clamping to [0, radius]
            r_j = radius + rng.uniform(-wiggle, wiggle) * radius # Scale wiggle by radius
            r_j = max(0, min(r_j, radius)) # Clamp to [0, radius]

            # Wiggle angle (can wrap around)
            angle_j = base_angles[j] + rng.uniform(-wiggle*np.pi, wiggle*np.pi) # Wiggle angle proportionally

            offset_vec = r_j * (np.cos(angle_j)*N + np.sin(angle_j)*B)
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves

def generate_offset_splines_random_disk(main_points, max_radius, n_offsets=8):
    """
    Offsets randomly sampled *within* the disk area.
    """
    rng = np.random.default_rng()
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    offset_curves = [[] for _ in range(n_offsets)]

    for i in range(n_points):
        T = tangents[i]
        N, B = get_frenet_frame(T)

        for j in range(n_offsets):
            # Sample radius uniformly within the disk area (sqrt for uniform area density)
            r_j = max_radius * np.sqrt(rng.random())
            angle_j = 2.0 * np.pi * rng.random()

            offset_vec = r_j * (np.cos(angle_j)*N + np.sin(angle_j)*B)
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves


def generate_offset_splines_random_walk(main_points, max_radius, n_offsets=8, step_sigma=0.2, seed=None):
    """ Random walk in local plane, clamped to boundary."""
    rng = np.random.default_rng(seed)
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    offset_curves = [[] for _ in range(n_offsets)]
    # Store offsets in the local (N, B) plane coordinates
    plane_offsets = [np.zeros((n_points, 2), dtype=float) for _ in range(n_offsets)]

    # Initialize first point randomly in disk for each offset curve
    for j in range(n_offsets):
        r = max_radius * np.sqrt(rng.random())
        angle = 2.0 * np.pi * rng.random()
        plane_offsets[j][0] = [r * np.cos(angle), r * np.sin(angle)]

    # Random walk for subsequent points
    for i in range(1, n_points):
        for j in range(n_offsets):
            prev_offset = plane_offsets[j][i-1]
            # Step size scales with radius? Or absolute? Let's use absolute based on sigma.
            step = rng.normal(loc=0.0, scale=step_sigma, size=2)
            candidate = prev_offset + step
            dist = np.linalg.norm(candidate)
            # Clamp candidate position to stay within the disk
            if dist > max_radius:
                candidate = (candidate / dist) * max_radius # Project back onto boundary
            plane_offsets[j][i] = candidate

    # Convert local plane offsets back to 3D coordinates
    for i in range(n_points):
        T = tangents[i]
        N, B = get_frenet_frame(T) # Get local frame for this point
        for j in range(n_offsets):
            # ox, oy are the coordinates in the N, B plane
            ox, oy = plane_offsets[j][i]
            offset_vec = ox*N + oy*B # Combine N and B vectors scaled by the plane coordinates
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves


def generate_offset_splines_random_walk_twist(main_points, max_radius, n_offsets=8, step_sigma=0.4, twist_sigma=0.2, seed=None):
    """ Random walk in local plane + shared frame twist angle."""
    rng = np.random.default_rng(seed)
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    plane_offsets = [np.zeros((n_points, 2), dtype=float) for _ in range(n_offsets)]
    twist_angles = np.zeros(n_points, dtype=float) # Shared twist angle along the curve

    # Initialize offsets randomly in disk at the first point
    for j in range(n_offsets):
        r = max_radius * np.sqrt(rng.random())
        angle = 2.0 * np.pi * rng.random()
        plane_offsets[j][0] = [r * np.cos(angle), r * np.sin(angle)]
    # Initial twist angle is 0

    # Walk and twist
    for i in range(1, n_points):
        # Accumulate twist angle (random walk for the angle itself)
        dtheta = rng.normal(0.0, twist_sigma)
        twist_angles[i] = twist_angles[i-1] + dtheta

        # Perform random walk step for each offset curve's position in the plane
        for j in range(n_offsets):
            prev_offset = plane_offsets[j][i-1]
            step = rng.normal(loc=0.0, scale=step_sigma, size=2)
            candidate = prev_offset + step
            dist = np.linalg.norm(candidate)
            # Clamp position to disk boundary
            if dist > max_radius:
                candidate = (candidate / dist) * max_radius
            plane_offsets[j][i] = candidate

    # Convert to 3D with twist applied to the local frame
    offset_curves = [[] for _ in range(n_offsets)]
    for i in range(n_points):
        T = tangents[i]
        N_base, B_base = get_frenet_frame(T) # Get the base local frame
        theta_i = twist_angles[i] # Get the accumulated twist angle at this point
        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)

        # Rotate the base frame vectors N_base, B_base by theta_i
        N_rot =  cos_t*N_base + sin_t*B_base
        B_rot = -sin_t*N_base + cos_t*B_base

        # Calculate the 3D offset point using the rotated frame
        for j in range(n_offsets):
            ox, oy = plane_offsets[j][i] # Get the 2D coords in the plane
            offset_vec = ox*N_rot + oy*B_rot # Combine using the *rotated* N and B
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves


def generate_offset_splines_random_walk_twist_independent(main_points, max_radius, n_offsets=8, step_sigma=0.1, twist_sigma=0.2, seed=None):
    """ Random walk + independent twist angle for each offset curve."""
    # Note: step_sigma reduced from 0.01 to 0.1 in default based on comparison with other methods.
    rng = np.random.default_rng(seed)
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    plane_offsets = [np.zeros((n_points, 2), dtype=float) for _ in range(n_offsets)]
    twist_angles = [np.zeros(n_points, dtype=float) for _ in range(n_offsets)] # Independent twist angles

    # Initialize offsets and initial random twists for each curve
    for j in range(n_offsets):
        r = max_radius * np.sqrt(rng.random())
        angle = 2.0 * np.pi * rng.random()
        plane_offsets[j][0] = [r * np.cos(angle), r * np.sin(angle)]
        twist_angles[j][0] = 2.0 * np.pi * rng.random() # Random start twist angle for each

    # Walk and twist independently for each offset curve
    for j in range(n_offsets):
        for i in range(1, n_points):
            # Walk step (clamped)
            prev_offset = plane_offsets[j][i-1]
            step = rng.normal(loc=0.0, scale=step_sigma, size=2)
            candidate = prev_offset + step
            dist = np.linalg.norm(candidate)
            if dist > max_radius:
                candidate = (candidate / dist) * max_radius
            plane_offsets[j][i] = candidate

            # Twist step (accumulated independently)
            dtheta = rng.normal(0.0, twist_sigma)
            twist_angles[j][i] = twist_angles[j][i-1] + dtheta

    # Convert to 3D with independent twists applied to the local frame
    offset_curves = [[] for _ in range(n_offsets)]
    for i in range(n_points):
        T = tangents[i]
        N_base, B_base = get_frenet_frame(T) # Base frame for this point on main curve

        for j in range(n_offsets):
            theta_ij = twist_angles[j][i] # Independent twist for curve j at point i
            cos_t, sin_t = np.cos(theta_ij), np.sin(theta_ij)

            # Rotate base frame
            N_rot =  cos_t*N_base + sin_t*B_base
            B_rot = -sin_t*N_base + cos_t*B_base

            # Calculate offset using rotated frame and plane coords
            ox, oy = plane_offsets[j][i]
            offset_vec = ox*N_rot + oy*B_rot
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves


def generate_offset_splines_random_walk_twist_no_clamp(main_points, max_radius, n_offsets=8, step_sigma=0.3, twist_sigma=0.1, max_retries=100, seed=None):
    """ Random walk with rejection sampling (stay inside disk) + independent twist."""
    rng = np.random.default_rng(seed)
    main_points_arr = np.array(main_points)
    n_points = main_points_arr.shape[0]
    if n_points == 0: return []

    tangents = compute_tangents(main_points_arr)
    plane_offsets = [np.zeros((n_points, 2), dtype=float) for _ in range(n_offsets)]
    twist_angles = [np.zeros(n_points, dtype=float) for _ in range(n_offsets)] # Independent twist

    # Initialize offsets and twists
    for j in range(n_offsets):
        r = max_radius * np.sqrt(rng.random())
        angle = 2.0 * np.pi * rng.random()
        plane_offsets[j][0] = [r*np.cos(angle), r*np.sin(angle)]
        twist_angles[j][0] = 2.0 * np.pi * rng.random()

    # Walk with rejection and twist independently
    for j in range(n_offsets):
        for i in range(1, n_points):
            prev_offset = plane_offsets[j][i-1]

            # Attempt random step, retry if proposed step goes outside disk
            for retry_count in range(max_retries):
                step = rng.normal(loc=0.0, scale=step_sigma, size=2)
                candidate = prev_offset + step
                if np.linalg.norm(candidate) <= max_radius:
                    plane_offsets[j][i] = candidate
                    break # Valid step found
            else:
                # If max_retries reached, stay put (or could implement other strategy)
                plane_offsets[j][i] = prev_offset
                if retry_count > 0: # Only print warning if retries actually happened
                    print(f"  Warning: Max retries ({max_retries}) reached for random walk step (offset {j}, point {i}). Staying put.")

            # Twist step (accumulated independently)
            dtheta = rng.normal(0.0, twist_sigma)
            twist_angles[j][i] = twist_angles[j][i-1] + dtheta

    # Convert to 3D with independent twists
    offset_curves = [[] for _ in range(n_offsets)]
    for i in range(n_points):
        T = tangents[i]
        N_base, B_base = get_frenet_frame(T) # Base frame

        for j in range(n_offsets):
            theta_ij = twist_angles[j][i] # Independent twist
            cos_t, sin_t = np.cos(theta_ij), np.sin(theta_ij)

            # Rotate frame
            N_rot =  cos_t*N_base + sin_t*B_base
            B_rot = -sin_t*N_base + cos_t*B_base

            # Calculate offset
            ox, oy = plane_offsets[j][i]
            offset_vec = ox*N_rot + oy*B_rot
            offset_point = main_points_arr[i] + offset_vec
            offset_curves[j].append(offset_point.tolist())

    return offset_curves

def resample_offset_curves(offset_curves, factor=5):
    """
    Resample each offset curve using spline fitting to increase point density.
    """
    resampled = []
    if factor <= 1: return offset_curves # No resampling needed

    for curve in offset_curves:
        if len(curve) < 2:
             resampled.append(curve) # Cannot resample < 2 points
             continue
        # Calculate desired number of points
        desired_num_points = max(2, int(round(len(curve) * factor)))
        # Use spline fitting (adaptive or loose) to resample
        # Using adaptive might preserve features better, but loose is simpler/faster
        # Let's use adaptive with moderate smoothing for resampling offsets
        new_curve = fit_spline_adaptive(curve, n_samples=desired_num_points, smoothing=0.5) # Moderate smoothing
        if not new_curve: # Handle potential failure of spline fit
            print(f"Warning: Resampling failed for an offset curve (length {len(curve)}). Keeping original.")
            resampled.append(curve)
        else:
            resampled.append(new_curve)
    return resampled


# --- ANNOTATION & SKELETON FORMATTING ---

def create_annotations_from_curve(curve_points, annotation_mode, spline_id=0):
    """ Create Neuroglancer annotations from curve points. """
    annotations = []
    num_pts = len(curve_points)
    if num_pts == 0:
        return annotations

    # Ensure points are lists of floats/ints for JSON serialization
    points_as_lists = [list(map(float, p)) for p in curve_points]


    if annotation_mode == "point":
        for i, pt in enumerate(points_as_lists):
            annotations.append({
                "type": "point",
                "point": pt,
                "id": str(uuid.uuid4()),
                #"description": f"Spline {spline_id}, Pt {i}" # Optional description
            })
    elif annotation_mode == "line":
        if num_pts == 1: # Handle single point case for lines -> output as point
             annotations.append({
                "type": "point",
                "point": points_as_lists[0],
                "id": str(uuid.uuid4()),
                #"description": f"Spline {spline_id}, Pt 0 (single)"
            })
        else:
            for i in range(num_pts - 1):
                annotations.append({
                    "type": "line",
                    "pointA": points_as_lists[i],
                    "pointB": points_as_lists[i+1],
                    "id": str(uuid.uuid4()),
                    #"description": f"Spline {spline_id}, Seg {i}" # Optional description
                })
    else:
        print(f"Warning: Unknown annotation_mode '{annotation_mode}'. No annotations created.")
    return annotations

def create_skeleton_info_json(output_dir, transform=None, voxel_size=[1.0, 1.0, 1.0]):
    """ Creates the 'info' file for precomputed skeletons. """
    if transform is None:
        # Default transform: identity matrix + zero translation
        transform = [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0]
    # Ensure transform is a list of 12 numbers
    if not isinstance(transform, list) or len(transform) != 12:
         print(f"Warning: Invalid transform provided {transform}. Using default identity.")
         transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Ensure voxel size is valid list/tuple of 3 numbers
    if not isinstance(voxel_size, (list, tuple)) or len(voxel_size) != 3:
        print(f"Warning: Invalid voxel_size provided {voxel_size}. Using default [1,1,1].")
        voxel_size = [1.0, 1.0, 1.0]

    skeleton_info = {
        "@type": "neuroglancer_skeletons",
        "transform": transform, # 3x4 matrix flattened [R T] column-major
        "vertex_attributes": [], # No extra attributes for now
        "voxel_size": voxel_size, # Add voxel size
        "sharding": None # Not using sharding for now
    }
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, "info")
    try:
        with open(info_path, 'w') as f:
            json.dump(skeleton_info, f, indent=2)
        print(f"Created skeleton info: {info_path}")
    except Exception as e:
        print(f"Error writing skeleton info file {info_path}: {e}")


def write_skeleton_file(output_dir, segment_id, vertices, edges):
    """ Writes a single precomputed skeleton binary file (gzipped). """
    if not vertices:
        # print(f"  Skipping skeleton write for segment {segment_id} (no vertices).")
        return
    file_path = os.path.join(output_dir, str(segment_id))
    # print(f"  Writing skeleton for segment {segment_id} -> {file_path}")
    num_vertices = len(vertices)
    num_edges = len(edges)

    try:
        # Use gzip compression
        with gzip.open(file_path, 'wb', compresslevel=6) as f: # Use moderate compression
            # Write header: num_vertices (uint32), num_edges (uint32), little-endian
            f.write(struct.pack('<II', num_vertices, num_edges))
            # Write vertices: x,y,z (float32) for each vertex, little-endian
            for v in vertices:
                # Ensure vertex coordinates are floats before packing
                f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
            # Write edges: v1_idx, v2_idx (uint32) for each edge, little-endian
            for e in edges:
                # Ensure edge indices are integers before packing
                f.write(struct.pack('<II', int(e[0]), int(e[1])))
        # print(f"  Skeleton file written for segment {segment_id} with {num_vertices} vertices, {num_edges} edges.")
    except Exception as e:
         print(f"Error writing skeleton file {file_path}: {e}")
def order_points_by_distance(points):
    """
    Return a new list where points are ordered by a greedy nearest-
    neighbour walk.  Input/Output are lists of (x,y,z) tuples or lists.
    """
    if len(points) < 3:
        return points[:]            # nothing to do

    pts = [np.array(p) for p in points]
    ordered = [pts.pop(0)]          # start with the first point
    while pts:
        last = ordered[-1]
        # index of closest remaining point
        i_min = min(range(len(pts)), key=lambda i: np.linalg.norm(last - pts[i]))
        ordered.append(pts.pop(i_min))
    return [p.tolist() for p in ordered]

def curves_to_vertices_edges(curves):
    """ Convert a list of curves (lists of points) to single vertex/edge list. """
    vertices = []
    edges = []
    vertex_offset = 0
    point_to_vertex_index = {} # To handle branching points correctly

    for curve in curves:
        num_curve_points = len(curve)
        if num_curve_points == 0:
            continue

        current_curve_indices = []
        for point in curve:
            point_tuple = tuple(point) # Make hashable
            if point_tuple in point_to_vertex_index:
                # Use existing vertex index if point already seen (branch)
                vertex_index = point_to_vertex_index[point_tuple]
            else:
                # Add new vertex and store its index
                vertex_index = vertex_offset
                vertices.append(list(point)) # Store as list [x,y,z]
                point_to_vertex_index[point_tuple] = vertex_index
                vertex_offset += 1
            current_curve_indices.append(vertex_index)

        # Add edges for this curve segment using the obtained vertex indices
        if len(current_curve_indices) > 1:
            for i in range(len(current_curve_indices) - 1):
                 idx1 = current_curve_indices[i]
                 idx2 = current_curve_indices[i+1]
                 if idx1 != idx2: # Avoid self-loops if a point was duplicated in input curve
                     # Add edge if not already present (handles branches merging back)
                     edge_tuple = tuple(sorted((idx1, idx2)))
                     # We don't explicitly check for duplicate edges here,
                     # assuming the binary format/NG handles it.
                     # If needed, maintain a set of added edge_tuples.
                     edges.append([idx1, idx2])

    return vertices, edges


# --- SYNTHETIC SEGMENTATION FUNCTIONS ---

def compute_global_bounding_box(spline_collection_processed, margin=10.0, use_zero_origin=False):
    """
    Compute bounding box encompassing all curves in the processed collection.
    Expand by margin. Finds the max radius within the collection for margin adjustment.
    'spline_collection_processed' here is expected to be the final output collection
    with format: [{'segment_id': id, 'curves': [(points, radii), ...]}, ...]
    """
    all_points = []
    max_radius_seen = 0.0

    for segment_data in spline_collection_processed:
        # Access the structured curves data: list of (points, radii) tuples
        curves_with_radii = segment_data.get('curves', []) # 'curves' holds [(pts, radii), ...]
        for points, radii_list in curves_with_radii: # radii_list contains radius for each point
            if points: # Check if points list is not empty
                all_points.extend(points)
            if radii_list: # Check if radii list is not empty
                current_max_r = np.max(radii_list) if radii_list else 0
                max_radius_seen = max(max_radius_seen, current_max_r)


    if not all_points:
        print("Warning: No points found to compute bounding box.")
        return np.array([0,0,0]), np.array([1,1,1]), 0.0 # Default tiny box and zero radius

    all_points_arr = np.array(all_points, dtype=float)
    min_coords = np.min(all_points_arr, axis=0)
    max_coords = np.max(all_points_arr, axis=0)

    # *** NEW: Handle 'use_zero_origin' flag ***
    if use_zero_origin:
        print("  Using [0,0,0] as the origin for the bounding box.")
        min_coords = np.array([0.0, 0.0, 0.0])

    # Expand by margin + largest radius seen to ensure painted disks fit
    effective_margin = margin + max_radius_seen
    min_coords -= effective_margin
    max_coords += effective_margin

    # If using zero origin, don't let margin push coordinates below zero
    if use_zero_origin:
        min_coords = np.maximum(0.0, min_coords)

    # Ensure min is less than max in all dimensions
    min_coords = np.minimum(min_coords, max_coords - 1e-6) # Ensure separation

    print(f"Computed BBox: Min={min_coords}, Max={max_coords} (Margin={margin}, MaxRadius={max_radius_seen})")
    return min_coords, max_coords, max_radius_seen


def compute_n_offsets_from_radius(radius, base_factor=0.5, random_variance=2, min_offsets=1, max_offsets=20, seed=None):
    """ Determine number of offsets based on radius. """
    rng = np.random.default_rng(seed)
    # Ensure radius is positive for calculation
    calc_radius = max(0.0, radius)
    # Base number of offsets proportional to radius (or area?) - proportional to radius seems okay.
    n = int(round(calc_radius * base_factor))
    # Add random variation
    n += rng.integers(-random_variance, random_variance + 1)
    # Clamp to min/max bounds
    n = max(min_offsets, n)
    n = min(max_offsets, n)
    return n

def assign_base_radii(num_curves, lower_bound, upper_bound, distribution='uniform', seed=None):
    """ Assign base radii for multiple curves based on distribution. """
    rng = np.random.default_rng(seed)
    # Ensure bounds are valid
    lower_bound = max(0.1, lower_bound) # Minimum radius slightly > 0
    upper_bound = max(lower_bound, upper_bound) # Upper must be >= lower

    if num_curves <= 0:
        return []

    if distribution == 'uniform':
        # Uniform distribution between lower and upper bounds
        return rng.uniform(lower_bound, upper_bound, size=num_curves).tolist()
    elif distribution == 'normal':
        # Normal distribution centered between bounds
        mean = (lower_bound + upper_bound) / 2.0
        # Set std dev such that ~95% fall within 2*std dev (e.g., std = range/4)
        std_dev = (upper_bound - lower_bound) / 4.0
        # Handle case where bounds are equal (avoid std_dev=0)
        if std_dev < 1e-6:
             values = np.full(num_curves, mean)
        else:
             values = rng.normal(mean, std_dev, size=num_curves)
             # Clip results to ensure they stay within the specified bounds
             values = np.clip(values, lower_bound, upper_bound)
        return values.tolist()
    else:
        print(f"Warning: Unsupported distribution '{distribution}'. Using uniform.")
        return rng.uniform(lower_bound, upper_bound, size=num_curves).tolist()


# def compute_modulated_radius_curve_for_axon(n_points, base_radius, step_sigma=0.05, smoothing_window=5, seed=None): # OLD Signature
def compute_modulated_radius_curve_for_axon(n_points, group_base_radius, mod_lower_delta, mod_upper_delta, step_sigma=0.05, smoothing_window=5, seed=None): # NEW Signature
    """
    Generate a smoothly varying radius sequence centered around group_base_radius,
    modulated within [group_base_radius + mod_lower_delta, group_base_radius + mod_upper_delta].

    Args:
        n_points (int): Number of points in the radius curve.
        group_base_radius (float): The center radius (e.g., from r<num> tag).
        mod_lower_delta (float): The minimum amount to add to group_base_radius (can be negative).
        mod_upper_delta (float): The maximum amount to add to group_base_radius.
        step_sigma (float): Std deviation for the random walk step, relative to group_base_radius.
        smoothing_window (int): Size of the moving average window for smoothing.
        seed: Random seed for reproducibility.

    Returns:
        list: A list of modulated radius values of length n_points.
    """
    if n_points <= 0: return []
    rng = np.random.default_rng(seed)

    # --- Calculate absolute bounds based on group radius and deltas ---
    abs_lower_bound = group_base_radius + mod_lower_delta
    abs_upper_bound = group_base_radius + mod_upper_delta

    # --- Ensure bounds are logical and radius stays positive ---
    # Ensure lower bound is strictly positive
    abs_lower_bound = max(0.1, abs_lower_bound)
    # Ensure upper bound is greater than lower bound
    abs_upper_bound = max(abs_lower_bound + 1e-6, abs_upper_bound) # Add epsilon for strict inequality if needed

    # print(f"    Modulating Radius: Base={group_base_radius:.2f}, Delta=[{mod_lower_delta:.2f}, {mod_upper_delta:.2f}] => Abs Bounds=[{abs_lower_bound:.2f}, {abs_upper_bound:.2f}]") # Debug print

    # --- Random walk within the *absolute* bounds ---
    values = np.zeros(n_points)
    # Start value randomized within the new absolute bounds
    if n_points > 0: # Guard against n_points = 0 here
        if n_points > 1:
            # Start somewhere within the allowed absolute range
            values[0] = rng.uniform(abs_lower_bound, abs_upper_bound)
        else: # n_points == 1
            # If only one point, use the base radius clamped to the bounds
             values[0] = np.clip(group_base_radius, abs_lower_bound, abs_upper_bound)
        # Clamp the start value just in case uniform produced something slightly out (float precision)
        values[0] = np.clip(values[0], abs_lower_bound, abs_upper_bound)


    # Random walk for subsequent radius values
    # Use max(0.1, ...) for base radius in step calculation to avoid zero/negative scaling
    scaled_step_sigma = step_sigma * max(0.1, group_base_radius)
    for i in range(1, n_points):
        step = rng.normal(0, scaled_step_sigma)
        values[i] = values[i-1] + step
        # Clamp the radius to the *absolute* bounds immediately
        values[i] = np.clip(values[i], abs_lower_bound, abs_upper_bound)

    # --- Smoothing (logic remains the same, operates on 'values') ---
    if smoothing_window > 1 and n_points >= smoothing_window:
        # Ensure window size is odd for 'same' padding symmetry if using np.convolve 'same'
        window_size = int(smoothing_window)
        if window_size % 2 == 0:
            window_size += 1 # Make it odd
        window_size = min(window_size, n_points) # Window cannot be larger than array

        kernel = np.ones(window_size) / window_size
        pad_width = (window_size - 1) // 2
        padded_values = np.pad(values, pad_width, mode='reflect')
        values_smoothed = np.convolve(padded_values, kernel, mode='valid')

        if len(values_smoothed) != n_points:
             # print(f"Warning: Smoothed radius length mismatch ({len(values_smoothed)} vs {n_points}). Adjusting.")
             if len(values_smoothed) > n_points:
                 values_smoothed = values_smoothed[:n_points]
             else: # len(values_smoothed) < n_points
                 pad_needed = n_points - len(values_smoothed)
                 # Pad with edge values, but ensure correct length after padding
                 if pad_needed > 0 :
                    values_smoothed = np.pad(values_smoothed, (0, pad_needed), mode='edge')


        # --- Final clamp after smoothing using absolute bounds ---
        values_smoothed = np.clip(values_smoothed, abs_lower_bound, abs_upper_bound)
    else:
        values_smoothed = values # No smoothing if window is too small or not enough points

    return values_smoothed.tolist()


def draw_disk(slice_img, center_xy, radius_pixels, value):
    """ Draw a filled circle (disk) into a 2D numpy array using integer coords. """
    height, width = slice_img.shape
    # Ensure coordinates and radius are integers for indexing
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1])) # Center X, Y (columns, rows)
    rad_int = int(np.ceil(radius_pixels)) # Use ceil to be inclusive

    # Determine bounds for the loop (relative to image dimensions)
    ymin = max(0, cy - rad_int)
    ymax = min(height, cy + rad_int + 1)
    xmin = max(0, cx - rad_int)
    xmax = min(width, cx + rad_int + 1)

    # Create coordinate grids relative to the *integer* center
    y, x = np.ogrid[ymin:ymax, xmin:xmax] # Grid covering the bounding box

    # Calculate distance squared from the *float* center (more accurate circle)
    dist_sq = (x - center_xy[0])**2 + (y - center_xy[1])**2

    # Create the circular mask within the bounding box
    mask = dist_sq <= radius_pixels**2

    # Apply the value using the mask
    slice_img[ymin:ymax, xmin:xmax][mask] = value


def process_slice_and_save_2(slice_idx, global_min, voxel_size_xyz, vol_dims_yxz, spline_collection_with_radii, output_dir):
    """ Worker function to paint one slice using modulated radii and save it as compressed TIF. """
    first_seg_id = spline_collection_with_radii[0]['segment_id']

    # Physical Z coordinate of the current slice center
    z_phys_slice = global_min[2] + (slice_idx + 0.5) * voxel_size_xyz[2]

    # Volume dimensions expected as (Y, X, Z) for creating the 2D slice_img[y, x]
    slice_img = np.zeros((vol_dims_yxz[0], vol_dims_yxz[1]), dtype=np.uint16)

    # Inverse voxel sizes (X, Y, Z order assumed for voxel_size_xyz)
    inv_voxel_size_x = 1.0 / voxel_size_xyz[0]
    inv_voxel_size_y = 1.0 / voxel_size_xyz[1]
    # Voxel Z thickness (used for slab intersection check)
    voxel_depth_z = voxel_size_xyz[2]

    for seg_data in spline_collection_with_radii:
        seg_id = seg_data['segment_id']
        curves = seg_data['curves'] # List of (points, radii) tuples for this segment ID



        for (curve_points, mod_radii) in curves:
            if not curve_points or len(curve_points) < 2: # Handling single points later
                if len(curve_points) == 1 and mod_radii: # Handle single points
                    pA = np.array(curve_points[0])
                    rA = mod_radii[0]
                    dz_to_slice = abs(pA[2] - z_phys_slice)
                    if dz_to_slice <= rA: # Check if sphere intersects slice Z
                        circle_radius_phys = math.sqrt(max(0, rA**2 - dz_to_slice**2))
                        circle_radius_vox_x = circle_radius_phys * inv_voxel_size_x
                        circle_radius_vox_y = circle_radius_phys * inv_voxel_size_y
                        circle_radius_vox = (circle_radius_vox_x + circle_radius_vox_y) / 2.0
                        center_x_vox = (pA[0] - global_min[0]) * inv_voxel_size_x
                        center_y_vox = (pA[1] - global_min[1]) * inv_voxel_size_y
                        if circle_radius_vox >= 0.5:
                            draw_disk(slice_img, (center_x_vox, center_y_vox), circle_radius_vox, seg_id)
                continue # Skip if less than 2 points and not handled as single point

            # Iterate through line segments of the curve for better slice coverage
            for i in range(len(curve_points) - 1):
                pA = np.array(curve_points[i]) # Start point [x,y,z]
                pB = np.array(curve_points[i+1]) # End point [x,y,z]
                rA = mod_radii[i] # Radius at pA
                rB = mod_radii[i+1] # Radius at pB

                # Coarse Z check: Does the segment's Z-range (expanded by max radius) overlap the slice's Z-slab?
                z_min_seg = min(pA[2], pB[2])
                z_max_seg = max(pA[2], pB[2])
                max_r_seg = max(rA, rB)
                slice_z_min = z_phys_slice - 0.5 * voxel_depth_z
                slice_z_max = z_phys_slice + 0.5 * voxel_depth_z

                if max(slice_z_min, z_min_seg - max_r_seg) <= min(slice_z_max, z_max_seg + max_r_seg):
                    # Potential intersection - proceed with more detailed check
                    seg_vec = pB - pA
                    dz_seg = pB[2] - pA[2] # dz of the segment itself

                    t = -1 # Initialize t to indicate no direct intersection yet
                    if abs(dz_seg) < 1e-9: # Segment is parallel to XY plane
                        if slice_z_min <= pA[2] <= slice_z_max : # And it's within the slice Z slab
                            # The entire segment (or its projection) could be relevant
                            # Use average properties or properties of closer point if segment is short
                            # For simplicity, if parallel and in slab, consider it intersecting at its Z.
                            # The actual "intersection point" is any point on the segment for this Z.
                            # We are interested in the projection of the *tube* onto the slice.
                            # Pick midpoint, its radius, and its Z distance to slice.
                            # This case is tricky as the 't' for slice intersection is ill-defined.
                            # We care about the circle projected. Closest point of segment to slice's Z.
                            # This is pA[2] (or pB[2]).
                            intersect_pt_on_line = pA # or pB, z is same
                            intersect_radius_on_line = (rA + rB) / 2.0 # Average radius
                            dz_to_slice_center = abs(intersect_pt_on_line[2] - z_phys_slice)
                            center_pt_phys = intersect_pt_on_line # x,y of the segment
                        else:
                            continue # Parallel but not in slice Z slab
                    else:
                        # Segment is not parallel to XY plane, find parametric 't' for intersection
                        t = (z_phys_slice - pA[2]) / dz_seg

                    # Now determine center point and radius for painting based on 't' or endpoints
                    circle_radius_phys = 0.0
                    center_pt_phys = None

                    if 0 <= t <= 1: # Centerline intersects the slice plane within the segment
                        center_pt_phys = pA + t * seg_vec
                        intersect_radius_on_line = rA + t * (rB - rA) # Interpolated radius at this 't'
                        # dz_to_slice_center is 0 because center_pt_phys[2] == z_phys_slice
                        circle_radius_phys = intersect_radius_on_line
                    else: # Centerline does not intersect slice plane *within* the segment
                          # Check if endpoints' spheres intersect the slice plane
                        dist_A_to_sliceZ = abs(pA[2] - z_phys_slice)
                        dist_B_to_sliceZ = abs(pB[2] - z_phys_slice)

                        # Check if sphere at A intersects
                        intersect_A = dist_A_to_sliceZ <= rA
                        # Check if sphere at B intersects
                        intersect_B = dist_B_to_sliceZ <= rB

                        if intersect_A and (not intersect_B or dist_A_to_sliceZ <= dist_B_to_sliceZ):
                            # A is closer or only A intersects
                            center_pt_phys = pA
                            intersect_radius_on_line = rA
                            dz_to_slice_center = dist_A_to_sliceZ
                            circle_radius_phys = math.sqrt(max(0, intersect_radius_on_line**2 - dz_to_slice_center**2))
                        elif intersect_B:
                            # B is closer or only B intersects
                            center_pt_phys = pB
                            intersect_radius_on_line = rB
                            dz_to_slice_center = dist_B_to_sliceZ
                            circle_radius_phys = math.sqrt(max(0, intersect_radius_on_line**2 - dz_to_slice_center**2))
                        else:
                            # No part of the segment's tube (based on endpoints) crosses this slice
                            continue


                    # If we have a valid circle to draw
                    if center_pt_phys is not None and circle_radius_phys > 0:
                        # Convert physical radius and center to voxel coordinates for drawing
                        circle_radius_vox_x = circle_radius_phys * inv_voxel_size_x
                        circle_radius_vox_y = circle_radius_phys * inv_voxel_size_y
                        # Use average radius for drawing for an isotropic disk approximation
                        circle_radius_vox = (circle_radius_vox_x + circle_radius_vox_y) / 2.0

                        center_x_vox = (center_pt_phys[0] - global_min[0]) * inv_voxel_size_x
                        center_y_vox = (center_pt_phys[1] - global_min[1]) * inv_voxel_size_y

                        # Draw the disk if its radius is reasonably large (e.g., >= 0.5 voxels)
                        if circle_radius_vox >= 0.5:
                            draw_disk(slice_img, (center_x_vox, center_y_vox), circle_radius_vox, seg_id)
        
            if slice_idx == 0 and seg_id == first_seg_id:
                print("debug:", pA, "nm ; rA =", rA)
            # Handle the last point of the curve as a sphere if it wasn't covered as pA of a segment
            # This is mostly for curves of 1 point, or to ensure last sphere is painted
            # Note: logic for single points already added at the start of the loop.
            # If a curve has >1 point, the last point is pB of the last segment.
            # The sphere at pB of the last segment would be considered by the segment logic.

    # Save the generated slice image using tifffile with compression
    outpath = os.path.join(output_dir, f"slice_{slice_idx:07d}.tif")
    try:
        # Use zlib compression (level 1 for speed, 6 for balance, 9 for max)
        imwrite(outpath, slice_img, compression='zlib', compressionargs={'level': 1})
    except Exception as e:
        print(f"Error writing slice file {outpath}: {e}")
        return None # Indicate failure

    return slice_idx # Return index on success


def generate_segmentation_slices(
        spline_collection_with_radii,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        output_dir="segmentation",
        use_zero_origin=False,
        full_volume_origin_nm=None,     # ← NEW
        full_volume_dims_vox=None,     # ← NEW
        src_voxel_nm=None):     # ← NEW
    """ Generate synthetic segmentation slices using parallel processing. """
    if not spline_collection_with_radii:
         print("Spline collection is empty. Cannot generate segmentation.")
         return

    print("Calculating bounding box for segmentation...")
    # Compute margin based on voxel size (e.g., 2 voxels in each dim)
    
    
    # -----------------------------------------------------------------
    # If the user checked “Use [0,0,0] …” we want to paint into the
    # *entire* source volume, not a tight crop.
    # -----------------------------------------------------------------
    if use_zero_origin and full_volume_origin_nm and full_volume_dims_vox:
        global_min  = np.array(full_volume_origin_nm, dtype=float)
        # FULL-FIELD (use_zero_origin) allocation
        ratio = np.array(voxel_size_xyz) / np.array(src_voxel_nm)   # ← 4 for 4×
        vol_dims_xyz = np.ceil(
            np.array(full_volume_dims_vox, dtype=float) / ratio
        ).astype(int)
        global_max = global_min + vol_dims_xyz * np.array(voxel_size_xyz)
        print("Full-field mode:")
        print("  Origin (nm):", global_min)
        print("  Size  (vox):", vol_dims_xyz)      # → [3051 3051 2338]
    else:
        # --- EXISTING tight-bbox path (unchanged) --------------------
        print("Calculating bounding box for segmentation...")
        margin_xyz = np.array(voxel_size_xyz) * 2.0
        global_min, global_max, _ = compute_global_bounding_box(
            spline_collection_with_radii,
            margin=np.max(margin_xyz),
            use_zero_origin=use_zero_origin            # still honoured
        )
        vol_dims_xyz = np.ceil((global_max - global_min) /
                               voxel_size_xyz).astype(int) + 1
    # -----------------------------------------------------------------

    # Ensure voxel size is positive
    if any(v <= 0 for v in voxel_size_xyz):
         print(f"Error: Voxel size must be positive in all dimensions. Got {voxel_size_xyz}")
         return

    # Calculate volume dimensions in voxels
    vol_dims_xyz = np.ceil((global_max - global_min) / voxel_size_xyz).astype(int) + 1
    # Ensure dimensions are at least 1
    vol_dims_xyz = np.maximum(1, vol_dims_xyz)

    # Store dimensions as Y, X, Z order for worker function's slice image creation
    vol_dims_yxz = (vol_dims_xyz[1], vol_dims_xyz[0], vol_dims_xyz[2])
    num_slices = vol_dims_yxz[2] # Number of slices along Z

    # *** NEW: Create a more descriptive, unique folder name ***
    safe_min_coords_str = f"{int(round(global_min[0]))}_{int(round(global_min[1]))}_{int(round(global_min[2]))}".replace('-', 'neg')
    safe_size_str = f"{vol_dims_xyz[0]}_{vol_dims_xyz[1]}_{vol_dims_xyz[2]}"
    safe_vox_str = f"{voxel_size_xyz[0]:.2f}_{voxel_size_xyz[1]:.2f}_{voxel_size_xyz[2]:.2f}".replace('.', 'p')
    folder_name = f"seg_min_{safe_min_coords_str}_size_{safe_size_str}_vox_{safe_vox_str}"
    seg_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(seg_output_dir, exist_ok=True)

    print(f"Generating {num_slices} slices (Size: {vol_dims_yxz[1]}x{vol_dims_yxz[0]}) -> {seg_output_dir} ...")
    print(f"Volume dimensions (Voxels XYZ): {vol_dims_xyz}")
    print(f"Physical bounds Min: {global_min}, Max: {global_max}")
    print(f"Voxel size (XYZ): {voxel_size_xyz}")


    completed_count = 0
    # Use ProcessPoolExecutor for CPU-bound tasks
    # Adjust max_workers based on your CPU cores (leave one free?)
    max_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    print(f"Using up to {max_workers} workers.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all slice processing tasks
        futures = [
            executor.submit(
                process_slice_and_save_2, # Use the function handling modulated radii
                slice_idx,
                global_min,
                voxel_size_xyz, # Pass XYZ voxel size
                vol_dims_yxz, # Pass YXZ dimensions for slice creation
                spline_collection_with_radii, # Pass data with [(points, radii), ...] structure
                seg_output_dir
            )
            for slice_idx in range(num_slices)
        ]

        # Process results as they complete and show progress
        for future in as_completed(futures):
            try:
                result_idx = future.result() # Get the slice index (or None if error)
                if result_idx is not None: # Check for success
                    completed_count += 1
                    # Print progress less frequently using modulo
                    if completed_count % max(1, num_slices // 20) == 0 or completed_count == num_slices:
                        progress = (completed_count / num_slices) * 100
                        print(f"  Processed slice {result_idx:04d} ({completed_count}/{num_slices} - {progress:.1f}%)", end='\r' if completed_count != num_slices else '\n')
                        sys.stdout.flush()
                else:
                     print(f"  A slice failed to process. Check logs above.") # Error message printed in worker
            except Exception as exc:
                 # Log exceptions raised during future.result() or within the worker
                 print(f"Error retrieving result from worker: {exc}")
                 import traceback
                 traceback.print_exc()
        if completed_count < num_slices: print() # Newline if progress bar didn't finish with one

    print(f"Slice generation finished. {completed_count}/{num_slices} slices successfully written to {seg_output_dir}.")


# --- CORE PROCESSING LOGIC (kept separate from GUI class for clarity) ---

def run_processing_logic(params):
    """Encapsulates the main processing steps called by the GUI."""

    print("\n--- Starting Processing ---")
    print("Parameters:")
    for key, val in params.items():
        # Don't print the whole input path maybe
        if key != 'input_json': print(f"  {key}: {val}")
        else: print(f"  {key}: ...{os.path.basename(val)}")

    try:
        # 1. Load state
        print(f"Loading state from: {params['input_json']}")
        with open(params['input_json'], "r") as f:
            state = json.load(f)

        # Extract viewer state info early (voxel size, transform) if needed
        # Handle potential variations in state structure ('viewerState' vs root level)
        viewer_state_data = state.get('viewerState', state)
        input_voxel_size = viewer_state_data.get('voxelSize', [1.0, 1.0, 1.0])
        input_transform = viewer_state_data.get('transform', None)
        print(f"  Detected Voxel Size from JSON: {input_voxel_size}")
        print(f"  Detected Transform from JSON: {'Yes' if input_transform else 'No'}")




        # --- NEW: full-volume bookkeeping -------------------------------
        full_origin_nm, full_size_vox, src_vox_nm = get_full_volume_info(state)
        params.update({
            "full_origin_nm": full_origin_nm,
            "full_size_vox":  full_size_vox,
            "src_voxel_nm":   src_vox_nm,
            "radius_mode":    "native_voxels"     # <-- keep your chosen mode here
        })

        # --------------------------------------------------------------------------------
        # 2. ***NOW*** derive the *target* voxel size the seg volume will use
        # --------------------------------------------------------------------------------
        if params['treat_voxel_size_as_multiplier']:
            # GUI number is a multiplier, e.g. 4×
            final_voxel_size_xyz = (
                np.array(params['src_voxel_nm']) * params['voxel_size']
            ).tolist()                                            # => [2800, 2800, 2800]
        else:
            # GUI number is absolute nm value or list
            if isinstance(params['voxel_size'], (int, float)):
                final_voxel_size_xyz = [params['voxel_size']]*3
            else:
                final_voxel_size_xyz = list(params['voxel_size'])

        params['final_voxel_size_xyz'] = final_voxel_size_xyz
        print("→ Final voxel size (XYZ):", final_voxel_size_xyz)

        # --------------------------------------------------------------------------------
        # 3. Convert GUI radius to physical nanometres **once**
        # --------------------------------------------------------------------------------
        if params['radius_mode'] == 'native_voxels':          # GUI value = native voxels
            params['radius_nm'] = params['radius'] * params['src_voxel_nm'][0]
        else:                                                 # GUI value already nm
            params['radius_nm'] = params['radius']
        print("→ Effective drawing radius (nm):", params['radius_nm'])

        # 2. Process state using the *grouped* approach
        # process_state_grouped modifies 'state' in-place and returns the collection
        updated_state, processed_spline_collection = process_state_grouped(state, params)
        print(f"Processed state, collected final structures for {len(processed_spline_collection)} segment groups.")

        # 3. Save updated Neuroglancer state JSON
        saved_json_path = save_new_json(updated_state, params['input_json'], params['output_suffix'])
        if not saved_json_path:
             raise RuntimeError("Failed to save the updated JSON state.")

        # --- Post-processing: Skeletons, Segmentation, Visualization ---
        # The `processed_spline_collection` contains the final geometry per ID group
        # Format: [{'segment_id': id, 'curves_structure': {...}, 'all_curves': [...], 'base_radius': r}, ...]
        # where 'base_radius' is the r<num> or default offset radius.

        # 4. Generate Skeletons (if enabled) - One file per segment ID group
        if params['generate_skeleton'] and processed_spline_collection:
            skel_dir = params['skeleton_output_dir']
            print(f"Generating skeletons in: {skel_dir}")
            os.makedirs(skel_dir, exist_ok=True)
            # Create info file once, using voxel size from input state if possible
            create_skeleton_info_json(skel_dir, transform=input_transform, voxel_size=input_voxel_size)

            num_skel_written = 0
            for segment_data in processed_spline_collection:
                 seg_id = segment_data['segment_id']
                 all_curves_for_group = segment_data['all_curves'] # Use the flattened list [curve1, curve2,...]
                 if all_curves_for_group:
                     # Convert potentially complex curve structures (main+offsets) into vertices/edges
                     vertices, edges = curves_to_vertices_edges(all_curves_for_group)
                     if vertices:
                          write_skeleton_file(skel_dir, seg_id, vertices, edges)
                          num_skel_written += 1
                     #else: print(f"  Skipping skeleton for ID {seg_id} (no vertices generated).") # Verbose
                 #else: print(f"  Skipping skeleton for ID {seg_id} (no curves found).") # Verbose
            print(f"  Finished writing {num_skel_written} skeleton files.")


        # 5. Prepare for Segmentation/Viz (Apply radius modulation or assign constant radius)
        # This creates the final_output_collection used by segmentation and viz export.
        # Format: [{'segment_id': id, 'curves': [(points, radii_list), (points, radii_list), ...]}, ...]
        final_output_collection = []
        if (params['generate_segmentation'] or params['generate_viz_json']) and processed_spline_collection:
            print("Preparing final curves with radii for segmentation/visualization...")
            for segment_data in processed_spline_collection:
                seg_id = segment_data['segment_id']
                all_curves_for_group = segment_data['all_curves']
                # This is the r<num> from layer name, or params['radius'] (default offset radius) if no r<num>
                group_effective_radius = segment_data['base_radius']
                
                num_curves = len(all_curves_for_group)
                processed_segment_curves_with_radii = [] # To store [(points, radii_list), ...]

                if num_curves == 0:
                     final_output_collection.append({"segment_id": seg_id, "curves": []})
                     continue

                # --- Apply modulation or assign constant radius to each curve's points ---
                for curve_idx, curve_points in enumerate(all_curves_for_group):
                    n_points = len(curve_points)
                    if n_points == 0:
                        processed_segment_curves_with_radii.append(([], []))
                        continue

                    current_curve_radii = []
                    if params['modulate_radius']:
                        # Modulate radius: use group_effective_radius as center, GUI bounds as deltas
                        mod_lower_delta = params['radius_lower_bound'] # GUI "Radius +/- Range (Min)"
                        mod_upper_delta = params['radius_upper_bound'] # GUI "Radius +/- Range (Max)"

                        current_curve_radii = compute_modulated_radius_curve_for_axon(
                            n_points,
                            group_effective_radius, # Base radius to modulate around (e.g., r80)
                            mod_lower_delta,        # Min delta (e.g., 5.0)
                            mod_upper_delta,        # Max delta (e.g., 15.0)
                            params['radius_step_sigma'],
                            params['radius_smoothing_window'],
                            seed=None # Optional: hash(seg_id) + curve_idx
                        )
                        # print(f"  ID {seg_id}, Curve {curve_idx}: Modulated radii around {group_effective_radius:.2f} with deltas [{mod_lower_delta:.2f}, {mod_upper_delta:.2f}]. Result range: [{min(current_curve_radii):.2f}-{max(current_curve_radii):.2f}]")
                    else:
                        # NOT modulating radius: use group_effective_radius directly as a constant value.
                        # GUI fields for radius range/deltas/distribution are disabled and ignored in this mode.
                        current_curve_radii = [group_effective_radius] * n_points
                        # print(f"  ID {seg_id}, Curve {curve_idx}: Using constant radius {group_effective_radius:.2f}")


                    processed_segment_curves_with_radii.append((curve_points, current_curve_radii))

                # Add the processed data for this segment ID to the final collection
                final_output_collection.append({
                     "segment_id": seg_id,
                     "curves": processed_segment_curves_with_radii # List of (points, radii_list) tuples
                })


        # 6. Generate Segmentation (if enabled)
        if params['generate_segmentation'] and final_output_collection:
            seg_dir = params['segmentation_output_dir']
            print(f"Generating segmentation slices in: {seg_dir}")
            
            # ------------------------------------------------------------------
            # Derive the *target* voxel size that the segmentation will use
            # ------------------------------------------------------------------
            if params['treat_voxel_size_as_multiplier']:
                # GUI number is a multiplier  (4×)
                final_voxel_size_xyz = (np.array(params['src_voxel_nm']) *
                                        params['voxel_size']).tolist()      # → [2800, 2800, 2800]
            else:
                # GUI number is an absolute nm value or list
                if isinstance(params['voxel_size'], (int, float)):
                    final_voxel_size_xyz = [params['voxel_size']]*3
                else:
                    final_voxel_size_xyz = list(params['voxel_size'])

            params['final_voxel_size_xyz'] = final_voxel_size_xyz
            print("→ Final voxel size (XYZ):", final_voxel_size_xyz)

            generate_segmentation_slices(
                final_output_collection,
                voxel_size_xyz     = params['final_voxel_size_xyz'],
                output_dir         = seg_dir,
                use_zero_origin    = params['use_zero_origin'],
                full_volume_origin_nm = params['full_origin_nm'],
                full_volume_dims_vox  = params['full_size_vox'],
                src_voxel_nm          = params['src_voxel_nm']
            )

        # 7. Export for Visualization (if enabled)
        if params['generate_viz_json'] and final_output_collection:
             viz_path = params['viz_json_path']
             print(f"Exporting splines for visualization to: {viz_path}")
             # Pass collection with radii - export function currently ignores radii but could use them
             export_splines_for_threejs(final_output_collection, viz_path)


        print(f"\n--- Processing Complete ---")
        print(f"Updated JSON state saved to: {saved_json_path}")
        # Return success status and message
        return True, f"Processing complete!\nUpdated JSON saved to: {os.path.basename(saved_json_path)}\nCheck console for details."

    except Exception as e:
        print("\n--- ERROR DURING PROCESSING ---")
        import traceback
        traceback.print_exc()
        # Return failure status and error message
        return False, f"An error occurred during processing:\n\n{e}\n\nCheck console for full traceback."


# --- CUSTOMTKINTER GUI APPLICATION ---

class SplineApp:
    def __init__(self, master):
        self.master = master
        master.title("Neuroglancer Spline Processor v4.3")
        # Let CustomTkinter determine initial size, adjust if needed
        # master.geometry("680x880")

        # Main frame using CTkFrame
        self.main_frame = customtkinter.CTkFrame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid layout weights for resizing rows containing frames
        self.main_frame.grid_rowconfigure(1, weight=0) # IO Frame row
        self.main_frame.grid_rowconfigure(3, weight=0) # Spline Frame row
        self.main_frame.grid_rowconfigure(5, weight=0) # Offset Frame row
        self.main_frame.grid_rowconfigure(7, weight=0) # Output Opts row
        self.main_frame.grid_rowconfigure(9, weight=0) # Seg Frame row
        self.main_frame.grid_rowconfigure(11, weight=0) # Viz Frame row
        self.main_frame.grid_rowconfigure(12, weight=1) # Spacer row before button? Or just rely on padding. Let's add button row config
        self.main_frame.grid_rowconfigure(13, weight=0) # Run button row

        self.main_frame.grid_columnconfigure(0, weight=1) # Allow frames to expand horizontally

        # Tkinter variables (remain the same)
        self.json_path = tk.StringVar()
        self.output_suffix = tk.StringVar(value="_processed.json") # Explicitly add .json
        self.skeleton_dir = tk.StringVar()
        self.segmentation_dir = tk.StringVar()
        self.viz_json_path = tk.StringVar()
        self.smoothing = tk.DoubleVar(value=1.0)
        self.sampling_factor = tk.DoubleVar(value=20.0)
        self.generate_offsets = tk.BooleanVar(value=True)
        self.radius = tk.DoubleVar(value=10.0) # Default radius for OFFSETS if r<num> not in layer name
        self.offset_step_sigma = tk.DoubleVar(value=0.3)
        self.offset_twist_sigma = tk.DoubleVar(value=0.1)
        self.n_offsets_factor = tk.DoubleVar(value=0.5)
        self.n_offsets_variance = tk.IntVar(value=2)
        self.min_offsets = tk.IntVar(value=1)
        self.max_offsets = tk.IntVar(value=20)
        self.resample_offsets = tk.BooleanVar(value=False)
        self.offset_resample_factor = tk.IntVar(value=5)
        self.post_offset_smoothing = tk.DoubleVar(value=0.5)
        self.annotation_mode = tk.StringVar(value="line")
        self.default_segment_id = tk.IntVar(value=100)
        self.generate_skeleton = tk.BooleanVar(value=False)
        self.generate_segmentation = tk.BooleanVar(value=False)
        self.voxel_size = tk.DoubleVar(value=1.0) # For segmentation output voxel grid
        self.modulate_radius = tk.BooleanVar(value=False) # For segmentation radius behavior
        self.radius_lower_bound = tk.DoubleVar(value=5.0) # Interpreted as DELTA if modulating
        self.radius_upper_bound = tk.DoubleVar(value=15.0)# Interpreted as DELTA if modulating
        self.radius_step_sigma = tk.DoubleVar(value=0.05)
        self.radius_smoothing_window = tk.IntVar(value=5)
        self.radius_distribution = tk.StringVar(value='uniform') # Only used if modulating (implicitly by compute_modulated_radius...)
        self.generate_viz_json = tk.BooleanVar(value=True)
        # *** NEW VARIABLES for new features ***
        self.treat_voxel_size_as_multiplier = tk.BooleanVar(value=False)
        self.use_zero_origin = tk.BooleanVar(value=False)
        self.order_by_distance = tk.BooleanVar(value=False)   # <- sort points NN-path


        # --- Input/Output Files ---
        # Use a CTkLabel placed *above* the CTkFrame for the title
        customtkinter.CTkLabel(self.main_frame, text="Files", font=customtkinter.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
        io_frame = customtkinter.CTkFrame(self.main_frame)
        io_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        io_frame.grid_columnconfigure(1, weight=1) # Allow entry field to expand

        customtkinter.CTkLabel(io_frame, text="Input JSON State:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(io_frame, textvariable=self.json_path).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        customtkinter.CTkButton(io_frame, text="Browse...", command=self.browse_json, width=80).grid(row=0, column=2, padx=(5,10), pady=5) # Add width hint

        customtkinter.CTkLabel(io_frame, text="Output JSON Suffix:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(io_frame, textvariable=self.output_suffix, width=150).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5) # Add width hint


        # --- Core Spline Params ---
        customtkinter.CTkLabel(self.main_frame, text="Main Spline Fitting", font=customtkinter.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", padx=10, pady=(10, 0))
        spline_frame = customtkinter.CTkFrame(self.main_frame)
        spline_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        # Give columns equal weight or configure specific weights if needed
        spline_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        customtkinter.CTkLabel(spline_frame, text="Smoothing (s):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(spline_frame, textvariable=self.smoothing, width=80).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkLabel(spline_frame, text="Sampling Factor:").grid(row=0, column=2, sticky=tk.W, padx=15, pady=5)
        customtkinter.CTkEntry(spline_frame, textvariable=self.sampling_factor, width=80).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)


        # --- Offset Generation Params ---
        customtkinter.CTkLabel(self.main_frame, text="Offset Spline Generation", font=customtkinter.CTkFont(weight="bold")).grid(row=4, column=0, sticky="w", padx=10, pady=(10, 0))
        offset_frame = customtkinter.CTkFrame(self.main_frame)
        offset_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        # Configure columns if needed, e.g.,ให้ labels and entries align well
        offset_frame.grid_columnconfigure((0, 2), weight=0) # Labels take minimum space
        offset_frame.grid_columnconfigure((1, 3), weight=1) # Entries expand

        customtkinter.CTkCheckBox(offset_frame, text="Generate Offset Splines", variable=self.generate_offsets).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        customtkinter.CTkLabel(offset_frame, text="Default Radius (if not r###):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.radius, width=80).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        customtkinter.CTkLabel(offset_frame, text="N Offsets Factor:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.n_offsets_factor, width=80).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkLabel(offset_frame, text="N Offsets Variance (+/-):").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.n_offsets_variance, width=80).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        customtkinter.CTkLabel(offset_frame, text="Min/Max Offsets:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        # Use a transparent CTkFrame for inline widgets
        min_max_frame = customtkinter.CTkFrame(offset_frame, fg_color="transparent")
        min_max_frame.grid(row=3, column=1, columnspan=3, sticky=tk.W)
        customtkinter.CTkEntry(min_max_frame, textvariable=self.min_offsets, width=50).pack(side=tk.LEFT, padx=(5,2), pady=2)
        customtkinter.CTkLabel(min_max_frame, text="/").pack(side=tk.LEFT, padx=0, pady=2)
        customtkinter.CTkEntry(min_max_frame, textvariable=self.max_offsets, width=50).pack(side=tk.LEFT, padx=(2,5), pady=2)

        customtkinter.CTkLabel(offset_frame, text="Offset Step Sigma:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.offset_step_sigma, width=80).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkLabel(offset_frame, text="Offset Twist Sigma:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.offset_twist_sigma, width=80).grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)

        # Use another transparent frame for the resample options
        resample_frame = customtkinter.CTkFrame(offset_frame, fg_color="transparent")
        resample_frame.grid(row=5, column=0, columnspan=4, sticky=tk.W, padx=0, pady=0)
        customtkinter.CTkCheckBox(resample_frame, text="Resample Offsets", variable=self.resample_offsets).pack(side=tk.LEFT, padx=(5,10), pady=5)
        customtkinter.CTkLabel(resample_frame, text="Factor:").pack(side=tk.LEFT, padx=5, pady=5)
        customtkinter.CTkEntry(resample_frame, textvariable=self.offset_resample_factor, width=50).pack(side=tk.LEFT, padx=5, pady=5)

        customtkinter.CTkLabel(offset_frame, text="Post-Offset Smoothing (s):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(offset_frame, textvariable=self.post_offset_smoothing, width=80).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        
        customtkinter.CTkCheckBox(
                spline_frame,
                text="Order points by distance",
                variable=self.order_by_distance
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        # --- Output Options (Annotations, Skeletons) ---
        customtkinter.CTkLabel(self.main_frame, text="Output Options", font=customtkinter.CTkFont(weight="bold")).grid(row=6, column=0, sticky="w", padx=10, pady=(10, 0))
        output_opts_frame = customtkinter.CTkFrame(self.main_frame)
        output_opts_frame.grid(row=7, column=0, sticky="ew", padx=5, pady=5)
        output_opts_frame.grid_columnconfigure(1, weight=1) # Allow entry field to expand

        customtkinter.CTkLabel(output_opts_frame, text="Annotation Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkComboBox(output_opts_frame, variable=self.annotation_mode, values=["point", "line"], width=100, state='readonly').grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        customtkinter.CTkLabel(output_opts_frame, text="Default Seg ID (if not ID###):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        customtkinter.CTkEntry(output_opts_frame, textvariable=self.default_segment_id, width=100).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        customtkinter.CTkCheckBox(output_opts_frame, text="Generate Skeletons", variable=self.generate_skeleton, command=self.toggle_skeleton_dir).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        # Skeleton Dir Widgets (initially potentially disabled)
        self.skel_dir_label = customtkinter.CTkLabel(output_opts_frame, text="Skeleton Output Dir:")
        self.skel_dir_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.skel_dir_entry = customtkinter.CTkEntry(output_opts_frame, textvariable=self.skeleton_dir)
        self.skel_dir_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.skel_dir_button = customtkinter.CTkButton(output_opts_frame, text="Browse...", command=self.browse_skeleton_dir, width=80)
        self.skel_dir_button.grid(row=3, column=2, padx=(5,10), pady=5)


        # --- Synthetic Segmentation Params ---
        customtkinter.CTkLabel(self.main_frame, text="Synthetic Segmentation", font=customtkinter.CTkFont(weight="bold")).grid(row=8, column=0, sticky="w", padx=10, pady=(10, 0))
        seg_frame = customtkinter.CTkFrame(self.main_frame)
        seg_frame.grid(row=9, column=0, sticky="ew", padx=5, pady=5)
        seg_frame.grid_columnconfigure(1, weight=1) # Allow entry field to expand
        seg_frame.grid_columnconfigure(3, weight=1) # Allow 2nd entry field to expand

        customtkinter.CTkCheckBox(seg_frame, text="Generate Segmentation Slices", variable=self.generate_segmentation, command=self.toggle_segmentation_options).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        # Segmentation Dir Widgets
        self.seg_dir_label = customtkinter.CTkLabel(seg_frame, text="Segmentation Output Dir:")
        self.seg_dir_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.seg_dir_entry = customtkinter.CTkEntry(seg_frame, textvariable=self.segmentation_dir)
        self.seg_dir_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5) # Span 2 cols
        self.seg_dir_button = customtkinter.CTkButton(seg_frame, text="Browse...", command=self.browse_segmentation_dir, width=80)
        self.seg_dir_button.grid(row=1, column=3, padx=(5,10), pady=5) # Place in last col

        # *** NEW: Voxel size multiplier checkbox and dynamic label ***
        self.seg_vox_multiplier_check = customtkinter.CTkCheckBox(seg_frame, text="Treat Voxel Size as Multiplier", variable=self.treat_voxel_size_as_multiplier, command=self.toggle_voxel_label)
        self.seg_vox_multiplier_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.seg_vox_label = customtkinter.CTkLabel(seg_frame, text="Voxel Size (X=Y=Z):") # Initial text
        self.seg_vox_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.seg_vox_entry = customtkinter.CTkEntry(seg_frame, textvariable=self.voxel_size, width=100)
        self.seg_vox_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # *** NEW: Zero origin checkbox ***
        self.seg_zero_origin_check = customtkinter.CTkCheckBox(seg_frame, text="Use [0,0,0] as Segmentation Origin", variable=self.use_zero_origin)
        self.seg_zero_origin_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self.seg_modulate_check = customtkinter.CTkCheckBox(seg_frame, text="Modulate Axon Radius Along Length", variable=self.modulate_radius, command=self.toggle_radius_modulation_options)
        self.seg_modulate_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Radius Modulation Widgets (initially potentially disabled)
        self.seg_radius_range_label = customtkinter.CTkLabel(seg_frame, text="Radius Delta Range (Min/Max):") # Clarified Label
        self.seg_radius_range_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        radius_range_frame = customtkinter.CTkFrame(seg_frame, fg_color="transparent")
        radius_range_frame.grid(row=5, column=1, columnspan=3, sticky=tk.W) # Span 3 cols
        self.seg_radius_lower_entry = customtkinter.CTkEntry(radius_range_frame, textvariable=self.radius_lower_bound, width=50)
        self.seg_radius_lower_entry.pack(side=tk.LEFT, padx=(5,2), pady=2)
        customtkinter.CTkLabel(radius_range_frame, text="/").pack(side=tk.LEFT, padx=0, pady=2)
        self.seg_radius_upper_entry = customtkinter.CTkEntry(radius_range_frame, textvariable=self.radius_upper_bound, width=50)
        self.seg_radius_upper_entry.pack(side=tk.LEFT, padx=(2,5), pady=2)

        self.seg_radius_step_label = customtkinter.CTkLabel(seg_frame, text="Radius Step Sigma:")
        self.seg_radius_step_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.seg_radius_step_entry = customtkinter.CTkEntry(seg_frame, textvariable=self.radius_step_sigma, width=100)
        self.seg_radius_step_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        self.seg_radius_smooth_label = customtkinter.CTkLabel(seg_frame, text="Smoothing Window:")
        self.seg_radius_smooth_label.grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)
        self.seg_radius_smooth_entry = customtkinter.CTkEntry(seg_frame, textvariable=self.radius_smoothing_window, width=50)
        self.seg_radius_smooth_entry.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)

        self.seg_radius_dist_label = customtkinter.CTkLabel(seg_frame, text="Base Radius Dist (Modulation):") # Clarified Label
        self.seg_radius_dist_label.grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.seg_radius_dist_combo = customtkinter.CTkComboBox(seg_frame, variable=self.radius_distribution, values=["uniform", "normal"], width=100, state='readonly')
        self.seg_radius_dist_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)


        # --- Visualization Params ---
        customtkinter.CTkLabel(self.main_frame, text="Visualization (three.js)", font=customtkinter.CTkFont(weight="bold")).grid(row=10, column=0, sticky="w", padx=10, pady=(10, 0))
        viz_frame = customtkinter.CTkFrame(self.main_frame)
        viz_frame.grid(row=11, column=0, sticky="ew", padx=5, pady=5)
        viz_frame.grid_columnconfigure(1, weight=1) # Allow entry field to expand

        customtkinter.CTkCheckBox(viz_frame, text="Export Splines for Visualization", variable=self.generate_viz_json, command=self.toggle_viz_json_path).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        # Viz Path Widgets
        self.viz_path_label = customtkinter.CTkLabel(viz_frame, text="Visualization JSON Path:")
        self.viz_path_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.viz_path_entry = customtkinter.CTkEntry(viz_frame, textvariable=self.viz_json_path)
        self.viz_path_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.viz_path_button = customtkinter.CTkButton(viz_frame, text="Browse...", command=self.browse_viz_json, width=80)
        self.viz_path_button.grid(row=1, column=2, padx=(5,10), pady=5)


        # --- Run Button ---
        # Place button at the bottom with some padding above
        run_button = customtkinter.CTkButton(self.main_frame, text="Run Processing", command=self.run_processing, height=35) # Increase height?
        run_button.grid(row=13, column=0, pady=(20, 10)) # Add padding top/bottom

        # Initial state update for optional fields
        self.toggle_skeleton_dir()
        self.toggle_segmentation_options()
        self.toggle_viz_json_path()


    # --- GUI Browse/Toggle Methods (No changes needed in logic) ---
    def browse_json(self):
        filepath = filedialog.askopenfilename(
            title="Select Neuroglancer JSON State File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.json_path.set(filepath)
            base, _ = os.path.splitext(filepath)
            # Auto-populate output directories if they are empty
            if not self.skeleton_dir.get(): self.skeleton_dir.set(base + "_skeletons")
            if not self.segmentation_dir.get(): self.segmentation_dir.set(base + "_segmentation_output") # Make distinct
            if not self.viz_json_path.get(): self.viz_json_path.set(base + "_viz_splines.json")
            # Auto-populate output json name based on input, if suffix is still default or empty
            current_suffix = self.output_suffix.get()
            if not current_suffix or current_suffix == "_processed.json":
                 self.output_suffix.set(os.path.basename(base) + "_processed.json")


    def browse_skeleton_dir(self):
        initial_dir_path = os.path.dirname(self.json_path.get()) if self.json_path.get() else os.getcwd()
        dirpath = filedialog.askdirectory(title="Select Skeleton Output Directory", initialdir=initial_dir_path)
        if dirpath: self.skeleton_dir.set(dirpath)

    def browse_segmentation_dir(self):
        initial_dir_path = os.path.dirname(self.json_path.get()) if self.json_path.get() else os.getcwd()
        dirpath = filedialog.askdirectory(title="Select Segmentation Output Directory", initialdir=initial_dir_path)
        if dirpath: self.segmentation_dir.set(dirpath)

    def browse_viz_json(self):
        # Suggest a filename based on input json
        suggested_name = "splines_viz.json"
        input_path = self.json_path.get()
        initial_dir_path = os.getcwd()
        if input_path:
            base, _ = os.path.splitext(os.path.basename(input_path))
            suggested_name = base + "_viz_splines.json"
            initial_dir_path = os.path.dirname(input_path)
        
        current_viz_path = self.viz_json_path.get()
        initial_file = os.path.basename(current_viz_path or suggested_name)
        initial_dir = os.path.dirname(current_viz_path or input_path or initial_dir_path)


        filepath = filedialog.asksaveasfilename(
            title="Select Output Path for Visualization JSON",
            initialfile=initial_file,
            initialdir=initial_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath: self.viz_json_path.set(filepath)

    def toggle_widget_state(self, widgets, state):
        """ Helper to set state for multiple CTk widgets. """
        ctk_state = "normal" if state == tk.NORMAL else "disabled"
        for widget in widgets:
             # Check if widget exists and has 'configure' method
             if widget and hasattr(widget, 'configure'):
                 try:
                    widget.configure(state=ctk_state)
                 except Exception as e:
                    print(f"Warning: Could not configure state for widget {widget}: {e}")

    def toggle_skeleton_dir(self):
        widgets = [self.skel_dir_label, self.skel_dir_entry, self.skel_dir_button]
        state = tk.NORMAL if self.generate_skeleton.get() else tk.DISABLED
        self.toggle_widget_state(widgets, state)

    def toggle_segmentation_options(self):
        """ Enable/disable all segmentation controls based on the main checkbox. """
        widgets = [
            self.seg_dir_label, self.seg_dir_entry, self.seg_dir_button,
            self.seg_vox_label, self.seg_vox_entry, self.seg_modulate_check,
            self.seg_vox_multiplier_check, self.seg_zero_origin_check # Also toggle new checkboxes
        ]
        main_state = tk.NORMAL if self.generate_segmentation.get() else tk.DISABLED
        self.toggle_widget_state(widgets, main_state)
        # Also update the dependent modulation options
        self.toggle_radius_modulation_options() # This must be called after main toggle
        self.toggle_voxel_label() # Update the voxel label text

    def toggle_radius_modulation_options(self):
        """ Enable/disable radius modulation specific controls. """
        widgets = [
            self.seg_radius_range_label, self.seg_radius_lower_entry, self.seg_radius_upper_entry,
            self.seg_radius_step_label, self.seg_radius_step_entry,
            self.seg_radius_smooth_label, self.seg_radius_smooth_entry,
            self.seg_radius_dist_label, self.seg_radius_dist_combo
        ]
        # Only enable if BOTH segmentation AND modulation are checked
        is_active = self.generate_segmentation.get() and self.modulate_radius.get()
        modulate_state = tk.NORMAL if is_active else tk.DISABLED
        self.toggle_widget_state(widgets, modulate_state)

    def toggle_viz_json_path(self):
        widgets = [self.viz_path_label, self.viz_path_entry, self.viz_path_button]
        state = tk.NORMAL if self.generate_viz_json.get() else tk.DISABLED
        self.toggle_widget_state(widgets, state)

    def toggle_voxel_label(self):
        """ *** NEW: Update the voxel size label based on the multiplier checkbox. *** """
        if self.treat_voxel_size_as_multiplier.get() and self.generate_segmentation.get():
            self.seg_vox_label.configure(text="Voxel Size Multiplier:")
        else:
            self.seg_vox_label.configure(text="Voxel Size (X=Y=Z):")


    # --- GUI Run Method (Calls the separated logic function) ---
    def run_processing(self):
        # Disable run button during processing? (Optional)
        # self.run_button.configure(state="disabled", text="Processing...")
        self.master.update_idletasks() # Update GUI immediately

        # 1. Get parameters from GUI
        input_json = self.json_path.get()
        if not input_json or not os.path.exists(input_json):
            messagebox.showerror("Error", "Please select a valid input JSON file.")
            # self.run_button.configure(state="normal", text="Run Processing")
            return

        # Ensure output suffix ends with .json
        output_suffix = self.output_suffix.get()
        if not output_suffix.lower().endswith('.json'):
            output_suffix += ".json"
            self.output_suffix.set(output_suffix) # Update GUI variable


        params = {
            'input_json': input_json,
            'output_suffix': output_suffix,
            'smoothing': self.smoothing.get(),
            'sampling_factor': self.sampling_factor.get(),
            'generate_offsets': self.generate_offsets.get(),
            'radius': self.radius.get(), # This is the default radius for OFFSETS if r<num> not in layer name
            'offset_step_sigma': self.offset_step_sigma.get(),
            'offset_twist_sigma': self.offset_twist_sigma.get(),
            'n_offsets_factor': self.n_offsets_factor.get(),
            'n_offsets_variance': self.n_offsets_variance.get(),
            'min_offsets': self.min_offsets.get(),
            'max_offsets': self.max_offsets.get(),
            'resample_offsets': self.resample_offsets.get(),
            'offset_resample_factor': self.offset_resample_factor.get(),
            'post_offset_smoothing': self.post_offset_smoothing.get(),
            'annotation_mode': self.annotation_mode.get(),
            'default_segment_id': self.default_segment_id.get(),
            'generate_skeleton': self.generate_skeleton.get(),
            'skeleton_output_dir': self.skeleton_dir.get().strip() if self.generate_skeleton.get() else None,
            'generate_segmentation': self.generate_segmentation.get(),
            'segmentation_output_dir': self.segmentation_dir.get().strip() if self.generate_segmentation.get() else None,
            'voxel_size': self.voxel_size.get(), # Voxel size for segmentation output grid
            'modulate_radius': self.modulate_radius.get(), # Boolean for segmentation radius behavior
            'radius_lower_bound': self.radius_lower_bound.get(), # Interpreted as DELTA if modulating radius
            'radius_upper_bound': self.radius_upper_bound.get(), # Interpreted as DELTA if modulating radius
            'radius_step_sigma': self.radius_step_sigma.get(),
            'radius_smoothing_window': self.radius_smoothing_window.get(),
            'radius_distribution': self.radius_distribution.get(), # Implicitly used by modulation logic
            'generate_viz_json': self.generate_viz_json.get(),
            'viz_json_path': self.viz_json_path.get().strip() if self.generate_viz_json.get() else None,
            # *** NEW PARAMETERS ***
            'treat_voxel_size_as_multiplier': self.treat_voxel_size_as_multiplier.get(),
            'use_zero_origin': self.use_zero_origin.get(),
            'order_by_distance': self.order_by_distance.get(),

        }

        # Validate required paths if options are enabled
        if params['generate_skeleton'] and not params['skeleton_output_dir']:
             messagebox.showerror("Error", "Skeleton output directory must be specified if 'Generate Skeletons' is checked.")
             # self.run_button.configure(state="normal", text="Run Processing")
             return
        if params['generate_segmentation'] and not params['segmentation_output_dir']:
             messagebox.showerror("Error", "Segmentation output directory must be specified if 'Generate Segmentation' is checked.")
             # self.run_button.configure(state="normal", text="Run Processing")
             return
        if params['generate_viz_json'] and not params['viz_json_path']:
             messagebox.showerror("Error", "Visualization JSON output path must be specified if 'Export Splines' is checked.")
             # self.run_button.configure(state="normal", text="Run Processing")
             return
        # Basic validation for numbers
        try:
             if params['voxel_size'] <= 0: raise ValueError("Voxel size must be positive.")
             if params['modulate_radius']: # Only validate bounds if modulating, as they are deltas
                 # Deltas can be negative, but lower_delta should not exceed upper_delta
                 if params['radius_lower_bound'] > params['radius_upper_bound']:
                     raise ValueError("Radius Delta Range: Min delta cannot be greater than Max delta.")
        except ValueError as ve:
             messagebox.showerror("Error", f"Invalid segmentation parameter: {ve}")
             # self.run_button.configure(state="normal", text="Run Processing")
             return

        # Call the processing logic function
        success, message = run_processing_logic(params)

        # Show result message box
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)

        # Re-enable run button
        # self.run_button.configure(state="normal", text="Run Processing")
        self.master.update_idletasks()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Set CustomTkinter appearance
    customtkinter.set_appearance_mode("System") # Options: "System", "Dark", "Light"
    customtkinter.set_default_color_theme("blue") # Options: "blue", "green", "dark-blue"

    # Create the main window using CTk
    root = customtkinter.CTk()

    app = SplineApp(root)
    root.mainloop()
