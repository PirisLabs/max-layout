"""Layer numbers, component kinds, and default parameter values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent

MODULE_DB_FILE = Path.home() / ".photonic_layout_editor_modules.json"

PHOTONIC_LAYER = 1

GC_LAYER = 2

MARKER_LAYER = 3

RF_LAYER = 4

PROBE_LAYER = 5

EBEAM_LAYER = 6

DEFAULT_DATATYPE = 0

SIMULATION_LAYER = 0

LAYER_NAME_MAP = {
    SIMULATION_LAYER: "Simulation only (not GDS)",
    PHOTONIC_LAYER: "WG",
    GC_LAYER: "GC",
    MARKER_LAYER: "Marker",
    RF_LAYER: "RF",
    PROBE_LAYER: "Probe",
    EBEAM_LAYER: "Ebeam",
}

SIMULATION_COMPONENT_KINDS = {
    "FDTD port",
    "Fiber-axis FDTD port",
    "Fiber geometry",
    # Legacy combined object: older projects still load, but it is hidden from
    # the library and exports as geometry only so it never creates a port.
    "Fiber port",
    "Power monitor",
    "Mode expansion monitor",
    "Field profile monitor",
    # RF-only sampling planes.  These remain visible/movable in the layout
    # editor but are metadata for the dedicated CPW MODE/FDTD exporters and
    # never become GDS polygons.
    "RF mode port",
    "RF power monitor",
}

RF_COMPONENT_KINDS = {
    "CPW",
    "CPW open",
    "CPW short",
    "Tapered CPW",
    "Symmetric CPW taper",
    "CPW bend",
    "Segmented electrode",
    "RF test block",
}

# These generators remain available so older JSON projects still load, but
# they are hidden from the component library in favor of the generic wizard.
LEGACY_PHOTONIC_TEST_BLOCK_KINDS = {
    "Long MZI test block",
    "Double-ring test block",
    "Grating test block",
    "Grating angle-taper test block",
    "MMI + Reference test block",
    "MMI split-combine test block",
    "Vertical-GC MZI test block",
    "Vertical-GC MZI + CPW test block",
    "Vertical-GC MZI + segmented electrode test block",
    "Straight-GC MZI + segmented RF bends test block",
    "Straight-GC MZI + CPW RF bends test block",
}

MARKER_COMPONENT_KINDS = {
    "Square mark",
    "Cross mark",
    "Pointy cross mark",
    "Cross + squares mark",
    "Vernier mark",
    "Text / Number",
    "Chip marker block",
}

GC_COMPOSITE_KINDS = {
    "1x2 MMI",
    "Cascaded MMI",
    "MMI split-combine cascade",
    "MMI split-combine test block",
    "MMI + Reference",
    "MZI",
    "MZI vertical GC",
    "Vertical-GC MZI test block",
    "Vertical-GC MZI + CPW test block",
    "Vertical-GC MZI + segmented electrode test block",
    "Straight-GC MZI + segmented RF bends test block",
    "Straight-GC MZI + CPW RF bends test block",
    "Long MZI test block",
    "Ring + two feedlines",
    "Double-ring test block",
    "Grating test block",
    "Grating angle-taper test block",
    "MMI + Reference test block",
    "Feedline",
    "Ring + feedline",
    "Racetrack + feedline",
}

DEFAULT_COMPONENT_VALUES = {'Straight': {'length': 50.0, 'width': 1.2, 'layer': 1, 'datatype': 0},
 'Taper': {'length': 30.0, 'width_start': 1.2, 'width_end': 2.5, 'layer': 1, 'datatype': 0},
 'S-bend': {'length': 50.0, 'offset': 20.0, 'width': 1.2, 'layer': 1, 'datatype': 0, 'tolerance': 0.001},
 'Euler bend': {'radius': 200.0, 'bend_angle_deg': 90.0, 'width': 1.2, 'euler_fraction': 1.0, 'layer': 1, 'datatype': 0, 'tolerance': 0.001},
 'Grating coupler': {'pitch': 0.8,
                     'fill_factor': 0.5,
                     'fill_factors': '',
                     'tooth_shape': 'curved',
                     'N': 30,
                     'alpha_t': 25.0,
                     'taper_L': 22.0,
                     'L_extra': 10.0,
                     'wg_width': 1.2,
                     'wg_length': 5.0,
                     # Signed local-X distance from the geometry-exact flare
                     # boundary to the fiber bottom center.  The boundary is
                     # wg_length - focus_offset + taper_L.
                     'fiber_offset': 5.0,
                     # One parent-level tilt drives the fiber core/cladding,
                     # source port, and passive fiber-power plane together.
                     'angle_theta': 7.0,
                     'fiber_power_monitor_below_source_um': 0.1,
                     'fdtd_port_offset_from_waveguide_end_um': 2.0,
                     # Automatic TFLN waveguide planes use at least 3 um and
                     # at least twice the physical waveguide width.
                     'waveguide_monitor_span_um': 3.0,
                     'waveguide_total_power_before_mode_um': 1.0,
                     # The solver derives its mode target from the active
                     # dispersive core and adjacent dielectric indices.
                     'waveguide_neff_tolerance': 0.3,
                     'waveguide_mode_search_count': 20,
                     'layer': 2,
                     'datatype': 0,
                     'tolerance': 0.0005,
                     'waveguide_layer': 1,
                     'waveguide_datatype': 0},
 '1x2 MMI': {'mmi_width': 6.0,
             'mmi_length': 29.0,
             'wg_width': 1.2,
             'taper_width': 2.7,
             'input_taper_length': 10.0,
             'output_taper_length': 10.0,
             'input_length': 6.0,
             'input_reference_before_taper_um': 2.0,
             'fdtd_port_clearance_um': 2.0,
             # All three ports share one stack-derived modal target.
             'waveguide_neff_tolerance': 0.3,
             'waveguide_mode_search_count': 20,
             'output_length': 6.0,
             'port_sep': 3.25,
             'taper_power': 1.0,
             'taper_points': 41,
             'add_grating_couplers': True,
             'gc_input_route': 'straight',
             'gc_upper_output_route': 'straight',
             'gc_lower_output_route': 'straight',
             'gc_output_separation': 133.0,
             'gc_s_bend_length': 80.0,
             'gc_euler_radius': 200.0,
             'gc_euler_fraction': 1.0,
             'gc_pitch': 0.75,
             'gc_fill_factor': 0.57,
             'gc_N': 30,
             'gc_alpha_t': 25.0,
             'gc_taper_L': 22.0,
             'gc_wg_length': 20.0,
             'gc_tolerance': 0.0005,
             'gc_euler_tolerance': 0.001,
             'layer': 1,
             'datatype': 0,
             'gc_layer': 2,
             'gc_datatype': 0},
 'Cascaded MMI': {'N_levels': 3,
                  's_bend_length': 300.0,
                  'minimum_s_bend_radius': 200.0,
                  'output_s_bend_length': 300.0,
                  'output_gc_spacing': 150.0,
                  'mmi_width': 6.0,
                  'mmi_length': 29.0,
                  'wg_width': 1.2,
                  'taper_width': 2.7,
                  'input_taper_length': 10.0,
                  'output_taper_length': 10.0,
                  'input_length': 6.0,
                  'output_length': 6.0,
                  'port_sep': 3.25,
                  'taper_power': 1.0,
                  'taper_points': 41,
                  'add_input_grating_coupler': False,
                  'add_output_grating_coupler': True,
                  'gc_route': 'straight',
                  'gc_euler_radius': 200.0,
                  'gc_euler_fraction': 1.0,
                  'gc_pitch': 0.75,
                  'gc_fill_factor': 0.57,
                  'gc_N': 30,
                  'gc_alpha_t': 25.0,
                  'gc_taper_L': 22.0,
                  'gc_wg_length': 20.0,
                  'gc_tolerance': 0.0005,
                  'gc_euler_tolerance': 0.001,
                  'layer': 1,
                  'datatype': 0,
                  'gc_layer': 2,
                  'gc_datatype': 0},
 'MMI + Reference': {'mmi_width': 6.0,
             'mmi_length': 29.0,
             'wg_width': 1.2,
             'taper_width': 2.7,
             'input_taper_length': 10.0,
             'output_taper_length': 10.0,
             'input_length': 6.0,
             'output_length': 6.0,
             'port_sep': 3.25,
             'taper_power': 1.0,
             'taper_points': 41,
             'add_grating_couplers': True,
             'gc_input_route': 'straight',
             'gc_upper_output_route': 'straight',
             'gc_lower_output_route': 'straight',
             'gc_output_separation': 133.0,
             'gc_s_bend_length': 80.0,
             'gc_euler_radius': 200.0,
             'gc_euler_fraction': 1.0,
             'gc_pitch': 0.75,
             'gc_fill_factor': 0.57,
             'gc_N': 30,
             'gc_alpha_t': 25.0,
             'gc_taper_L': 22.0,
             'gc_wg_length': 20.0,
             'gc_tolerance': 0.0005,
             'gc_euler_tolerance': 0.001,
             'reference_dy': 250.0,
             'reference_branch': 'upper',
             'layer': 1,
             'datatype': 0,
             'gc_layer': 2,
             'gc_datatype': 0},
 'MZI': {'mmi_width': 6.0,
         'mmi_length': 29.0,
         'wg_width': 1.2,
         'taper_width': 2.7,
         'input_taper_length': 10.0,
         'output_taper_length': 10.0,
         'input_length': 6.0,
         'output_length': 6.0,
         'port_sep': 3.25,
         'arm_separation': 133.0,
         's_bend_length': 80.0,
         'arm_length': 9718.0,
         'taper_power': 1.0,
         'taper_points': 41,
         'add_grating_couplers': True,
         'gc_input_route': 'straight',
         'gc_output_route': 'straight',
         'gc_euler_radius': 200.0,
         'gc_euler_fraction': 1.0,
         'gc_pitch': 0.75,
         'gc_fill_factor': 0.57,
         'gc_N': 30,
         'gc_alpha_t': 25.0,
         'gc_taper_L': 22.0,
         'gc_wg_length': 20.0,
         'gc_tolerance': 0.0005,
         'gc_euler_tolerance': 0.001,
         'layer': 1,
         'datatype': 0,
         'gc_layer': 2,
         'gc_datatype': 0},
 'Long MZI test block': {'mzi_total_length': 10000.0,
                         'mzi_count': 5,
                         'vertical_spacing': 1000.0,
                         'include_ebeam_fields': True,
                         'ebeam_field_size': 520.0,
                         'ebeam_edge_clearance': 10.0,
                         'parameter_text_height': 10.0,
                         'gc_straight_length': 200.0,
                         'mmi_width': 6.0,
                         'mmi_length': 29.0,
                         'wg_width': 1.2,
                         'taper_width': 2.7,
                         'input_taper_length': 10.0,
                         'output_taper_length': 10.0,
                         'input_length': 6.0,
                         'output_length': 6.0,
                         'port_sep': 3.25,
                         'arm_separation': 133.0,
                         's_bend_length': 200.0,
                         'taper_power': 1.0,
                         'taper_points': 41,
                         'gc_input_route': 'straight',
                         'gc_output_route': 'straight',
                         'gc_euler_radius': 200.0,
                         'gc_euler_fraction': 1.0,
                         'gc_pitch': 0.75,
                         'gc_fill_factor': 0.57,
                         'gc_N': 30,
                         'gc_alpha_t': 25.0,
                         'gc_taper_L': 22.0,
                         'gc_wg_length': 20.0,
                         'gc_tolerance': 0.0005,
                         'gc_euler_tolerance': 0.001,
                         'layer': 1,
                         'datatype': 0,
                         'gc_layer': 2,
                         'gc_datatype': 0},
 'MZI + CPW module': {'mmi_width': 6.0,
                      'mmi_length': 29.0,
                      'wg_width': 1.2,
                      'taper_width': 2.7,
                      'input_taper_length': 10.0,
                      'output_taper_length': 10.0,
                      'input_length': 6.0,
                      'output_length': 6.0,
                      'port_sep': 3.25,
                      'arm_separation': 133.0,
                      's_bend_length': 80.0,
                      'arm_length': 9718.0,
                      'taper_power': 1.0,
                      'taper_points': 41,
                      'signal_width': 130.0,
                      'interaction_gap': 3.0,
                      'external_gap': 14.5,
                      'ground_width': 130.0,
                      'rf_input_taper_length': 300.0,
                      'rf_output_taper_length': 300.0,
                      'rf_input_taper_profile': 'klopfenstein',
                      'rf_output_taper_profile': 'klopfenstein',
                      'target_s11_db': 20.0,
                      'exponential_factor': 1.0,
                      'points': 161,
                      'layer': 1,
                      'datatype': 0,
                      'rf_layer': 4,
                      'rf_datatype': 0},
 'CPW': {'length': 500.0, 'signal_width': 130.0, 'gap': 3.0, 'ground_width': 130.0, 'layer': 4, 'datatype': 0},
 'CPW open': {'length': 150.0, 'signal_width': 130.0, 'gap': 3.0, 'ground_width': 130.0, 'signal_recess': 20.0, 'layer': 4, 'datatype': 0},
 'CPW short': {'length': 150.0, 'signal_width': 130.0, 'gap': 3.0, 'ground_width': 130.0, 'bridge_length': 20.0, 'layer': 4, 'datatype': 0},
 'Tapered CPW': {'length': 500.0,
                 'signal_width': 130.0,
                 'initial_gap': 3.0,
                 'final_gap': 14.5,
                 'ground_width': 130.0,
                 'profile': 'klopfenstein',
                 'target_s11_db': 20.0,
                 'exponential_factor': 1.0,
                 'points': 1001,
                 'layer': 4,
                 'datatype': 0},
 'Symmetric CPW taper': {'end_straight_length': 50.0,
                         'taper_length': 300.0,
                         'middle_straight_length': 50.0,
                         'signal_width': 130.0,
                         'initial_gap': 14.5,
                         'middle_gap': 3.0,
                         'ground_width': 130.0,
                         'profile': 'klopfenstein',
                         'target_s11_db': 20.0,
                         'exponential_factor': 1.0,
                         'points': 601,
                         'layer': 4,
                         'datatype': 0},
 'CPW bend': {'R_eff': 500.0, 'bend_angle_deg': 90.0, 'signal_width': 130.0, 'gap': 3.0, 'ground_width': 130.0, 'points': 161, 'layer': 4, 'datatype': 0},
 'Segmented electrode': {'signal_width': 130.0,
                         'gap': 3.0,
                         'end_gap': 3.0,
                         'ground_width': 130.0,
                         'transition_length': 0.0,
                         'end_flat_length': 100.0,
                         'inner_flat_length': 0.0,
                         't_top_width': 2.0,
                         't_top_length': 45.0,
                         't_neck_width': 4.0,
                         't_neck_length': 18.0,
                         'segment_spacing': 3.0,
                         'segment_count': 80,
                         'include_oxide_masks': False,
                         'layer': 4,
                         'datatype': 0,
                         'oxide_layer': 4,
                         'oxide_datatype': 0},
 'Chip outline': {'width': 14000.0, 'height': 12000.0, 'line_width': 10.0, 'show_dimensions': 1, 'dimension_offset': 150.0, 'dimension_text_scale': 1.0, 'layer': 100, 'datatype': 0, 'dimension_layer': 101},
 '4-inch wafer outline': {'diameter': 100000.0, 'primary_flat_length': 32500.0, 'line_width': 100.0, 'points': 720, 'filled': 0, 'layer': 100, 'datatype': 0},
 'E-beam multipass': {'field_size': 520.0,
                       'overlap_x_enabled': False,
                       'overlap_y_enabled': False,
                       'overlap_x_percent': 0.0,
                       'overlap_y_percent': 0.0,
                       'edge_clearance': 10.0,
                       'target_width': 1000.0,
                       'target_height': 1000.0,
                       'start_corner': 'top-left',
                       'primary_axis': 'x',
                       'serpentine': True,
                       'preserve_manual_grid_position': True,
                       'manual_layout_locked': False,
                       'show_order': True,
                       'outline_width': 1.0,
                       'field_layer': 6,
                       'field_datatype': 0,
                       'beamer_wg_dose': 1.8,
                       'beamer_gc_dose': 1.8,
                       'beamer_rf_dose': 1.8,
                       'beamer_probe_dose': 1.8,
                       'beamer_marker_dose': 1.8,
                       'beamer_marker_layers': '3',
                       'beamer_region_layer': 6},
 'Square mark': {'size': 100.0, 'line_width': 10.0, 'filled': 0, 'layer': 3, 'datatype': 0},
 'Cross mark': {'size': 100.0, 'line_width': 10.0, 'layer': 3, 'datatype': 0},
 'Pointy cross mark': {'size': 120.0, 'line_width': 10.0, 'tip_length': 18.0, 'layer': 3, 'datatype': 0},
 'Cross + squares mark': {'size': 140.0, 'bar_width': 10.0, 'square_size': 34.0, 'square_gap': 8.0, 'layer': 3, 'datatype': 0},
 'Vernier mark': {'finger_count': 11, 'finger_width': 2.0, 'finger_length': 40.0, 'pitch': 10.0, 'pitch_delta': 0.2, 'row_gap': 8.0, 'base_thickness': 8.0, 'layer_a': 3, 'layer_b': 3, 'datatype': 0},
 'Chip marker block': {'chip_width': 14000.0,
                       'chip_height': 12000.0,
                       'corner_square_size': 50.0,
                       'corner_label_height': 14.0,
                       'edge_clearance': 0.0,
                       'side_mark_count': 3,
                       'side_mark_size': 100.0,
                       'side_mark_bar_width': 8.0,
                       'side_mark_square_size': 18.0,
                       'side_mark_square_gap': 5.0,
                       'side_solid_square_count': 2,
                       'side_solid_square_size': 50.0,
                       'side_solid_square_interval': 50.0,
                       'include_center_vernier': True,
                       'bottom_vernier_label_height': 18.0,
                       'vernier_finger_count': 11,
                       'vernier_finger_width': 2.0,
                       'vernier_finger_length': 40.0,
                       'vernier_pitch': 10.0,
                       'vernier_pitch_delta': 0.2,
                       'vernier_row_gap': 8.0,
                       'vernier_base_thickness': 8.0,
                       'layer': 3,
                       'datatype': 0},
 'Ring': {'radius': 200.0, 'width': 1.2, 'points': 256, 'layer': 1, 'datatype': 0},
 'Photonic crystal': {'length': 50.0,
                           'width': 20.0,
                           'slab_shape': 'rectangle',
                           'device_type': 'bulk crystal',
                           'mask_tone': 'negative: slab minus holes',
                           'lattice': 'triangular',
                           'columns': 120,
                           'rows': 24,
                           'pitch_x': 0.42,
                           'pitch_y': 0.3637306696,
                           'hole_shape': 'circular',
                           'hole_radius_x': 0.12,
                           'hole_radius_y': 0.12,
                           'defect_rows': 1,
                           'points': 48,
                           'layer': 1,
                           'datatype': 0,
                           'hole_layer': 3,
                           'hole_datatype': 0},
 'Boolean geometry': {'polygons': [], 'operation': 'union', 'layer': 1, 'datatype': 0},
 'Elliptical ring': {'radius_x': 200.0, 'radius_y': 100.0, 'width': 1.2, 'points': 256, 'layer': 1, 'datatype': 0},
 'Racetrack': {'radius': 30.0, 'coupling_length': 100.0, 'width': 1.2, 'points': 128, 'layer': 1, 'datatype': 0},
 'Ring + two feedlines': {'pitch': 0.75,
                          'fill_factor': 0.57,
                          'N': 30,
                          'alpha_t': 25.0,
                          'taper_L': 22.0,
                          'gc_wg_length': 20.0,
                          'gc_tolerance': 0.0005,
                          'ring_radius': 200.0,
                          'ring_width': 1.2,
                          'feedline_length': 500.0,
                          'feedline_width': 1.2,
                          'coupling_gap': 0.3,
                          'grating_coupler_separation': 150.0,
                          's_bend_length': 80.0,
                          's_bend_offset': 50.0,
                          'input_s_bend_direction': 'down',
                          'output_s_bend_direction': 'up',
                          'points': 256,
                          'layer': 1,
                          'datatype': 0,
                          'tolerance': 0.001,
                          'gc_layer': 2,
                          'gc_datatype': 0},
 'Double-ring test block': {'gap_values': '0.5, 0.6, 0.7, 0.8, 0.9, 1.0',
                            'parameter_text_height': 10.0,
                            'radius_values': '20, 50, 100, 200',
                            'column_spacing': 600.0,
                            'row_spacing': 600.0,
                            'ebeam_field_size': 520.0,
                            'ebeam_edge_clearance': 10.0,
                            'include_ebeam_fields': True,
                            'grating_end_to_end_distance': 365.0,
                            'feedline_width': 1.2,
                            'ring_width': 1.2,
                            's_bend_length': 80.0,
                            's_bend_offset': 20.0,
                            'input_s_bend_direction': 'up',
                            'output_s_bend_direction': 'up',
                            'pitch': 0.75,
                            'fill_factor': 0.57,
                            'N': 30,
                            'alpha_t': 25.0,
                            'taper_L': 22.0,
                            'gc_wg_length': 20.0,
                            'layer': 1,
                            'datatype': 0,
                            'gc_layer': 2,
                            'gc_datatype': 0},
 'Grating test block': {'pitch_start': 0.735,
                        'pitch_stop': 0.765,
                        'pitch_step': 0.005,
                        'fill_start': 0.52,
                        'fill_stop': 0.62,
                        'fill_step': 0.025,
                        'nominal_pitch': 0.75,
                        'nominal_fill': 0.57,
                        'max_devices': 36,
                        'parameter_text_height': 10.0,
                        'grating_end_to_end_distance': 365.0,
                        'endpoint_offset': 100.0,
                        'device_x_spacing': 600.0,
                        'packing_pitch': 100.0,
                        'devices_per_field': 4,
                        'write_field_size': 520.0,
                        'ebeam_edge_clearance': 10.0,
                        'include_ebeam_fields': True,
                        'wg_width': 1.2,
                        's_bend_length': 80.0,
                        'N': 30,
                        'alpha_t': 25.0,
                        'taper_L': 22.0,
                        'gc_wg_length': 20.0,
                        'layer': 1,
                        'datatype': 0,
                        'gc_layer': 2,
                        'gc_datatype': 0},
 'Grating angle-taper test block': {'angle_start_deg': 22.0,
                                    'angle_stop_deg': 28.0,
                                    'angle_step_deg': 1.0,
                                    'taper_length_start': 20.0,
                                    'taper_length_stop': 24.0,
                                    'taper_length_step': 1.0,
                                    'nominal_angle_deg': 25.0,
                                    'nominal_taper_length': 22.0,
                                    'max_devices': 36,
                                    'parameter_text_height': 10.0,
                                    'pitch': 0.75,
                                    'fill_factor': 0.57,
                                    'grating_end_to_end_distance': 365.0,
                                    'endpoint_offset': 100.0,
                                    'device_x_spacing': 600.0,
                                    'packing_pitch': 100.0,
                                    'devices_per_field': 4,
                                    'write_field_size': 520.0,
                                    'ebeam_edge_clearance': 10.0,
                                    'include_ebeam_fields': True,
                                    'wg_width': 1.2,
                                    's_bend_length': 80.0,
                                    'N': 30,
                                    'gc_wg_length': 20.0,
                                    'layer': 1,
                                    'datatype': 0,
                                    'gc_layer': 2,
                                    'gc_datatype': 0},
 'MMI + Reference test block': {'mmi_length_start': 26.0,
                                'parameter_text_height': 10.0,
                                'mmi_length_stop': 33.0,
                                'mmi_length_step': 1.0,
                                'taper_width_start': 2.5,
                                'taper_width_stop': 3.1,
                                'taper_width_step': 0.1,
                                'nominal_taper_width': 2.7,
                                'device_x_spacing': 600.0,
                                'device_y_spacing': 600.0,
                                'ebeam_field_size': 520.0,
                                'ebeam_edge_clearance': 10.0,
                                'include_ebeam_fields': True,
                                'mmi_width': 7.0,
                                'wg_width': 1.2,
                                'input_taper_length': 10.0,
                                'output_taper_length': 10.0,
                                'input_length': 6.0,
                                'output_length': 6.0,
                                'port_sep': 3.25,
                                'reference_dy': 150.0,
                                'reference_branch': 'upper',
                                'pitch': 0.75,
                                'fill_factor': 0.57,
                                'N': 30,
                                'alpha_t': 25.0,
                                'taper_L': 22.0,
                                'gc_wg_length': 20.0,
                                'layer': 1,
                                'datatype': 0,
                                'gc_layer': 2,
                                'gc_datatype': 0},
 'Text / Number': {'text': 'TEXT', 'height': 50.0, 'layer': 3, 'datatype': 0},
 'Edge coupler': {'tip_width': 0.2, 'wg_width': 1.2, 'taper_length': 200.0, 'wg_straight_length': 50.0, 'layer': 1, 'datatype': 0},
 'Loopback mirror': {'Lc': 200.0, 'gap': 10.0, 's_bend_length': 80.0, 'arc_radius': 30.0, 'width': 1.2, 'layer': 1, 'datatype': 0, 'tolerance': 0.001},
 'Feedline': {'pitch': 0.75,
              'fill_factor': 0.57,
              'N': 30,
              'alpha_t': 25.0,
              'taper_L': 22.0,
              'wg_width': 1.2,
              'gc_wg_length': 20.0,
              'input_straight_length': 50.0,
              's_bend_length': 80.0,
              'offset': 50.0,
              'Lc': 3000.0,
              'output_straight_length': 50.0,
              'layer': 1,
              'datatype': 0,
              'tolerance': 0.001,
              'gc_tolerance': 0.0005,
              'add_grating_couplers': True,
              'input_s_bend_direction': 'up',
              'output_s_bend_direction': 'down',
              'gc_layer': 2,
              'gc_datatype': 0},
 'Ring + feedline': {'pitch': 0.75,
                     'fill_factor': 0.57,
                     'N': 30,
                     'alpha_t': 25.0,
                     'taper_L': 22.0,
                     'wg_width': 1.2,
                     'gc_wg_length': 20.0,
                     'input_straight_length': 50.0,
                     's_bend_length': 80.0,
                     'offset': 50.0,
                     'Lc': 3000.0,
                     'output_straight_length': 50.0,
                     'ring_radius': 200.0,
                     'ring_width': 1.2,
                     'coupling_gap': 0.3,
                     'resonator_count': 1,
                     'resonator_spacing': 120.0,
                     'resonator_side': 'upper',
                     'resonator_layer': 1,
                     'resonator_datatype': 0,
                     'layer': 1,
                     'datatype': 0,
                     'tolerance': 0.001,
                     'gc_tolerance': 0.0005,
                     'add_grating_couplers': True,
                     'input_s_bend_direction': 'up',
                     'output_s_bend_direction': 'down',
                     'gc_layer': 2,
                     'gc_datatype': 0},
 'Racetrack + feedline': {'pitch': 0.75,
                          'fill_factor': 0.57,
                          'N': 30,
                          'alpha_t': 25.0,
                          'taper_L': 22.0,
                          'wg_width': 1.2,
                          'gc_wg_length': 20.0,
                          'input_straight_length': 50.0,
                          's_bend_length': 80.0,
                          'offset': 50.0,
                          'Lc': 3000.0,
                          'output_straight_length': 50.0,
                          'racetrack_radius': 200.0,
                          'racetrack_length': 4000.0,
                          'racetrack_width': 1.2,
                          'coupling_gap': 0.3,
                          'resonator_count': 1,
                          'resonator_spacing': 180.0,
                          'resonator_side': 'upper',
                          'resonator_layer': 1,
                          'resonator_datatype': 0,
                          'layer': 1,
                          'datatype': 0,
                          'tolerance': 0.001,
                          'gc_tolerance': 0.0005,
                          'add_grating_couplers': True,
                          'input_s_bend_direction': 'up',
                          'output_s_bend_direction': 'down',
                          'gc_layer': 2,
                          'gc_datatype': 0}}

INTEGER_PARAMETERS = {"N", "gc_N", "mzi_count", "columns", "rows", "defect_rows", "layer", "datatype", "hole_layer", "hole_datatype", "gc_layer", "gc_datatype", "waveguide_layer", "waveguide_datatype", "taper_points", "points", "show_dimensions", "dimension_layer", "filled", "finger_count", "layer_a", "layer_b", "resonator_count", "resonator_layer", "resonator_datatype", "segment_count", "oxide_layer", "oxide_datatype", "rf_layer", "rf_datatype", "field_layer", "field_datatype", "beamer_region_layer"}

BOOL_PARAMETERS = {"add_grating_couplers", "include_oxide_masks", "include_ebeam_fields", "include_symmetric_cpw", "gc_three_euler_inward", "gc_align_gc_to_mzi_center", "overlap_x_enabled", "overlap_y_enabled", "serpentine", "show_order"}

DEFAULT_COMPONENT_VALUES["MZI vertical GC"] = dict(DEFAULT_COMPONENT_VALUES["MZI"])

DEFAULT_COMPONENT_VALUES["MZI vertical GC"].update({"add_grating_couplers":True,"gc_three_euler_inward":True,"gc_prebend_straight":10.0,"gc_vertical_side":"up","gc_vertical_run":100.0,"gc_inward_run_fraction":0.25,"gc_align_gc_to_mzi_center":True,"gc_inward_run":300.0,"gc_euler_radius":100.0})

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI test block"] = dict(DEFAULT_COMPONENT_VALUES["MZI vertical GC"])

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI test block"].update({"mzi_count":5,"mmi_length_start":25.0,"mmi_length_step":2.0,"mzi_total_length":10000.0,"vertical_spacing":1800.0,"include_ebeam_fields":True,"ebeam_field_size":520.0,"ebeam_edge_clearance":10.0,"parameter_text_height":10.0})

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + CPW test block"] = dict(DEFAULT_COMPONENT_VALUES["Vertical-GC MZI test block"])

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + CPW test block"].update({"include_symmetric_cpw":True,"vertical_spacing":1000.0,"cpw_align_to_mzi_s_bends":True,"cpw_s_bend_clearance":10.0,"cpw_middle_flat_fraction":.95,"cpw_taper_length":500.0,"cpw_outer_flat_length":500.0,"cpw_signal_width":130.0,"cpw_ground_width":130.0,"cpw_end_gap":14.5,"cpw_middle_gap":3.0,"cpw_profile":"klopfenstein","cpw_target_s11_db":20.0,"cpw_exponential_factor":1.0,"cpw_points":1001})

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + CPW test block"].pop("cpw_middle_flat_fraction",None)

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + CPW test block"].pop("cpw_outer_flat_length",None)

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + segmented electrode test block"] = dict(DEFAULT_COMPONENT_VALUES["Vertical-GC MZI test block"])

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + segmented electrode test block"].update({"include_segmented_electrode":True,"vertical_spacing":1000.0,"seg_s_bend_clearance":10.0,"seg_signal_width":130.0,"seg_end_gap":3.0,"seg_gap":3.0,"seg_ground_width":130.0,"seg_transition_length":1.0,"seg_end_flat_length":50.0,"seg_inner_flat_length":50.0,"seg_t_top_width":2.0,"seg_t_top_length":45.0,"seg_t_neck_width":4.0,"seg_t_neck_length":18.0,"seg_segment_spacing":3.0,"seg_segment_count":80,"seg_auto_segment_count":True,"seg_include_oxide_masks":False})

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + segmented electrode test block"]["seg_taper_length"]=1.0

DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + segmented electrode test block"].pop("seg_transition_length",None)

DEFAULT_COMPONENT_VALUES["Straight-GC MZI + segmented RF bends test block"] = dict(DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + segmented electrode test block"])

DEFAULT_COMPONENT_VALUES["Straight-GC MZI + segmented RF bends test block"].update({"gc_three_euler_inward":False,"gc_input_route":"straight","gc_output_route":"straight","gc_wg_length":3000.0,"mzi_total_length":8000.0,"vertical_spacing":1000.0,"seg_s_bend_clearance":0.0,"seg_end_flat_length":0.0,"seg_inner_flat_length":200.0,"include_rf_edge_bends":True,"rf_edge_straight_length":0.0,"rf_edge_bend_radius":500.0,"rf_edge_bend_angle_deg":90.0,"rf_edge_bend_points":321})

DEFAULT_COMPONENT_VALUES["Straight-GC MZI + CPW RF bends test block"] = dict(DEFAULT_COMPONENT_VALUES["Vertical-GC MZI + CPW test block"])

DEFAULT_COMPONENT_VALUES["Straight-GC MZI + CPW RF bends test block"].update({"gc_three_euler_inward":False,"gc_input_route":"straight","gc_output_route":"straight","gc_wg_length":3000.0,"mzi_total_length":8000.0,"vertical_spacing":1000.0,"cpw_s_bend_clearance":0.0,"cpw_end_straight_length":0.0,"include_rf_edge_bends":True,"rf_edge_bend_radius":500.0,"rf_edge_bend_angle_deg":90.0,"rf_edge_bend_points":321})

DEFAULT_COMPONENT_VALUES["RF test block"] = {
    "rf_component_kind": "CPW",
    "rf_base_params": dict(DEFAULT_COMPONENT_VALUES["CPW"]),
    "device_label_prefix": "CPW",
    "label_height": 20.0,
    "label_offset_x": 0.0,
    "label_offset_y": 10.0,
    "label_layer": MARKER_LAYER,
    "label_datatype": DEFAULT_DATATYPE,
    "taper_test_structure": False,
    "taper_test_center": "CPW",
    "probe_cpw_length": 100.0,
    "input_transition_length": 0.0,
    "output_transition_length": 0.0,
    "t_electrode_transition_length": 0.0,
    "sweep_parameters": ["signal_width", "length"],
    "sweep_ranges": {
        "signal_width": {"values": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
        "length": {"values": [500.0, 1000.0, 2000.0]},
    },
    "edge_spacing": 300.0,
    "layer": RF_LAYER,
    "datatype": DEFAULT_DATATYPE,
}

DEFAULT_COMPONENT_VALUES["MMI split-combine cascade"] = dict(DEFAULT_COMPONENT_VALUES["1x2 MMI"])

DEFAULT_COMPONENT_VALUES["MMI split-combine cascade"].update({"cascade_count":3,"interconnect_length":5.0,"add_grating_couplers":False,"add_input_grating_coupler":True,"add_output_grating_coupler":True,"add_reference_waveguide":True,"reference_vertical_offset":150.0,"add_opposed_output_s_bends":True,"output_s_bend_length":200.0,"output_s_bend_offset":100.0,"gc_wg_length":50.0})

DEFAULT_COMPONENT_VALUES["MMI split-combine test block"] = dict(DEFAULT_COMPONENT_VALUES["MMI split-combine cascade"])

DEFAULT_COMPONENT_VALUES["MMI split-combine test block"].update({"mmi_width":7.0,"taper_length_start":8.0,"taper_length_stop":12.0,"taper_length_step":1.0,"taper_width_start":2.5,"taper_width_stop":3.1,"taper_width_step":0.1,"nominal_taper_width":2.7,"device_x_spacing":950.0,"device_y_spacing":950.0,"ebeam_field_size":850.0,"ebeam_edge_clearance":10.0,"include_ebeam_fields":True,"parameter_text_height":10.0})

# Official Ansys Lumerical 3D SOI focusing-grating example.  These names and
# values intentionally follow the model setup script in grating_coupler_3D.fsp.
DEFAULT_COMPONENT_VALUES["GC-SOI"] = {
    "target_length": 25.0,
    "h_total": 0.22,
    "etch_depth": 0.10,
    "duty_cycle": 0.3992,
    "fill_factors": "",
    "tooth_shape": "curved",
    "pitch": 0.6713,
    "radius": 25.0,
    "y_span": 15.0,
    "L_extra": 10.0,
    "wg_width": 0.5,
    "wg_length": 10.0,
    "taper_exponent": 1.15,
    # Signed local-X distance from the first SOI grating flare
    # (wg_length + radius) to the fiber bottom center.
    "fiber_offset": 2.74533,
    # Canonical parent-level fiber/source tilt.  Older projects used
    # ``fiber_tilt_deg`` and are migrated when loaded.
    "angle_theta": 10.0,
    "fiber_tox_offset_um": 0.65,
    "fiber_core_diameter_um": 9.0,
    "fiber_core_index": 1.44427,
    "fiber_cladding_diameter_um": 50.0,
    "fiber_cladding_index": 1.43482,
    "fiber_length_um": 20.0,
    "fiber_power_monitor_below_source_um": 0.1,
    "fdtd_port_offset_from_waveguide_end_um": 2.0,
    "waveguide_monitor_span_um": 2.5,
    "waveguide_total_power_before_mode_um": 1.0,
    "waveguide_neff_tolerance": 0.3,
    "waveguide_mode_search_count": 20,
    "slab_layer": PHOTONIC_LAYER,
    "slab_datatype": DEFAULT_DATATYPE,
    "etched_layer": GC_LAYER,
    "etched_datatype": DEFAULT_DATATYPE,
    # GDS curve tolerance in micrometres: 0.005 um = 5 nm.
    "tolerance": 0.005,
}

INTEGER_PARAMETERS.add("seg_segment_count")

INTEGER_PARAMETERS.add("cascade_count")

BOOL_PARAMETERS.add("add_output_grating_coupler")

BOOL_PARAMETERS.add("include_rf_edge_bends")

BOOL_PARAMETERS.update({"add_input_grating_coupler","add_reference_waveguide"})

BOOL_PARAMETERS.add("add_opposed_output_s_bends")

BOOL_PARAMETERS.update({"cpw_align_to_mzi_s_bends","include_segmented_electrode","seg_include_oxide_masks"})

BOOL_PARAMETERS.add("seg_auto_segment_count")

BOOL_PARAMETERS.add("taper_test_structure")

# Simulation-only objects are ordinary movable editor items, but the GDS
# builder explicitly ignores them. Their parameter names mirror lumapi/CML
# names so the project JSON and exported notebook remain easy to compare.
DEFAULT_COMPONENT_VALUES.update(
    {
        "FDTD port": {
            "name": "opt_1", "dir": "Bidirectional", "loc": 0.5, "pos": "Left", "order": 1,
            "port geometry": "surface", "plane normal": "X", "distance_um": 0.0, "span_um": 3.0, "z_span_um": 2.25,
            "mode": "fundamental mode",
        },
        "Fiber-axis FDTD port": {
            "name": "opt_1", "dir": "Bidirectional", "loc": 0.5, "pos": "Top", "order": 1,
            "port geometry": "surface", "plane normal": "Z", "distance_um": 0.0, "z reference": "top of stack",
            "span_um": 20.0, "z_span_um": 0.0, "mode": "user select",
            # Zero means the exporter calculates the near-degenerate pair and
            # selects the member polarized transverse to the grating axis.
            "mode number": 0, "polarization": "local TE",
            "candidate mode numbers": [1, 2, 3],
            "mode degeneracy tolerance": 0.01,
            "minimum local TE fraction": 0.8,
            "angle theta": 7.0, "angle phi": 0.0,
            "align to fiber axis": True,
            "rotation offset_um": 4.420244193,
        },
        "Fiber geometry": {
            "name": "fiber", "distance_um": 0.0, "z reference": "top of SiO2 cladding",
            "angle theta": 7.0, "angle phi": 0.0,
            "core diameter_um": 9.0, "core index": 1.44427,
            "cladding diameter_um": 50.0, "cladding index": 1.43482,
            "fiber length_um": 20.0,
        },
        "Fiber port": {
            "name": "opt_1", "dir": "Bidirectional", "loc": 0.5, "pos": "Top", "order": 1,
            "port geometry": "fiber", "plane normal": "Z", "distance_um": 0.0, "span_um": 20.0,
            "angle theta": 7.0, "angle phi": 0.0,
            "core diameter_um": 9.0, "core index": 1.44427,
            "cladding diameter_um": 50.0, "cladding index": 1.43482,
            "fiber length_um": 20.0,
        },
        "Power monitor": {
            "name": "power_monitor", "monitor geometry": "surface", "plane normal": "X", "distance_um": 0.0,
            "x span": 0.0, "y span": 4.0, "z span": 2.0,
        },
        "Mode expansion monitor": {
            "name": "mode_expansion", "monitor geometry": "surface", "plane normal": "X", "distance_um": 0.0,
            "x span": 0.0, "y span": 4.0, "z span": 2.0, "mode": "fundamental TE mode",
            "target neff": 2.5, "neff tolerance": 0.3, "mode search count": 20,
            "expansion for": "",
        },
        "Field profile monitor": {
            "name": "field_profile", "monitor geometry": "surface", "plane normal": "X", "distance_um": 0.0,
            "x span": 0.0, "y span": 4.0, "z span": 2.0,
        },
        "RF mode port": {
            "name": "rf_port_1", "rf role": "Source", "order": 1,
            "port geometry": "surface", "plane normal": "X", "distance_um": 0.0,
            "span_um": 450.0, "z_span_um": 650.0,
            "mode": "fundamental quasi-TEM mode", "reference impedance_ohm": 50.0,
            "deembed_um": 0.0, "multifrequency mode injection": True,
        },
        "RF power monitor": {
            "name": "rf_power_1", "rf role": "Output",
            "monitor geometry": "surface", "plane normal": "X", "distance_um": 0.0,
            "span_um": 450.0, "z_span_um": 650.0,
            "expansion port": "",
        },
    }
)

INTEGER_PARAMETERS.add("order")

INTEGER_PARAMETERS.add("waveguide_mode_search_count")

INTEGER_PARAMETERS.add("mode search count")

BOOL_PARAMETERS.add("align to fiber axis")

BOOL_PARAMETERS.add("multifrequency mode injection")

PHOTONIC_COMPONENT_KINDS = (
    set(DEFAULT_COMPONENT_VALUES)
    - RF_COMPONENT_KINDS
    - MARKER_COMPONENT_KINDS
    - LEGACY_PHOTONIC_TEST_BLOCK_KINDS
    - SIMULATION_COMPONENT_KINDS
    - {
        "MZI + CPW module",
        "Chip outline",
        "4-inch wafer outline",
        "E-beam multipass",
        "Boolean geometry",
    }
)

DEFAULT_COMPONENT_VALUES["Photonic test block"] = {
    # Write fields are deliberately a separate, movable editor object.  New
    # photonic scan blocks therefore contain only their fabrication geometry;
    # users can cover the finished block from the E-beam toolbar afterwards.
    # An older project that explicitly saved True remains supported by the
    # GDS builder for backwards compatibility.
    "include_ebeam_fields": False,
    "ebeam_field_size": 520.0,
    "ebeam_edge_clearance": 10.0,
    "parameter_text_height": 12.0,
    "photonic_component_kind": "Straight",
    "photonic_base_params": dict(DEFAULT_COMPONENT_VALUES["Straight"]),
    "device_label_prefix": "Straight",
    "label_height": 20.0,
    "label_offset_x": 0.0,
    "label_offset_y": 10.0,
    "label_layer": MARKER_LAYER,
    "label_datatype": DEFAULT_DATATYPE,
    "sweep_parameters": ["length", "width"],
    "sweep_ranges": {
        "length": {"start": 25.0, "stop": 100.0, "step": 25.0},
        "width": {"start": 0.8, "stop": 1.6, "step": 0.2},
    },
    "edge_spacing": 300.0,
    "layer": PHOTONIC_LAYER,
    "datatype": DEFAULT_DATATYPE,
}

CHOICE_PARAMETERS = {"reference_branch": ["upper", "lower"], "profile": ["linear", "exponential", "klopfenstein"], "input_profile": ["linear", "exponential", "klopfenstein"], "output_profile": ["linear", "exponential", "klopfenstein"], "rf_input_taper_profile": ["linear", "exponential", "klopfenstein"], "rf_output_taper_profile": ["linear", "exponential", "klopfenstein"], "taper_test_center": ["CPW", "T electrode"], "resonator_side": ["upper", "lower"], "tooth_shape": ["curved", "rectangular"], "grating_route": ["straight", "up", "down"], "gc_input_route": ["straight", "up", "down"], "gc_output_route": ["straight", "up", "down"], "gc_upper_output_route": ["straight", "up", "down"], "gc_lower_output_route": ["straight", "up", "down"], "gc_vertical_side": ["up", "down"], "input_s_bend_direction": ["up", "down"], "output_s_bend_direction": ["up", "down"], "start_corner": ["top-left", "top-right", "bottom-left", "bottom-right"], "primary_axis": ["x", "y"]}

CHOICE_PARAMETERS["cpw_profile"] = ["linear", "exponential", "klopfenstein"]

CHOICE_PARAMETERS.update({"lattice":["triangular","square"],"hole_shape":["circular","elliptical"]})

CHOICE_PARAMETERS.update({"slab_shape":["rectangle","ellipse","hexagon"],"device_type":["bulk crystal","line-defect waveguide"],"mask_tone":["positive: holes only","negative: slab minus holes"]})

CHOICE_PARAMETERS.update({
    "port geometry": ["line", "surface"],
    "monitor geometry": ["line", "surface"],
    "plane normal": ["X", "Y", "Z"],
    "dir": ["Bidirectional", "Forward", "Backward"],
    "pos": ["Left", "Right", "Top", "Bottom"],
    "z reference": ["top of SiO2 cladding", "top of stack", "device top"],
    "rf role": ["Source", "Input reference", "Output"],
})

COMPONENT_DISPLAY_NAMES={
    "GC-SOI":"GC-SOI — official Ansys 3D SOI grating coupler",
    "1x2 MMI":"1×2 Multimode Interferometer (MMI)",
    "Cascaded MMI":"Cascaded 1×2 MMI splitter tree",
    "MMI + Reference":"MMI device with reference waveguide",
    "MMI + Reference test block":"MMI/reference sweep test block",
    "MMI split-combine cascade":"MMI splitter–combiner cascade",
    "MMI split-combine test block":"MMI splitter–combiner sweep block",
    "MZI vertical GC":"MZI with vertical grating couplers",
    "Vertical-GC MZI test block":"Vertical-grating-coupler MZI sweep block",
    "Straight-GC MZI + segmented RF bends test block":"Straight-grating MZI with segmented RF edge bends",
    "Straight-GC MZI + CPW RF bends test block":"Straight-grating MZI with continuous CPW edge bends",
    "Double-ring test block":"Dual-ring resonator sweep block",
    "Grating test block":"Grating-coupler pitch/fill sweep block",
    "Grating angle-taper test block":"Grating-coupler angle/taper sweep block",
    "Photonic test block":"Photonic component parameter-scan test block",
    "RF test block":"RF component parameter-scan test block",
    "4-inch wafer outline":"4-inch (100 mm) wafer outline with primary flat",
    "FDTD port":"Ansys standard FDTD port — waveguide",
    "Fiber-axis FDTD port":"Ansys standard FDTD port — fiber axis",
    "Fiber geometry":"Ansys fiber geometry group — core + cladding",
    "Fiber port":"Legacy combined fiber object",
    "Power monitor":"Power monitor",
    "Mode expansion monitor":"Mode expansion monitor",
    "Field profile monitor":"Field profile monitor",
    "RF mode port":"RF modal source / port — CPW quasi-TEM",
    "RF power monitor":"RF DFT power / mode-expansion plane",
}


def component_display_name(kind:str)->str:return COMPONENT_DISPLAY_NAMES.get(kind,kind)


def load_component_specs() -> dict[str, dict[str, list[Any]]]:
    specs: dict[str, dict[str, list[Any]]] = {}
    for kind, defaults in DEFAULT_COMPONENT_VALUES.items():
        specs[kind] = {}
        for key, value in defaults.items():
            if key in CHOICE_PARAMETERS:
                specs[kind][key] = ["choice", value, CHOICE_PARAMETERS[key]]
            elif key in BOOL_PARAMETERS:
                specs[kind][key] = ["bool", bool(value)]
            elif key in INTEGER_PARAMETERS:
                specs[kind][key] = ["int", value]
            elif isinstance(value, str):
                specs[kind][key] = ["string", value]
            else:
                specs[kind][key] = ["float", value]
    return specs

COMPONENT_SPECS = load_component_specs()

NATIVE_APP_VERSION = "V1.1"
