"""
G-code Parser and Settings Extractor
Extracts printer and printing settings from PrusaSlicer G-code files
"""
import re
from typing import Dict, Any, Optional


class GCodeParser:
    """Parser for extracting settings from G-code files"""
    
    def __init__(self):
        self.settings = {}
    
    def parse_gcode_file(self, gcode_content: str) -> Dict[str, Any]:
        """
        Parse G-code content and extract all settings
        
        Args:
            gcode_content: Full G-code file content as string
            
        Returns:
            Dictionary containing extracted settings
        """
        settings = {
            'printer': {},
            'printing': {},
            'extrusion': {},
            'temperatures': {},
            'speeds': {},
            'retraction': {},
            'layer': {},
            'infill': {}
        }
        
        # Extract PrusaSlicer config section
        config_match = re.search(r'prusaslicer_config\s*=\s*\{([^}]+)\}', gcode_content, re.DOTALL)
        if config_match:
            config_section = config_match.group(1)
            settings.update(self._parse_config_section(config_section))
        
        # Extract temperatures
        temp_patterns = {
            'bed_temperature': r'bed_temperature\s*=\s*(\d+)',
            'first_layer_bed_temperature': r'first_layer_bed_temperature\s*=\s*(\d+)',
            'extruder_temperature': r'extruder_temperature\s*=\s*(\d+)',
            'first_layer_temperature': r'first_layer_temperature\s*=\s*(\d+)',
        }
        for key, pattern in temp_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                settings['temperatures'][key] = int(match.group(1))
        
        # Extract layer height
        layer_height_match = re.search(r'layer_height\s*=\s*([\d.]+)', gcode_content)
        if layer_height_match:
            settings['layer']['height'] = float(layer_height_match.group(1))
        
        first_layer_height_match = re.search(r'first_layer_height\s*=\s*([\d.]+)', gcode_content)
        if first_layer_height_match:
            settings['layer']['first_height'] = float(first_layer_height_match.group(1))
        
        # Extract extrusion widths
        width_patterns = {
            'external_perimeter': r'external_perimeter_extrusion_width\s*=\s*([\d.]+)',
            'perimeter': r'perimeter_extrusion_width\s*=\s*([\d.]+)',
            'infill': r'infill_extrusion_width\s*=\s*([\d.]+)',
            'solid_infill': r'solid_infill_extrusion_width\s*=\s*([\d.]+)',
            'top_infill': r'top_infill_extrusion_width\s*=\s*([\d.]+)',
            'first_layer': r'first_layer_extrusion_width\s*=\s*([\d.]+)',
        }
        for key, pattern in width_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                settings['extrusion'][key] = float(match.group(1))
        
        # Extract speeds
        speed_patterns = {
            'perimeter': r'perimeter_speed\s*=\s*([\d.]+)',
            'external_perimeter': r'external_perimeter_speed\s*=\s*([\d.]+)',
            'infill': r'infill_speed\s*=\s*([\d.]+)',
            'solid_infill': r'solid_infill_speed\s*=\s*([\d.]+)',
            'top_solid_infill': r'top_solid_infill_speed\s*=\s*([\d.]+)',
            'support_material': r'support_material_speed\s*=\s*([\d.]+)',
            'travel': r'travel_speed\s*=\s*([\d.]+)',
            'first_layer': r'first_layer_speed\s*=\s*([\d.]+)',
        }
        for key, pattern in speed_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                settings['speeds'][key] = float(match.group(1))
        
        # Extract retraction settings
        retraction_patterns = {
            'length': r'retract_length\s*=\s*([\d.]+)',
            'lift': r'retract_lift\s*=\s*([\d.]+)',
            'speed': r'retract_speed\s*=\s*([\d.]+)',
            'deretract_speed': r'deretract_speed\s*=\s*([\d.]+)',
        }
        for key, pattern in retraction_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                settings['retraction'][key] = float(match.group(1))
        
        # Extract infill settings
        infill_patterns = {
            'pattern': r'infill_pattern\s*=\s*(\w+)',
            'density': r'fill_density\s*=\s*([\d.]+)',
            'anchor': r'infill_anchor\s*=\s*([\d.]+)',
            'anchor_max': r'infill_anchor_max\s*=\s*([\d.]+)',
            'angles': r'infill_angles\s*=\s*\[([^\]]+)\]',
        }
        for key, pattern in infill_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                if key == 'angles':
                    # Parse comma-separated angles
                    angles_str = match.group(1)
                    angles = [float(a.strip()) for a in angles_str.split(',') if a.strip()]
                    settings['infill'][key] = angles if angles else [45, 135]  # Default angles
                elif key == 'pattern':
                    settings['infill'][key] = match.group(1)
                else:
                    settings['infill'][key] = float(match.group(1))
        
        # Extract printer settings
        printer_patterns = {
            'bed_shape': r'bed_shape\s*=\s*([^\n]+)',
            'nozzle_diameter': r'nozzle_diameter\s*=\s*([\d.]+)',
        }
        for key, pattern in printer_patterns.items():
            match = re.search(pattern, gcode_content)
            if match:
                settings['printer'][key] = match.group(1)
        
        return settings
    
    def _parse_config_section(self, config_section: str) -> Dict[str, Any]:
        """Parse the PrusaSlicer config section"""
        settings = {}
        
        # Extract printer settings
        printer_match = re.search(r'printer_settings_id\s*=\s*"([^"]+)"', config_section)
        if printer_match:
            settings['printer_name'] = printer_match.group(1)
        
        # Extract print settings
        print_match = re.search(r'print_settings_id\s*=\s*"([^"]+)"', config_section)
        if print_match:
            settings['print_profile'] = print_match.group(1)
        
        return settings
    
    def merge_settings(self, *settings_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple settings dictionaries, with later ones taking precedence
        
        Args:
            *settings_dicts: Variable number of settings dictionaries
            
        Returns:
            Merged settings dictionary
        """
        merged = {}
        for settings in settings_dicts:
            merged = self._deep_merge(merged, settings)
        return merged
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default settings template"""
        return {
            'printer': {
                'nozzle_diameter': 0.4,
                'bed_shape': '0x0,220x0,220x220,0x220',
            },
            'temperatures': {
                'bed_temperature': 60,
                'first_layer_bed_temperature': 60,
                'extruder_temperature': 210,
                'first_layer_temperature': 210,
            },
            'layer': {
                'height': 0.2,
                'first_height': 0.2,
            },
            'extrusion': {
                'external_perimeter': 0.45,
                'perimeter': 0.45,
                'infill': 0.45,
                'solid_infill': 0.45,
                'top_infill': 0.45,
                'first_layer': 0.5,
            },
            'speeds': {
                'perimeter': 60,
                'external_perimeter': 30,
                'infill': 80,
                'solid_infill': 60,
                'top_solid_infill': 30,
                'support_material': 60,
                'travel': 120,
                'first_layer': 20,
            },
            'retraction': {
                'length': 2.0,
                'lift': 0.2,
                'speed': 40,
                'deretract_speed': 40,
            },
            'infill': {
                'pattern': 'rectilinear',
                'density': 20.0,
                'angles': [45, 135],  # Default angles for alternating layers
                'anchor': 0.0,
                'anchor_max': 0.0,
            },
        }

