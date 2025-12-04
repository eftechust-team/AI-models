# G-code Generation Guide

This application now supports direct G-code generation for 3D printing! You can learn printer and printing settings from existing G-code files, customize them, and generate G-code from your STL files.

## Features

1. **Learn from G-code Files**: Upload G-code files to extract printer and printing settings
2. **Customize Settings**: View and edit all G-code settings (temperatures, speeds, layer heights, etc.)
3. **Generate G-code**: Convert STL files to G-code using your learned/customized settings

## How to Use

### Step 1: Learn Settings from G-code Files

1. Generate an STL file first (using the main interface)
2. Scroll down to the "G-code Settings" section
3. Click "Choose File" and select a G-code file (`.gcode` or `.txt`)
4. Click "Learn Settings" to extract printer and printing settings
5. The system will automatically save these settings for future use

**Note**: You can upload multiple G-code files - each new file will update/merge settings with existing ones.

### Step 2: Customize Settings (Optional)

1. Click "View/Edit Settings" in the G-code Settings section
2. A modal will open showing all current settings organized by category:
   - **Temperatures**: Bed and extruder temperatures
   - **Layer**: Layer heights (first layer and regular layers)
   - **Extrusion**: Extrusion widths for different parts
   - **Speeds**: Printing speeds for different operations
   - **Retraction**: Retraction settings
   - **Infill**: Infill pattern and density
   - **Printer**: Printer-specific settings (nozzle diameter, bed shape)
3. Edit any values you want to change
4. Click "Save Settings" to apply changes

### Step 3: Generate G-code

1. After generating an STL file, you'll see a "🖨️ Generate G-code" button
2. Click it to generate G-code from your STL using the current settings
3. Once generated, click "📥 Download G-code" to download the file

## Settings Extracted from G-code Files

The parser extracts the following settings from PrusaSlicer G-code files:

### Temperatures (°C)
- Bed temperature (°C)
- First layer bed temperature (°C)
- Extruder temperature (°C)
- First layer extruder temperature (°C)

### Layer Settings (mm)
- Layer height (mm)
- First layer height (mm)

### Extrusion Widths (mm)
- External perimeter (mm)
- Perimeter (mm)
- Infill (mm)
- Solid infill (mm)
- Top infill (mm)
- First layer (mm)

### Speeds (mm/s)
- Perimeter speed (mm/s)
- External perimeter speed (mm/s)
- Infill speed (mm/s)
- Solid infill speed (mm/s)
- Top solid infill speed (mm/s)
- Support material speed (mm/s)
- Travel speed (mm/s)
- First layer speed (mm/s)

### Retraction
- Retract length (mm)
- Retract lift (mm)
- Retract speed (mm/s)
- Deretract speed (mm/s)

### Infill
- Infill pattern (text, e.g., "rectilinear", "honeycomb")
- Fill density (%)

### Printer
- Nozzle diameter (mm)
- Bed shape (text, e.g., "0x0,220x0,220x220,0x220")

## Default Settings

If you haven't learned from any G-code files, the system uses these default settings:

- **Nozzle diameter**: 0.4 mm
- **Bed temperature**: 60 °C
- **Extruder temperature**: 210 °C
- **Layer height**: 0.2 mm
- **First layer height**: 0.2 mm
- **Extrusion widths**: 0.45 mm (all types)
- **Speeds**: Various (perimeter: 60 mm/s, infill: 80 mm/s, etc.)
- **Retraction**: 2.0 mm length, 0.2 mm lift, 40 mm/s speed

## Tips

1. **Learn from Multiple Files**: Upload several G-code files to build a comprehensive settings profile
2. **Customize Per Print**: You can adjust settings before each G-code generation
3. **Save Your Settings**: Settings are automatically saved and persist between sessions
4. **Start with Defaults**: If you're new, start with the default settings and adjust as needed

## Technical Notes

- The G-code generator creates a simplified G-code file suitable for basic 3D printing
- For complex models, you may want to use a full slicer (like PrusaSlicer) for optimal results
- The generated G-code includes proper temperature settings, layer commands, and basic movement patterns
- The system uses a test pattern when STL slicing is not available (this is a simplified implementation)

## API Endpoints

- `POST /api/gcode/learn` - Learn settings from uploaded G-code file
- `GET /api/gcode/settings` - Get current G-code settings
- `POST /api/gcode/settings` - Update G-code settings
- `POST /api/gcode/generate` - Generate G-code from STL file
- `GET /api/gcode/download/<filename>` - Download generated G-code file

