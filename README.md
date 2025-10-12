# Bonsai10D Federation Module

Spatial query engine for multi-model BIM coordination in Bonsai/BlenderBIM.

## Features

- Preprocess multiple IFC files to SQLite spatial index
- Query across disciplines without loading geometry
- Sub-second spatial queries on 100K+ elements
- Memory-efficient (<10GB for federated models)

## Installation

```bash
cd src/bonsai/bonsai/bim/module/
git clone https://github.com/red1oon/federation.git
```

Restart Blender to load the module.

## Usage
Preprocess IFC files using federation_preprocessor.py
Load spatial index in Bonsai Federation panel
Query obstacles for routing/clash detection workflows
Part of Bonsai10D Vision
10-dimensional construction coordination: 3D geometry + time + cost + full ERP integration.

## License
GPL-3.0-or-later

## Author
Redhuan D. Oon (red1), and Naquib Danial Oon
