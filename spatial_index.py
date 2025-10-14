"""
Qualified Path: src/bonsai/bonsai/bim/module/federation/spatial_index.py

Spatial Index Module - Rtree-based Federated Model Queries
-----------------------------------------------------------
Provides fast spatial queries over preprocessed IFC bounding boxes
using rtree R-tree indexing for multi-model coordination.

Usage:
    from bonsai.bim.module.federation.spatial_index import FederationIndex
    
    index = FederationIndex("/path/to/federation.db")
    index.build()
    
    # Query obstacles in routing corridor
    obstacles = index.query_by_bbox(
        min_xyz=(1000, 2000, 3000),
        max_xyz=(1500, 2500, 4000)
    )
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rtree import index as rtree_index
except ImportError:
    raise ImportError(
        "rtree library required for spatial indexing. "
        "Install with: pip install rtree"
    )


class FederationElement:
    """Represents a federated element with spatial data"""
    
    def __init__(self, guid: str, discipline: str, ifc_class: str,
                 bbox: Tuple[float, float, float, float, float, float],
                 filepath: str):
        self.guid = guid
        self.discipline = discipline
        self.ifc_class = ifc_class
        self.min_x, self.min_y, self.min_z = bbox[:3]
        self.max_x, self.max_y, self.max_z = bbox[3:]
        self.filepath = filepath
    
    @property
    def bbox(self) -> Tuple[float, float, float, float, float, float]:
        """Return bounding box as (min_x, min_y, min_z, max_x, max_y, max_z)"""
        return (self.min_x, self.min_y, self.min_z, 
                self.max_x, self.max_y, self.max_z)
    
    @property
    def centroid(self) -> Tuple[float, float, float]:
        """Calculate bbox centroid"""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        )
    
    def intersects_bbox(self, other_bbox: Tuple[float, float, float, float, float, float]) -> bool:
        """Check if this element's bbox intersects another bbox"""
        other_min_x, other_min_y, other_min_z, other_max_x, other_max_y, other_max_z = other_bbox
        
        # Axis-Aligned Bounding Box (AABB) intersection test
        return not (
            self.max_x < other_min_x or self.min_x > other_max_x or
            self.max_y < other_min_y or self.min_y > other_max_y or
            self.max_z < other_min_z or self.min_z > other_max_z
        )
    
    def __repr__(self):
        return (f"FederationElement(guid={self.guid}, "
                f"discipline={self.discipline}, "
                f"ifc_class={self.ifc_class})")


class FederationIndex:
    """Spatial index for federated IFC models"""
    
    def __init__(self, database_path: Path, logger: Optional[logging.Logger] = None):
        self.database_path = Path(database_path)
        self.logger = logger or self._setup_logging()
        self.rtree_idx = None
        self.guid_to_element: Dict[str, FederationElement] = {}
        self.is_loaded = False
        
        # Statistics
        self.stats = {
            'total_elements': 0,
            'disciplines': set(),
            'ifc_classes': set()
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('FederationIndex')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build(self, force_rebuild: bool = False) -> None:
        """
        Build rtree spatial index from database
        
        Args:
            force_rebuild: If True, rebuild even if already loaded
        """
        if self.is_loaded and not force_rebuild:
            self.logger.info("Index already loaded. Use force_rebuild=True to reload.")
            return
        
        self.logger.info(f"Building spatial index from {self.database_path.name}...")
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        
        # Validate database
        self._validate_database()
        
        # Create rtree index properties for 3D
        props = rtree_index.Property()
        props.dimension = 3
        props.leaf_capacity = 100
        props.fill_factor = 0.9
        
        # Initialize rtree
        self.rtree_idx = rtree_index.Index(properties=props)
        self.guid_to_element.clear()
        
        # Load elements from database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT guid, discipline, ifc_class, 
                   min_x, min_y, min_z, max_x, max_y, max_z, filepath
            FROM elements
        """)
        
        element_count = 0
        for row in cursor.fetchall():
            guid, discipline, ifc_class = row[:3]
            bbox = tuple(row[3:9])
            filepath = row[9]
            
            # Create element
            element = FederationElement(guid, discipline, ifc_class, bbox, filepath)
            
            # Insert into rtree (rtree expects interleaved min/max coords)
            # Format: (min_x, min_y, min_z, max_x, max_y, max_z)
            self.rtree_idx.insert(element_count, bbox, obj=guid)
            
            # Store in lookup dict
            self.guid_to_element[guid] = element
            
            # Update statistics
            self.stats['disciplines'].add(discipline)
            self.stats['ifc_classes'].add(ifc_class)
            
            element_count += 1
            
            if element_count % 10000 == 0:
                self.logger.info(f"  Loaded {element_count} elements...")
        
        conn.close()
        
        self.stats['total_elements'] = element_count
        self.is_loaded = True
        
        self.logger.info(f"✓ Index built: {element_count:,} elements from "
                        f"{len(self.stats['disciplines'])} disciplines")
    
    def _validate_database(self):
        """Validate database schema"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check schema version
        try:
            cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
            version = cursor.fetchone()
            if not version:
                raise ValueError("Missing schema version in database")
        except sqlite3.OperationalError:
            raise ValueError("Invalid database: missing schema_info table")
        
        # Check elements table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='elements'
        """)
        if not cursor.fetchone():
            raise ValueError("Invalid database: missing elements table")
        
        conn.close()
    
    def query_by_bbox(self, min_xyz: Tuple[float, float, float],
                      max_xyz: Tuple[float, float, float],
                      disciplines: Optional[List[str]] = None,
                      ifc_classes: Optional[List[str]] = None) -> List[FederationElement]:
        """
        Query elements intersecting bounding box
        
        Args:
            min_xyz: Minimum corner (x, y, z)
            max_xyz: Maximum corner (x, y, z)
            disciplines: Optional filter by discipline tags
            ifc_classes: Optional filter by IFC classes
            
        Returns:
            List of FederationElement instances
        """
        if not self.is_loaded:
            raise RuntimeError("Index not loaded. Call build() first.")
        
        # Initialize results list
        results = []
        
        # Normalize discipline filters early
        if disciplines:
            disciplines = [self._normalize_discipline(d) for d in disciplines]
        
        # Construct bbox for rtree query (always needed)
        query_bbox = min_xyz + max_xyz
        
        # Query rtree spatial index
        for item in self.rtree_idx.intersection(query_bbox, objects=True):
            guid = item.object
            element = self.guid_to_element[guid]
            
            # Apply discipline filter with normalization
            if disciplines and self._normalize_discipline(element.discipline) not in disciplines:
                continue
            
            # Apply IFC class filter
            if ifc_classes and element.ifc_class not in ifc_classes:
                continue
            
            results.append(element)
        
        return results
    
    def query_by_point(self, point: Tuple[float, float, float],
                       radius: float = 0.0,
                       disciplines: Optional[List[str]] = None) -> List[FederationElement]:
        """
        Query elements at or near a point
        
        Args:
            point: Query point (x, y, z)
            radius: Search radius (0 = exact point)
            disciplines: Optional filter by discipline tags
            
        Returns:
            List of FederationElement instances
        """
        x, y, z = point
        min_xyz = (x - radius, y - radius, z - radius)
        max_xyz = (x + radius, y + radius, z + radius)
        
        return self.query_by_bbox(min_xyz, max_xyz, disciplines)
    
    def query_corridor(self, start: Tuple[float, float, float],
                      end: Tuple[float, float, float],
                      buffer: float = 500.0,
                      disciplines: Optional[List[str]] = None) -> List[FederationElement]:
        """
        Query elements along a corridor path with buffer
        
        Args:
            start: Start point (x, y, z)
            end: End point (x, y, z)
            buffer: Corridor width/height buffer in METERS
            disciplines: Optional filter by discipline tags
            
        Returns:
            List of FederationElement instances along corridor
        """
        # Calculate bbox encompassing corridor with buffer
        min_x = min(start[0], end[0]) - buffer
        max_x = max(start[0], end[0]) + buffer
        min_y = min(start[1], end[1]) - buffer
        max_y = max(start[1], end[1]) + buffer
        min_z = min(start[2], end[2]) - buffer
        max_z = max(start[2], end[2]) + buffer
        
        return self.query_by_bbox(
            (min_x, min_y, min_z),
            (max_x, max_y, max_z),
            disciplines
        )
    
    def _normalize_discipline(self, discipline: str) -> str:
        """
        Normalize discipline name to core abbreviation
        SJTII-STR- → STR
        SJTII-ACMV → ACMV
        STR → STR
        """
        import re
        
        # Extract 2-4 letter uppercase abbreviations
        parts = re.split(r'[-_]', discipline.upper())
        
        known = ['STR', 'ACMV', 'ARC', 'ELEC', 'FP', 'SP', 'CW', 
                'STRUCT', 'ARCH', 'HVAC', 'MECH', 'PLUMB', 'FIRE']
        
        for part in parts:
            if part in known:
                return part
        
        # Return first 2-4 letter alpha part
        for part in parts:
            if 2 <= len(part) <= 4 and part.isalpha():
                return part
        
        return discipline  # Fallback unchanged
    
    def get_element_by_guid(self, guid: str) -> Optional[FederationElement]:
        """Retrieve element by GlobalId"""
        return self.guid_to_element.get(guid)
    
    def get_disciplines(self) -> List[str]:
        """Get list of all disciplines in index"""
        return sorted(self.stats['disciplines'])
    
    def get_ifc_classes(self) -> List[str]:
        """Get list of all IFC classes in index"""
        return sorted(self.stats['ifc_classes'])
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        return {
            'total_elements': self.stats['total_elements'],
            'disciplines': self.get_disciplines(),
            'ifc_classes': self.get_ifc_classes(),
            'discipline_count': len(self.stats['disciplines']),
            'class_count': len(self.stats['ifc_classes']),
            'is_loaded': self.is_loaded
        }
    
    def clear(self):
        """Clear index from memory"""
        if self.rtree_idx:
            self.rtree_idx.close()
            self.rtree_idx = None
        
        self.guid_to_element.clear()
        self.is_loaded = False
        self.stats = {
            'total_elements': 0,
            'disciplines': set(),
            'ifc_classes': set()
        }
        
        self.logger.info("Index cleared from memory")


# Convenience function for quick queries
def quick_query(database_path: Path, min_xyz: Tuple[float, float, float],
                max_xyz: Tuple[float, float, float]) -> List[FederationElement]:
    """
    Quick spatial query without persistent index
    
    Args:
        database_path: Path to federation database
        min_xyz: Minimum corner (x, y, z)
        max_xyz: Maximum corner (x, y, z)
        
    Returns:
        List of FederationElement instances
    """
    index = FederationIndex(database_path)
    index.build()
    return index.query_by_bbox(min_xyz, max_xyz)


if __name__ == "__main__":
    # Test/demo code
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python spatial_index.py <database_path>")
        sys.exit(1)
    
    db_path = Path(sys.argv[1])
    
    print(f"Loading federation index from {db_path}...")
    index = FederationIndex(db_path)
    index.build()
    
    stats = index.get_statistics()
    print("\nIndex Statistics:")
    print(f"  Total Elements:  {stats['total_elements']:,}")
    print(f"  Disciplines:     {', '.join(stats['disciplines'])}")
    print(f"  IFC Classes:     {stats['class_count']} unique types")
    
    # Example query
    print("\nExample Query (1000mm cube at origin):")
    results = index.query_by_bbox((-500, -500, -500), (500, 500, 500))
    print(f"  Found {len(results)} elements")
    for element in results[:5]:
        print(f"    - {element}")