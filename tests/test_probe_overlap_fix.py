import sys, unittest
from pathlib import Path
import numpy as np
from pyproj import CRS
from rasterio.transform import Affine
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"oregon"))
from select_tiles import select_probe_tiles
from terrain_utils import TerrainTile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"oregon"/"diagnostics"))
from probe_matched_tiles import crop_tile_to_wgs84_geometry

def tile(name,project,bbox,size=100):
    return {"title":name,"project":project,"bbox":bbox,"sizeInBytes":size,"downloadURL":f"https://example/{name}.laz"}

class ProbeOverlapFixTests(unittest.TestCase):
    def test_smaller_metric_accepts_nested_grids_and_records_intersection(self):
        result=select_probe_tiles([tile("large","a",[0,0,2,1]),tile("small","b",[0,0,1,1])],projects=["a","b"],count=1,overlap_threshold=.8,overlap_metric="smaller")
        self.assertAlmostEqual(.5,result["a"][0]["_probe_footprint_iou"])
        self.assertAlmostEqual(1.0,result["a"][0]["_probe_smaller_overlap"])
        self.assertIn("_probe_intersection_geometry",result["a"][0])
    def test_iou_default_remains_strict(self):
        with self.assertRaises(ValueError):
            select_probe_tiles([tile("large","a",[0,0,2,1]),tile("small","b",[0,0,1,1])],projects=["a","b"],count=1,overlap_threshold=.8)
    def test_large_catalog_no_anchor_recursion(self):
        records=[tile(f"a{i:04d}","a",[i,0,i+1,1]) for i in range(1200)]
        records += [tile(f"b{i:04d}","b",[i,0,i+1,1]) for i in range(1192,1200)]
        result=select_probe_tiles(records,projects=["a","b"],count=8,overlap_threshold=.99)
        self.assertEqual(8,len(result["a"]))
    def test_crop_identical_window(self):
        value=TerrainTile(np.ones((10,10),dtype=np.float32),Affine(.1,0,0,0,-.1,1),CRS.from_epsg(4326),np.ones((10,10),dtype=bool),100,1.0,Path("fake.laz"))
        geom={"type":"Polygon","coordinates":[[[.2,.2],[.8,.2],[.8,.8],[.2,.8],[.2,.2]]]}
        cropped=crop_tile_to_wgs84_geometry(value,geom)
        self.assertLess(cropped.shape[0],10)
        self.assertAlmostEqual(1.0,cropped.ground_cell_fraction)
if __name__=="__main__": unittest.main()
