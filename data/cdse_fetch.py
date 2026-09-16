"""
CDSE (Copernicus Data Space Ecosystem) Sentinel-2 L2A download.
Free account required: https://dataspace.copernicus.eu/
"""
import os
import json
import requests
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


class CDSEClient:
    """Download and cache Sentinel-2 L2A from CDSE."""

    def __init__(self, username, password, cache_dir="data/scenes"):
        self.username = username
        self.password = password
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = None
        self.authenticate()

    def authenticate(self):
        """Get access token."""
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public",
        }
        resp = requests.post(url, data=data)
        resp.raise_for_status()
        self.token = resp.json()["access_token"]
        print(f"✓ CDSE authenticated")

    def query_products(self, bbox, start_date, end_date, cloud_cover_max=20):
        """
        Query Sentinel-2 L2A products.
        bbox: (west, south, east, north)
        Returns list of product IDs.
        """
        west, south, east, north = bbox
        wkt = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"

        url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
        params = {
            "filter": json.dumps({
                "type": "Polygon",
                "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]
            }),
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "limit": 100,
        }
        headers = {"Authorization": f"Bearer {self.token}"}

        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        products = []
        for feature in data.get("features", []):
            cloud_cover = feature.get("properties", {}).get("eo:cloud_cover", 100)
            if cloud_cover <= cloud_cover_max:
                products.append({
                    "id": feature["id"],
                    "date": feature["properties"]["datetime"],
                    "cloud_cover": cloud_cover,
                })

        return sorted(products, key=lambda x: x["cloud_cover"])

    def download_product(self, product_id):
        """Download a single L2A product. Returns path to local .zip."""
        zip_path = self.cache_dir / f"{product_id}.zip"
        if zip_path.exists():
            print(f"✓ {product_id} cached")
            return str(zip_path)

        url = f"https://sh.dataspace.copernicus.eu/api/odata/v1/Products('{product_id}')/$value"
        headers = {"Authorization": f"Bearer {self.token}"}

        resp = requests.get(url, headers=headers, stream=True)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=product_id) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded {product_id}")
        return str(zip_path)


def list_bands():
    """Sentinel-2 L2A bands we use: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12."""
    return {
        "B02": 10,  # 10m Blue
        "B03": 10,  # 10m Green
        "B04": 10,  # 10m Red
        "B05": 20,  # 20m Veg Red Edge
        "B06": 20,  # 20m Veg Red Edge
        "B07": 20,  # 20m Veg Red Edge
        "B08": 10,  # 10m NIR
        "B8A": 20,  # 20m Narrow NIR
        "B11": 20,  # 20m SWIR
        "B12": 20,  # 20m SWIR
    }
