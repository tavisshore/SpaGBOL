
from pathlib import Path

from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import re
import math
import io
import random
import concurrent.futures
import threading
import hashlib
import requests
import shapefile
import shapely.geometry
from PIL import Image, ImageEnhance, ImageOps
import ray
from ray.experimental import tqdm_ray
import osmnx as ox
import networkx as nx
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
from streetview import search_panoramas, get_panorama
from haversine import haversine, Unit
from pathlib import Path
from geographiclib.geodesic import Geodesic
import polarTransform
import cv2

from utils.guillame import ImageDatabase
from utils.write_db import write_database


torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 256 
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  
GOOGLE_MAPS_VERSION_FALLBACK = '934'
GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK = '148'
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15"
LOGGER = None
VERBOSITY = None

resize_pov = T.Resize((256, 256), antialias=True)
resize_pano = T.Resize((512, 2048), antialias=True)
to_ten = T.ToTensor()
to_pil = T.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViewDirection:
    def __init__(self, direction):
        self.angle = -1
        if direction == "downward": pass
        elif direction == "northward": self.angle = 0
        elif direction == "eastward": self.angle = 90
        elif direction == "southward": self.angle = 180
        elif direction == "westward": self.angle = 270
        else: raise ValueError(f"not a recognized view direction: {direction}")
        self.direction = direction

    def __repr__(self): return f"ViewDirection({self.direction})"

    def __str__(self): return self.direction

    def is_downward(self): return self.angle == -1

    def is_oblique(self): return not self.is_downward()

    def is_northward(self): return self.angle == 0

    def is_eastward(self): return self.angle == 90

    def is_southward(self): return self.angle == 180

    def is_westward(self): return self.angle == 270


class WebMercator:
    @staticmethod
    def project(geopoint, zoom):
        factor = (1 / (2 * math.pi)) * 2 ** zoom
        x = factor * (math.radians(geopoint.lon) + math.pi)
        y = factor * (math.pi - math.log(math.tan((math.pi / 4) + (math.radians(geopoint.lat) / 2))))
        return (x, y)


class ObliqueWebMercator:
    @staticmethod
    def project(geopoint, zoom, direction):
        x0, y0 = WebMercator.project(geopoint, zoom)
        width_and_height_of_world_in_tiles = 2 ** zoom
        equator_offset_from_edges = width_and_height_of_world_in_tiles / 2
        x, y = x0, y0
        if direction.is_northward(): pass
        elif direction.is_eastward():
            x = y0
            y = width_and_height_of_world_in_tiles - x0
        elif direction.is_southward():
            x = width_and_height_of_world_in_tiles - x0
            y = width_and_height_of_world_in_tiles - y0
        elif direction.is_westward():
            x = width_and_height_of_world_in_tiles - y0
            y = x0
        else: raise ValueError("direction must be one of 'northward', 'eastward', 'southward', or 'westward'")
        y = ((y - equator_offset_from_edges) / math.sqrt(2)) + equator_offset_from_edges
        return (x, y)


class GeoPoint:
    def __init__(self, lat, lon):
        assert -90 <= lat <= 90 and -180 <= lon <= 180
        self.lat = lat
        self.lon = lon

    def __repr__(self): return f"GeoPoint({self.lat}, {self.lon})"

    def fancy(self):
        def fancy_coord(coord, pos, neg):
            coord_dir = pos if coord > 0 else neg
            coord_tmp = abs(coord)
            coord_deg = math.floor(coord_tmp)
            coord_tmp = (coord_tmp - math.floor(coord_tmp)) * 60
            coord_min = math.floor(coord_tmp)
            coord_sec = round((coord_tmp - math.floor(coord_tmp)) * 600) / 10
            coord = f"{coord_deg}°{coord_min}'{coord_sec}\"{coord_dir}"
            return coord
        lat = fancy_coord(self.lat, "N", "S")
        lon = fancy_coord(self.lon, "E", "W")
        return f"{lat} {lon}"

    @classmethod
    def random(cls, georect):
        north = math.radians(georect.ne.lat)
        south = math.radians(georect.sw.lat)
        lat = math.degrees(math.asin(random.random() * (math.sin(north) - math.sin(south)) + math.sin(south)))
        west = georect.sw.lon
        east = georect.ne.lon
        width = east - west
        if width < 0: width += 360
        lon = west + width * random.random()
        if lon > 180: lon -= 360
        elif lon < -180: lon += 360
        return cls(lat, lon)

    def to_maptile(self, zoom, direction):
        x, y = WebMercator.project(self, zoom)
        if direction.is_oblique(): x, y = ObliqueWebMercator.project(self, zoom, direction)
        return MapTile(zoom, direction, math.floor(x), math.floor(y))

    def to_shapely_point(self): return shapely.geometry.Point(self.lon, self.lat)

    def compute_zoom_level(self, max_meters_per_pixel):
        meters_per_pixel_at_zoom_0 = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * math.cos(math.radians(self.lat)))
        for zoom in reversed(range(0, 23+1)):
            meters_per_pixel = meters_per_pixel_at_zoom_0 / (2 ** zoom)
            if meters_per_pixel > max_meters_per_pixel: return zoom + 1
        else: raise RuntimeError("your settings seem to require a zoom level higher than is commonly available")


class GeoRect:
    def __init__(self, sw, ne):
        assert sw.lat <= ne.lat
        self.sw = sw
        self.ne = ne

    def __repr__(self): return f"GeoRect({self.sw}, {self.ne})"

    @classmethod
    def from_shapefile_bbox(cls, bbox):
        sw = GeoPoint(bbox[1], bbox[0])
        ne = GeoPoint(bbox[3], bbox[2])
        return cls(sw, ne)

    @classmethod
    def around_geopoint(cls, geopoint, width, height):
        assert width > 0 and height > 0
        meters_per_degree = (EARTH_CIRCUMFERENCE / 360)
        width_geo = width / (meters_per_degree * math.cos(math.radians(geopoint.lat)))
        height_geo = height / meters_per_degree
        southwest = GeoPoint(geopoint.lat - height_geo / 2, geopoint.lon - width_geo / 2)
        northeast = GeoPoint(geopoint.lat + height_geo / 2, geopoint.lon + width_geo / 2)
        return cls(southwest, northeast)

    def area(self):
        earth_radius = EARTH_CIRCUMFERENCE / (1000 * 2 * math.pi)
        earth_surface_area_in_km = 4 * math.pi * earth_radius ** 2
        spherical_cap_difference = (2 * math.pi * earth_radius ** 2) * abs(math.sin(math.radians(self.sw.lat)) - math.sin(math.radians(self.ne.lat)))
        area = spherical_cap_difference * (self.ne.lon - self.sw.lon) / 360
        assert area > 0 and area <= spherical_cap_difference and area <= earth_surface_area_in_km
        return area


class GeoShape:
    def __init__(self, shapefile_path):
        sf = shapefile.Reader(shapefile_path)
        self.shapes = sf.shapes()
        assert len(self.shapes) > 0
        assert all([shape.shapeTypeName == 'POLYGON' for shape in self.shapes])
        self.shapes_data = None

    def random_geopoint(self):
        if self.shapes_data is None:
            self.shapes_data = []
            for shape in self.shapes:
                bounds = GeoRect.from_shapefile_bbox(shape.bbox)
                area = GeoRect.area(bounds)
                self.shapes_data.append({"outline": shape, "bounds": bounds, "area": area, "area_relative_prefix_sum": 0})
            total = sum([shape["area"] for shape in self.shapes_data])
            area_prefix_sum = 0
            for shape in self.shapes_data:
                area_prefix_sum += shape["area"]
                shape["area_relative_prefix_sum"] = area_prefix_sum / total

        i = 0
        while i < 250:
            area_relative_prefix_sum = random.random()
            shape = None
            for shape_candidate in self.shapes_data:
                if area_relative_prefix_sum < shape_candidate["area_relative_prefix_sum"]:
                    shape = shape_candidate
                    break

            geopoint = GeoPoint.random(shape["bounds"])
            point = geopoint.to_shapely_point()
            polygon = shapely.geometry.shape(shape["outline"])
            contains = polygon.contains(point)
            if contains: return geopoint
        raise ValueError("cannot seem to find a point in the shape's bounding box that's within the shape – is your data definitely okay (it may well be if it's a bunch of spread-out islands)? if you're sure, you'll need to raise the iteration limit in this function")


class MapTileStatus:
    PENDING = 1
    CACHED = 2
    DOWNLOADING = 3
    DOWNLOADED = 4
    ERROR = 5


class MapTile:
    tile_path_template = None
    tile_url_template = None
    def __init__(self, zoom, direction, x, y):
        self.zoom = zoom
        self.direction = direction
        self.x = x
        self.y = y

        # initialize the other variables
        self.status = MapTileStatus.PENDING
        self.image = None
        self.filename = None
        if (MapTile.tile_path_template):
            self.filename = MapTile.tile_path_template.format(
                angle_if_oblique=("" if self.direction.is_downward() else f"deg{self.direction.angle}"),
                zoom=self.zoom,
                x=self.x,
                y=self.y,
                hash=hashlib.sha256(MapTile.tile_url_template.encode("utf-8")).hexdigest()[:8])

    def __repr__(self): return f"MapTile({self.zoom}, {self.direction}, {self.x}, {self.y})"

    def zoomed(self, zoom_delta):
        zoom = self.zoom + zoom_delta
        fac = (2 ** zoom_delta)
        return MapTileGrid([[MapTile(zoom, self.direction, self.x * fac + x, self.y * fac + y) for y in range(0, fac)] for x in range(0, fac)])

    def load(self):
        if self.filename is None: self.download()
        else:
            try:
                self.image = Image.open(self.filename)
                self.image.load()
                self.status = MapTileStatus.CACHED
            except IOError:
                self.download()

    def download(self):
        self.status = MapTileStatus.DOWNLOADING
        try:
            url = MapTile.tile_url_template.format(angle=self.direction.angle, x=self.x, y=self.y, zoom=self.zoom)
            r = requests.get(url, headers={'User-Agent': USER_AGENT})
        except requests.exceptions.ConnectionError:
            self.status = MapTileStatus.ERROR
            return

        if r.status_code != 200:
            LOGGER.warning(f"Unable to download {self}, status code {r.status_code}.")
            self.status = MapTileStatus.ERROR
            return
        
        data = r.content
        self.image = Image.open(io.BytesIO(data))

        assert self.image.mode == "RGB"
        assert self.image.size == (TILE_SIZE, TILE_SIZE)

        if self.filename is not None:
            d = os.path.dirname(self.filename)
            if not os.path.isdir(d):
                os.makedirs(d)
            with open(self.filename, 'wb') as f:
                f.write(data)
        self.status = MapTileStatus.DOWNLOADED


class MapTileGrid:
    def __init__(self, maptiles):
        self.maptiles = maptiles
        self.width = len(maptiles)
        self.height = len(maptiles[0])
        self.image = None

    def __repr__(self):
        return f"MapTileGrid({self.maptiles})"

    @classmethod
    def from_georect(cls, georect, zoom, direction):
        bottomleft = georect.sw.to_maptile(zoom, direction)
        topright = georect.ne.to_maptile(zoom, direction)
        if bottomleft.x > topright.x: bottomleft.x, topright.x = topright.x, bottomleft.x
        if bottomleft.y < topright.y: bottomleft.y, topright.y = topright.y, bottomleft.y

        maptiles = []
        for x in range(bottomleft.x, topright.x + 1):
            col = []
            for y in range(topright.y, bottomleft.y + 1):
                maptile = MapTile(zoom, direction, x, y)
                col.append(maptile)
            maptiles.append(col)

        return cls(maptiles)

    def at(self, x, y):
        if x < 0: x += self.width
        if y < 0: y += self.height
        return self.maptiles[x][y]

    def flat(self): return [maptile for col in self.maptiles for maptile in col]

    def has_high_quality_imagery(self, quality_check_delta):
        corners = [self.at(x, y).zoomed(quality_check_delta).at(x, y) for x in [0, -1] for y in [0, -1]]
        all_good = True
        for c in corners:
            c.load()
            if c.status == MapTileStatus.ERROR:
                all_good = False
                break
        return all_good

    def download(self):
        prog_thread = threading.Thread()#target=prog.loop)
        prog_thread.start()
        tiles = self.flat()
        random.shuffle(tiles)
        threads = max(self.width, self.height)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor: {executor.submit(maptile.load): maptile for maptile in tiles}

        missing_tiles = [maptile for maptile in self.flat() if maptile.status == MapTileStatus.ERROR]
        if 0 < len(missing_tiles) < 0.02 * len(self.flat()):
            if VERBOSITY != "quiet": print("Retrying missing tiles...")
            for maptile in missing_tiles: maptile.load()

        prog_thread.join()
        # prog.cleanup()

        missing_tiles = [maptile for maptile in self.flat() if maptile.status == MapTileStatus.ERROR]
        if missing_tiles: raise RuntimeError(f"unable to load one or more map tiles: {missing_tiles}")

    def stitch(self):
        image = Image.new('RGB', (self.width * TILE_SIZE, self.height * TILE_SIZE))
        for x in range(0, self.width):
            for y in range(0, self.height): image.paste(self.maptiles[x][y].image, (x * TILE_SIZE, y * TILE_SIZE))
        self.image = image


class MapTileImage:
    def __init__(self, image):
        self.image = image

    def save(self, path, quality=90):
        self.image.save(path, quality=quality)

    def crop(self, zoom, direction, georect):
        left, bottom = WebMercator.project(georect.sw, zoom)  # sw_x, sw_y
        right, top = WebMercator.project(georect.ne, zoom)  # ne_x, ne_y
        if direction.is_oblique():
            left, bottom = ObliqueWebMercator.project(georect.sw, zoom, direction)
            right, top = ObliqueWebMercator.project(georect.ne, zoom, direction)

        if left > right: left, right = right, left
        if bottom < top: bottom, top = top, bottom

        left_crop = round(TILE_SIZE * (left % 1))
        bottom_crop = round(TILE_SIZE * (1 - bottom % 1))
        right_crop = round(TILE_SIZE * (1 - right % 1))
        top_crop = round(TILE_SIZE * (top % 1))

        crop = (left_crop, top_crop, right_crop, bottom_crop)
        self.image = ImageOps.crop(self.image, crop)

    def scale(self, width, height): self.image = self.image.resize((round(width), round(height)), resample=Image.Resampling.LANCZOS)

    def enhance(self):
        contrast = 1.07
        brightness = 1.01
        self.image = ImageEnhance.Contrast(self.image).enhance(contrast)
        self.image = ImageEnhance.Brightness(self.image).enhance(brightness)


##### Graph Dataset Construction #####

def crop_image_only_outside(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    img = np.array(img)
    mask = img>tol
    if img.ndim==3: mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    img = img[row_start:row_end,col_start:col_end]
    # resize to original size
    img = Image.fromarray(img)
    img = img.resize((n,m))
    return img

def download_sat(   
        tile_path_template: str = str(Path.cwd() / 'data/images'),# + '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.png',
        image_path_template: str = str(Path.cwd() / 'data/images'),# + '/{latitude},{longitude}-{width}x{height}m-z{zoom}-{image_height}x{image_width}.png',
        max_tries: int = 10,
        tile_url_template: str = "googlemaps",
        shapefile: str = "aerialbot/shapefiles/usa/usa.shp",
        point: tuple = (51.243594, -0.576837),
        width: int = 1000,
        height: int = 1000,
        image_width: int = 2048,
        image_height: int = 2048,
        max_meters_per_pixel: float = 0.4,
        apply_adjustments: bool = True,
        image_quality: int = 100):

    tile_path_template += '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.jpg'
    image_path_template += '/raw/{latitude},{longitude}-{width}x{height}m-z{zoom}-{image_height}x{image_width}.jpg'

    direction = ViewDirection("downward")
    if tile_url_template == "googlemaps": tile_url_template = "https://khms2.google.com/kh/v={google_maps_version}?x={x}&y={y}&z={zoom}"
    elif tile_url_template == "navermap": tile_url_template = "https://map.pstatic.net/nrb/styles/satellite/{naver_map_version}/{zoom}/{x}/{y}.jpg?mt=bg"

    if "{google_maps_version}" in tile_url_template:
        google_maps_version = GOOGLE_MAPS_VERSION_FALLBACK
        if direction.is_oblique(): google_maps_version = GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK

        try:
            google_maps_page = requests.get("https://maps.googleapis.com/maps/api/js", headers={"User-Agent": USER_AGENT}).content
            match = re.search(rb'null,\[\[\"https:\/\/khms0\.googleapis\.com\/kh\?v=([0-9]+)', google_maps_page)
            if direction.is_oblique(): match = re.search(rb'\],\[\[\"https:\/\/khms0\.googleapis\.com\/kh\?v=([0-9]+)', google_maps_page)
            if match: google_maps_version = match.group(1).decode('ascii')
        except:
            print("Failed to determine current Google Maps version, falling back to", google_maps_version)
        tile_url_template = tile_url_template.replace("{google_maps_version}", google_maps_version)

    if "{naver_map_version}" in tile_url_template:
        naver_map_version = requests.get("https://map.pstatic.net/nrb/styles/satellite.json", headers={'User-Agent': USER_AGENT}).json()["version"]
        tile_url_template = tile_url_template.replace("{naver_map_version}", naver_map_version)

    MapTile.tile_path_template = tile_path_template
    MapTile.tile_url_template = tile_url_template

    geowidth = width
    geoheight = height
    foreshortening_factor = 1
    if direction.is_oblique(): foreshortening_factor = math.sqrt(2)

    # process max_meters_per_pixel setting
    if image_width is None and image_height is None: assert max_meters_per_pixel is not None
    elif image_height is None: max_meters_per_pixel = (max_meters_per_pixel or 1) * (width / image_width)
    elif image_width is None: max_meters_per_pixel = (max_meters_per_pixel or 1) * (height / image_height) / foreshortening_factor
    else:
        if width / image_width <= (height / image_height) / foreshortening_factor: max_meters_per_pixel = (max_meters_per_pixel or 1) * (width / image_width)
        else: max_meters_per_pixel = (max_meters_per_pixel or 1) * (height / image_height) / foreshortening_factor

    # process image width and height for scaling
    if image_width is not None or image_height is not None:
        if image_height is None: image_height = height * (image_width / width) / foreshortening_factor
        elif image_width is None: image_width = width * (image_height / height) * foreshortening_factor

    ############################################################################
    if shapefile is None and point is None: raise RuntimeError("neither shapefile path nor point configured")
    elif point is None: shapes = GeoShape(shapefile)

    for tries in range(0, max_tries):
        if tries > max_tries: raise RuntimeError("too many retries – maybe there's no internet connection? either that, or your max_meters_per_pixel setting is too low")
        if point is None: p = shapes.random_geopoint()
        else: p = GeoPoint(point[0], point[1])
        zoom = p.compute_zoom_level(max_meters_per_pixel)
        rect = GeoRect.around_geopoint(p, geowidth, geoheight)
        grid = MapTileGrid.from_georect(rect, zoom, direction)
        if point is not None: break

    ############################################################################
    grid.download()
    grid.stitch()
    image = MapTileImage(grid.image)
    image.crop(zoom, direction, rect)

    if image_width is not None or image_height is not None: image.scale(image_width, image_height)

    ## Added
    if image_width is None and image_height is None:
        # Resize image to 1000x1000
        image.scale(1000, 1000)
        image_height = 1000
        image_width = 1000

    if apply_adjustments: image.enhance()

    image_path = image_path_template.format(latitude=p.lat, longitude=p.lon, width=width, height=height, zoom=zoom, image_height=image_height, image_width=image_width)

    d = os.path.dirname(image_path) 
    if not os.path.isdir(d): os.makedirs(d)
    image.save(image_path, image_quality)
    image_path = image_path.split('/')[-1]
    return image_path

def generate_spiral_coordinates(num_points=10000, radius_increment=0.00001, theta_increment=0.1, origin=(0, 0)):
    coordinates = []
    for i in range(num_points):
        radius_y = radius_increment * i
        radius_x = radius_increment * i * 0.4
        theta = theta_increment * i
        x = origin[0] + radius_x * math.cos(theta)
        y = origin[1] + radius_y * math.sin(theta)
        coordinates.append((round(x, 7), round(y, 7)))
    return coordinates

def streetview_image(pano, point, cwd):
    panos = []
    distances = []
    for pan in pano: 
        if pan.heading is not None: 
            distances.append(haversine((pan.lat, pan.lon), point, unit=Unit.METERS))
    distances = sorted(range(len(distances)), key=lambda k: distances[k])

    if len(distances) > 5: 
        distances = distances[:5] # Reduce distances to first 5 values at most

    for idx, i in enumerate(distances):
        p = pano[i].pano_id
        date = pano[i].date
        heading = round(pano[i].heading, 3)
        image_name = f'{point}_{date}_{idx}.jpg'
        try:
            if Path(f'{cwd}/data/images/raw/{image_name}').is_file(): 
                panos.append([image_name, heading])
            else:
                panorama = get_panorama(pano_id=p)

                if type(panorama) != Image.Image: 
                    continue # If not an image, try next pano
                else:
                    panorama = crop_image_only_outside(panorama)
                    panorama = panorama.crop((0, 0.25 * panorama.size[1], panorama.size[0], 0.75 * panorama.size[1]))
                    panorama = panorama.resize((2048, 512))
                    panonumpy = np.array(panorama)
                    panorama = Image.fromarray(panonumpy)
                    panorama.save(f'{cwd}/data/images/raw/{image_name}')
                    panos.append([image_name, heading])
        except: 
            pass # if fails gets the next further away pano
    return panos

def get_streetview(point, cwd='/vol/research/deep_localisation/sat/', pano=False):
    panos = []
    pano = None
    try:
        pano = search_panoramas(lat=point[0], lon=point[1])
        p = streetview_image(pano, point, cwd) # List of [[image_name, heading], ...]
        panos.extend(p)
    except:
        points = generate_spiral_coordinates(origin=point)
        for p in tqdm(points, desc='Searching for Panorama', position=1, leave=True):
            try:
                pano = search_panoramas(lat=p[0], lon=p[1])
                if pano is not None:
                    p = streetview_image(pano, point, cwd) 
                    panos.extend(p)
                    break
            except: 
                pass
        
        if pano is None:
            print(f'Missing: {point}')
            print()
    return panos

@ray.remote
def download_junction_data(node_list, positions, cwd, bar=None, width=20):
    missing = 0
    sub_dict = {}
    for node in tqdm(node_list, 'Downloading Junction Data', position=0) if bar is None else node_list:
        pos = positions[node]

        panos = get_streetview(pos, cwd=cwd)
        sat_path = download_sat(point=pos, width=width, height=width, image_width=None, image_height=None, max_meters_per_pixel=0.1)

        if len(panos) == 0:
            missing = missing + 1
            # selects random pov from other nodes - CHANGE to selecting closest node's pov
            pano_paths = sub_dict[list(sub_dict.keys())[random.randint(0, len(sub_dict) - 1)]]['pov']
            headings = [0 for _ in range(len(pano_paths))]
        else:
            pano_paths = [pair[0] for pair in panos]
            headings = [pair[1] for pair in panos]

        sub_dict[node] = {'sat': sat_path, 'pov': pano_paths, 'heading': headings}

        if bar is not None: bar.update.remote(1)
    return sub_dict, missing


def warp_pano(input_image):
    top_input = input_image[input_image.shape[0]//2:, :]
    top_input = cv2.rotate(top_input, cv2.ROTATE_90_CLOCKWISE)
    top_image, _ = polarTransform.convertToCartesianImage(image=top_input, hasColor=True, center='bottom-middle',
                                                        initialAngle=0, finalAngle=2*np.pi)
    top_image = cv2.rotate(top_image, cv2.ROTATE_180)

    bottom_image = np.roll(input_image, input_image.shape[1]//2, axis=1)
    bottom_image = bottom_image[bottom_image.shape[0]//2:, :]
    bottom_image = cv2.rotate(bottom_image, cv2.ROTATE_90_CLOCKWISE)
    bottom_image, _ = polarTransform.convertToCartesianImage(image=bottom_image, hasColor=True, center='bottom-middle',
                                                                    initialAngle=0, finalAngle=2*np.pi)
    combined = np.concatenate((top_image, bottom_image), axis=0)
    polared = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return polared



class GraphData():
    def __init__(self, args):
        self.args = args
        self.to_ten = T.ToTensor()
        self.coords = {'london': (51.51492, -0.09057), 'brussels': (50.853481, 4.355358), 
                        'boston': (42.358191, -71.060911), 'tokyo': (35.680886, 139.777483), 'chicago': (41.883181, -87.629645),
                        'new york': (40.7484, -73.9857), 'philly': (39.952364, -75.163616), 'singapore': (1.280999, 103.845047),
                        'seoul': (37.566989, 126.989192), 'hong kong': (22.280144, 114.158341), 'guildford': (51.246194, -0.57425)}

        if self.args.dataset == 'spagbol':
            self.train_coords = [self.coords[point] for point in self.args.train_localisations]
            self.test_coords = [self.coords[point] for point in self.args.test_localisations]
        else:
            self.train_coords = self.args.train_localisation_vigor
            self.test_coords = self.args.test_localisation_vigor

        self.graphs, self.test_graphs = {}, {}
        self.train_walks, self.val_walks, self.test_walks = {}, {}, {} # {point: walks}

        for coord in self.train_coords: 
            self.prepare_graph(coord)
        for coord in self.test_coords: 
            self.prepare_graph(coord, stage='test')

    def prepare_graph(self, point=None, stage='train'):
        # Download graph or open if available
        if not Path(f'{self.args.data}/spagbol/graphs/raw/graph_{point}_{self.args.width}.pt').is_file():
            corpus_graph = self.create_graph(centre=point, dist=self.args.width, cwd=self.args.path, workers=self.args.workers)
            torch.save(corpus_graph, f'{self.args.path}/data/{self.args.dataset}/graphs/raw/graph_{point}_{self.args.width}.pt')
            src_images = Path(f'{self.args.path}/data/{self.args.dataset}/images/raw/')
            dst_database = Path(f'{self.args.path}/data/{self.args.dataset}/images/lmdb/')
            image_paths = {image_path.stem: image_path for image_path in sorted(src_images.rglob(f"*jpg"))}
            write_database(image_paths, dst_database) # Write to LMDB
        else: 
            corpus_graph = torch.load(f'{self.args.data}/spagbol/graphs/raw/graph_{point}_{self.args.width}.pt')
            
        if stage == 'train': 
            self.graphs[point] = corpus_graph
        else: 
            self.test_graphs[point] = corpus_graph

        # Get midpoint of graph - selecting lower right quadrant as val set
        if stage != 'test':
            poses = []
            for node in corpus_graph.nodes: 
                poses.append(corpus_graph.nodes[node]["pos"])
            poses = list(set(poses))
            lats, lons = [p[0] for p in poses], [p[1] for p in poses]
            min_lat, min_lon, max_lat, max_lon = min(lats), min(lons), max(lats), max(lons)
            lat_diff = max_lat - min_lat
            lon_diff = max_lon - min_lon
            lat_thres = (lat_diff * 0.33) + min_lat
            lon_thres = (lon_diff * 0.33) + min_lon
            train_count, val_count = 0, 0

            # Seperate nodes into train and val
            train_nodes, val_nodes = [], []
            for node in corpus_graph.nodes:
                pos = corpus_graph.nodes[node]["pos"]
                if pos[0] < lat_thres and pos[1] < lon_thres: 
                    val_nodes.append(node)
                    val_count += 1
                else: 
                    train_nodes.append(node)
                    train_count += 1

            # Create train and val graphs - only required for calculating walks
            train_graph = nx.Graph()
            for node in train_nodes: 
                train_graph.add_node(node)
                for key in ['sat', 'pov', 'pos', 'north', 'yaws', 'heading']:
                    if key in corpus_graph.nodes[node].keys():
                        train_graph.nodes[node][key] = corpus_graph.nodes[node][key]

            for node in train_nodes:
                for neighbour in corpus_graph.neighbors(node):
                    if neighbour in train_nodes: 
                        train_graph.add_edge(node, neighbour)

            val_graph = nx.Graph()
            for node in val_nodes:
                val_graph.add_node(node)
                for key in ['sat', 'pov', 'pos', 'north', 'yaws', 'heading']:
                    if key in corpus_graph.nodes[node].keys():
                        val_graph.nodes[node][key] = corpus_graph.nodes[node][key]

            for node in val_nodes:
                for neighbour in corpus_graph.neighbors(node):
                    if neighbour in val_nodes: 
                        val_graph.add_edge(node, neighbour)

            # TEMPORARY
            # walks = self.exhaustive_walks(corpus_graph, corpus_graph.nodes, 'full', self.args.walk)
            # print(f'city: {point}, walks: {len(walks)}')
            ############################# REMOVE

            sets = ['train', 'val']
            for s in sets:
                walk_name = f'{self.args.path}/data/{self.args.dataset}/graphs/walks/{s}_walks_{point}_{self.args.width}_{self.args.walk}.npy'

                if Path(walk_name).is_file():
                    walks = np.load(walk_name, allow_pickle=True)
                elif not Path(walk_name).is_file():
                    if s == 'train':
                        walks = self.exhaustive_walks(train_graph, train_nodes, s, self.args.walk)
                    else: 
                        walks = self.exhaustive_walks(val_graph, val_nodes, s, self.args.walk)
                    np.save(walk_name, walks)



                if s == 'train': 
                    self.train_walks[point] = walks
                else: 
                    self.val_walks[point] = walks

        else: # test - use whole corpus_graph
            walk_name = f'{self.args.path}/data/{self.args.dataset}/graphs/walks/full_walks_{point}_{self.args.width}_{self.args.walk}.npy'

            if Path(walk_name).is_file() and self.args.walk == self.args.walk:
                walks = np.load(walk_name, allow_pickle=True)
            elif not Path(walk_name).is_file():
                walks = self.exhaustive_walks(corpus_graph, corpus_graph.nodes, 'full', self.args.walk)
                np.save(walk_name, walks)
            self.test_walks[point] = walks

    def create_graph(self, centre=(51.509865, -0.118092), dist=1000, cwd='/home/ts00987/spagbol/data', workers=4):
        g = ox.graph.graph_from_point(center_point=centre, dist=dist, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, 
                                    truncate_by_edge=False, clean_periphery=None, custom_filter=None)
        g = ox.projection.project_graph(g, to_latlong=True)
        graph = nx.Graph()

        for n in g.nodes(data=True): 
            position = (n[1]['y'], n[1]['x'])
            graph.add_node(n[0], pos=position)

        for start, end in g.edges(): graph.add_edge(start, end)

        positions = nx.get_node_attributes(graph, 'pos')

        ray.init(include_dashboard=False, num_cpus=workers)
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        node_list = list(graph.nodes)
        bar = remote_tqdm.remote(total=len(node_list))
        node_lists = [node_list[i::workers] for i in range(workers)]
        street_getters = [download_junction_data.remote(node_lists[i], positions, cwd, bar, width=self.args.sat_width) for i in range(workers)]
        graph_images = ray.get(street_getters)


        # node_list = list(graph.nodes)
        # image_paths, missing = download_junction_data(node_list, positions, cwd)

    
        image_paths = dict((key, d[key]) for d, _ in graph_images for key in d)
        missing = sum([m for _, m in graph_images])
        print(f'Missing: {missing}, {round((missing / len(node_list))*100, 2)}%')

        bar.close.remote()
        ray.shutdown()

        for node in image_paths.keys():
            graph.nodes[node]['pov'] = image_paths[node]['pov']
            graph.nodes[node]['sat'] = image_paths[node]['sat']
            graph.nodes[node]['north'] = [round(i, 3) for i in image_paths[node]['heading']]

        # Add potential yaw for each node, list of north-aligned angles between neighbours
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            positions = [graph.nodes[n]['pos'] for n in neighbours]
            angles = []
            for pos in positions:
                og = graph.nodes[node]["pos"]
                angles.append(round(Geodesic.WGS84.Inverse(og[0], og[1], pos[0], pos[1])['azi1'], 3))
            graph.nodes[node]['yaws'] = angles # Concerning satellite image
            graph.nodes[node]['heading'] = 180 # Default in POV image centre

        return graph
        
    def random_walk(self, graph, start_node, length):
        walk = [start_node]
        while len(walk) < length:
            neighbours = list(graph.neighbors(walk[-1]))
            if len(walk) > 1:
                if walk[-2] in neighbours:
                    neighbours.remove(walk[-2])
            if len(neighbours) == 0: break
            else: walk.append(random.choice(neighbours))
        walk = tuple(walk)
        return walk
        
    def exhaustive_walks(self, graph, nodes, stage='Train', walk_length=3):
        attempts = 10000
        node_walks = set()
        for node in tqdm(nodes, desc=f'Generating {stage} Walks - Length {walk_length}'):
            for attempts in range(attempts):
                walk = self.random_walk(graph, node, walk_length) # tuple not list
                if len(walk) == walk_length: node_walks.add(walk)            
        node_walks = list(node_walks)
        return node_walks
        

to_pil = T.ToPILImage()

def remove_extension(path_string):
    if path_string[-4:] == '.png' or path_string[-4:] == '.jpg': return path_string[:-4]
    else: return path_string

# Selects walks and pov images - random for train, else first
def select_index(input, stage):
    if stage == 'train': 
        if isinstance(input[0], np.ndarray): output = random.choice(input)
        elif isinstance(input[0], str): output = random.randint(0, len(input) - 1)
    else:
        if isinstance(input[0], np.ndarray): output = input[0]
        elif isinstance(input[0], str): output = 0
    return output

def select_graph(dataset, item):
    if dataset == 'spagbol':
        coord = item.split('_')[0].split('(')[1].split(')')[0]
        coord = tuple(map(float, coord.split(', ')))
    else: coord = item.split('_')[0]
    return coord


class GraphDataset(Dataset):
    def __init__(self, args, graphs, stage='train'):
        self.args = args
        self.stage = stage
        self.graphs = graphs.graphs if stage != 'test' else graphs.test_graphs
        self.walks = graphs.train_walks if stage == 'train' else graphs.val_walks if stage == 'val' else graphs.test_walks

        self.stage_walks = {}
        for coord in self.walks:
            for walk in self.walks[coord]:
                key = f'{coord}_{walk[0]}'
                if key not in self.stage_walks: 
                    self.stage_walks[key] = []
                if len(walk) == self.args.walk:
                    self.stage_walks[key].append(walk)
        self.stage_keys = list(self.stage_walks.keys())

        self.lmdb = ImageDatabase(f'{self.args.path}/data/{self.args.dataset}/lmdb/', readahead=True)
        self.lmdb_keys = list(self.lmdb.keys)
        self.stage = stage

    def __len__(self): 
        return len(self.stage_keys)

    def __getitem__(self, index):
        item = self.stage_keys[index]
        graph = self.graphs[select_graph(self.args.dataset, item)]
        walk_nodes = self.stage_walks[item]

        # Get random walk from nodes exhaustive list
        walk_nodes = select_index(walk_nodes, self.stage)
        walk_nodes = [int(w) for w in walk_nodes]
        walk = graph.subgraph(walk_nodes).copy() # deepcopy() ?

        # nx.set_node_attributes(walk, walk.nodes[walk_nodes[0]]['pos'], 'start_point')
        poses = nx.get_node_attributes(walk, 'pos')

        for idx, node in enumerate(walk_nodes):
            yaws = np.array([y for y in walk.nodes[node]['yaws']]) # precalculated at graph construction time
            pano_keys = walk.nodes[node]['pov']

            if self.args.limit_povs: pano_keys = pano_keys[:self.args.limit_povs]
            indice = select_index(pano_keys, self.stage)
            if self.args.dataset == 'spagbol':
                pano_key = pano_keys[indice]
            else: pano_key = pano_keys

            pano_key = remove_extension(pano_key)
            pano = np.array(self.lmdb[pano_key])
            _, width, _ = pano.shape

            if self.args.dataset == 'spagbol':
                norths = walk.nodes[node]['north']
                north = norths[indice]
            else:
                north = 0

            # North aligned pano and yaws
            north_pixels = int(((north)/360) * width)
            pano = np.roll(pano, north_pixels, axis=1)
            
            # Crops to opposing angle of previous node
            if idx < len(walk_nodes) - 1 and self.args.roll_direction:                
                target_node = poses[node]
                current_node = poses[walk_nodes[idx+1]]
                angle = Geodesic.WGS84.Inverse(current_node[0], current_node[1], target_node[0], target_node[1])['azi1']
                angle_pixels = int((angle/360) * width)
                pano = np.roll(pano, angle_pixels, axis=1)
                yaws = np.array([(y+angle)%360 for y in yaws])
            yaws = np.array([(y+180)%360 for y in yaws])
            # yaws = np.sort(yaws)

            # Crop Pano to FOV from centre of image
            pov_pixel_width = int((self.args.fov/360) * width)
            pano = pano[:, width//2 - pov_pixel_width//2: width//2 + pov_pixel_width//2]

            pano = to_ten(pano)
            pano = F.resize(pano, (256, 256), antialias=True)

            sat_key = walk.nodes[node]['sat']
            sat_key = remove_extension(sat_key)
            sat = to_ten(self.lmdb[sat_key])
            sat = F.resize(sat, (256, 256), antialias=True)

            half_width = (360/self.args.num_yaws) / 2
            shifted_yaws = yaws - half_width

            bin_ranges = np.arange(-half_width, 360, half_width*2)
            bins = np.digitize(shifted_yaws, bin_ranges, right=False)
            bins -= 1
            m = np.zeros(self.args.num_yaws)
            m[bins] = 1

            walk.nodes[node]['pov_image'] = pano
            walk.nodes[node]['sat_image'] = sat
            walk.nodes[node]['yaws_image'] = m
            walk.nodes[node]['pos_image'] = poses[node]
            walk.nodes[node]['north_image'] = north
            

        for node in walk.nodes:
            del walk.nodes[node]['pov'] 
            del walk.nodes[node]['sat']
            del walk.nodes[node]['yaws']
            if self.args.dataset == 'spagbol':
                del walk.nodes[node]['north']
                del walk.nodes[node]['heading']

            # make sure remaining are tensors
            for key in walk.nodes[node]:
                if (not isinstance(walk.nodes[node][key], torch.Tensor) or walk.nodes[node][key].dtype != torch.float32) and not isinstance(walk.nodes[node][key], str):
                    walk.nodes[node][key] = torch.tensor(walk.nodes[node][key], dtype=torch.float32)

        walk = from_networkx(walk)
        return walk




if __name__ == '__main__':
    # If this file is run directly - download SpaGBOL data
    from configs.config import return_defaults

    args = return_defaults()

    graphs = GraphData(args)
    dataset = GraphDataset(args, graphs, stage='train')
