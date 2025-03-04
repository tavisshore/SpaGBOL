#!/usr/bin/env python3
from pathlib import Path
from tqdm import tqdm
import numpy as np
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
from PIL import Image, ImageEnhance, ImageOps
from streetview import search_panoramas, get_panorama
from haversine import haversine, Unit
from datetime import datetime

import ray

Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 256 
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  
GOOGLE_MAPS_VERSION_FALLBACK = '934'
GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK = '148'
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15"
LOGGER = None
VERBOSITY = None


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
    """
    Removes black spaces in images
    """
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
        tile_path_template: str = str(Path.cwd() / 'data/images') + '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.jpg',
        image_path_template: str = str(Path.cwd() / 'data/images') + '/raw/{latitude},{longitude}-{width}x{height}m-z{zoom}.jpg',
        max_tries: int = 10,
        tile_url_template: str = "googlemaps",
        point: tuple = (51.243594, -0.576837),
        width: int = 2000,
        height: int = 2000,
        max_meters_per_pixel: float = 0.3,
        apply_adjustments: bool = True,
        image_quality: int = 100):
    """
    Downloads satellite images from online services
    """
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

    for tries in range(0, max_tries):
        if tries > max_tries: raise RuntimeError("too many retries – maybe there's no internet connection? either that, or your max_meters_per_pixel setting is too low")
        p = GeoPoint(point[0], point[1])
        zoom = p.compute_zoom_level(max_meters_per_pixel)
        rect = GeoRect.around_geopoint(p, geowidth, geoheight)
        grid = MapTileGrid.from_georect(rect, zoom, direction)
        if point is not None: break

    ############################################################################
    grid.download()
    grid.stitch()
    image = MapTileImage(grid.image)
    image.crop(zoom, direction, rect)

    image_width, image_height = 1000, 1000
    image.scale(image_width, image_height)

    if apply_adjustments: 
        image.enhance()

    image_path = image_path_template.format(latitude=p.lat, longitude=p.lon, width=width, height=height, zoom=zoom)

    d = os.path.dirname(image_path) 
    if not os.path.isdir(d): os.makedirs(d)
    image.save(image_path, image_quality)
    image_path = image_path.split('/')[-1]
    return image_path






def generate_spiral_coordinates(num_points=10000, radius_increment=0.00001, theta_increment=0.1, origin=(0, 0)):
    """
    When an image isn't available at an exact coordinate, spiral outwards to find the closest image
    """
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
    """
    Downloads streetview images for each node
    """
    panos = []
    distances = []
    for pan in pano: 
        if pan.heading is not None: 
            distances.append(haversine((pan.lat, pan.lon), point, unit=Unit.METERS))
    distances = sorted(range(len(distances)), key=lambda k: distances[k])

    counter = 0
    while len(panos) < 5:
        i = distances[counter]
        p = pano[i].pano_id
        date = pano[i].date
        if date is not None: 
            date = date.replace('-', '')
            date = datetime.strptime(date, '%Y%m%d').strftime('%d-%m-%Y')
            if date > '01-01-2024': # If data is newer, skip
                i += 1
                continue
        else:
            continue

        heading = round(pano[i].heading, 3)
        image_name = f'{point}_{date}.jpg'
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
        i += 1
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
