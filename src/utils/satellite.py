#!/usr/bin/env python3
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
import math
import random
from PIL import Image
from streetview import search_panoramas, get_panorama
from haversine import haversine, Unit
from datetime import datetime
import utm
from image_downloading import download_image
Image.MAX_IMAGE_PIXELS = None

##### Graph Dataset Construction #####
DATE_LIMIT = datetime.strptime('2025-01', '%Y-%m')

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


def download_sat_point(point, cwd, zoom=20):
    image_name = f'{point[0]}_{point[1]}_{zoom}.jpg'
    image_path = f'{cwd}/raw/images/{image_name}'
    if not Path(image_path).is_file():
        headers = {
                    'cache-control': 'max-age=0',
                    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'none',
                    'sec-fetch-user': '?1',
                    'upgrade-insecure-requests': '1',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
                }

        lat, lon, zn, zl = utm.from_latlon(point[0], point[1])

        # Get satellite images that cover area + 100m each side
        lat_min = lat - 50
        lat_max = lat + 50
        lon_min = lon - 50
        lon_max = lon + 50
        # Back to lat
        lat_min, lon_min = utm.to_latlon(lat_min, lon_min, zn, zl)
        lat_max, lon_max = utm.to_latlon(lat_max, lon_max, zn, zl)

        tl = (lat_max, lon_min)
        br = (lat_min, lon_max)

        lat1, lon1 = re.findall(r'[+-]?\d*\.\d+|d+', str(tl))
        lat2, lon2 = re.findall(r'[+-]?\d*\.\d+|d+', str(br))
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)

        img = download_image(lat1, lon1, lat2, lon2, zoom, 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', headers, 256, 3)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        img = img.resize((512, 512))
        
        img.save(image_path)

        # close file?
        img.close()

    return image_name





def generate_spiral_coordinates(num_points=1000, radius_increment=0.00001, theta_increment=0.1, origin=(0, 0)):
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
    - sort this code out - added function that adds random noise to lat and lon if no images found
    """
    new_pan = []
    panos = []
    distances = []
    secondaries = [] # examples without explicit date, only use if no other options

    for pan in pano: 
        if pan.heading is not None: # Absolutely necessary
            if (pan.date is not None and datetime.strptime(pan.date, '%Y-%m') < DATE_LIMIT): 
                distances.append(haversine((pan.lat, pan.lon), point, unit=Unit.METERS))
                new_pan.append(pan)
            elif pan.date is None:
                secondaries.append(pan)

    if len(distances) < 5:
        for pan in secondaries: 
            if pan.heading is not None:
                distances.append(haversine((pan.lat, pan.lon), point, unit=Unit.METERS))    
                new_pan.append(pan)

    # indices of the closest panoramas increasing distance
    distances = sorted(range(len(distances)), key=lambda k: distances[k])

    
    for i in distances:
        p = new_pan[i].pano_id
        date = new_pan[i].date

        heading = round(new_pan[i].heading, 5)
        image_name = f'{point[0]}_{point[1]}_{date}_street.jpg'
        image_path = f'{cwd}/raw/images/{image_name}'
        # try:
        if Path(image_path).is_file(): 
            panos.append([image_name, heading])
        else:
            try:
                panorama = get_panorama(pano_id=p, multi_threaded=True) # True is faster but may fail

                if isinstance(panorama, Image.Image):
                    panorama = crop_image_only_outside(panorama)
                    panorama = panorama.crop((0, 0.25 * panorama.size[1], panorama.size[0], 0.75 * panorama.size[1]))
                    panorama = panorama.resize((2048, 512))
                    panonumpy = np.array(panorama)
                    panorama = Image.fromarray(panonumpy)
                    panorama.save(image_path)
                    panos.append([image_name, heading])
            except:
                # print('Error downloading panorama')
                pass
        if len(panos) >= 5:
            break

    return panos

def retry_pano(point):
    # randomly add or subtract 0.00001 to lat and lon
    point = (point[0] + random.uniform(-0.00001, 0.00001), point[1] + random.uniform(-0.00001, 0.00001))
    # search for panoramas again
    return point

def get_streetview(point, cwd='/vol/research/deep_localisation/sat/', pano=False):
    panos = []
    pano = []

    pano = search_panoramas(lat=point[0], lon=point[1])

    while len(pano) == 0:
        point = retry_pano(point)
        pano = search_panoramas(lat=point[0], lon=point[1])

    p = streetview_image(pano, point, cwd) # List of [[image_name, heading], ...]

    panos.extend(p)
    return panos

def download_junction_data(node_list, positions, cwd):
    sub_dict = {}
    for node in tqdm(node_list, 'Downloading Junction Data', position=0):
        pos = positions[node]
        panos = get_streetview(pos, cwd=cwd)
        sat_path = download_sat_point(point=pos, cwd=cwd)
        sub_dict[node] = {'sat': sat_path, 'street': [pair[0] for pair in panos], 'heading': [pair[1] for pair in panos]}
    return sub_dict
