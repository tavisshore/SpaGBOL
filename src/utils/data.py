#!/usr/bin/env python3
from pathlib import Path
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from PIL import Image
import osmnx as ox
import networkx as nx
import torchvision.transforms as T
import torchvision.transforms.functional as F
from geographiclib.geodesic import Geodesic
import polarTransform
import cv2
from src.utils.guillame import ImageDatabase
from src.utils.write_db import write_database

import ray
from ray.experimental import tqdm_ray

from src.utils.satellite import download_junction_data

to_ten = T.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

        self.train_coords = [self.coords[point] for point in self.args.train_localisations]
        self.test_coords = [self.coords[point] for point in self.args.test_localisations]


        self.graphs, self.test_graphs = {}, {}
        self.train_walks, self.val_walks, self.test_walks = {}, {}, {} # {point: walks}

        for coord in self.train_coords: 
            self.prepare_graph(coord)
        for coord in self.test_coords: 
            self.prepare_graph(coord, stage='test')

    def prepare_graph(self, point=None, stage='train'):
        # Download graph or open if available
        if not Path(f'{self.args.data}/graphs/graph_{point}_{self.args.width}.pt').is_file():
            corpus_graph = self.create_graph(centre=point, dist=self.args.width, cwd=self.args.path, workers=self.args.workers)
            torch.save(corpus_graph, f'{self.args.path}/data/{self.args.dataset}/graphs/graph_{point}_{self.args.width}.pt')
            src_images = Path(f'{self.args.path}/data/{self.args.dataset}/images/raw/')
            dst_database = Path(f'{self.args.path}/data/{self.args.dataset}/images/lmdb/')
            image_paths = {image_path.stem: image_path for image_path in sorted(src_images.rglob(f"*jpg"))}
            write_database(image_paths, dst_database) # Write to LMDB
        else: 
            corpus_graph = torch.load(f'{self.args.data}/graphs/graph_{point}_{self.args.width}.pt')
            
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

    def create_graph(self, centre=(51.509865, -0.118092), dist=1000, cwd='/home/ts00987/data', workers=4):
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
    coord = item.split('_')[0].split('(')[1].split(')')[0]
    coord = tuple(map(float, coord.split(', ')))
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

            if self.args.num_povs: 
                pano_keys = pano_keys[:self.args.num_povs]
            indice = select_index(pano_keys, self.stage)
            pano_key = pano_keys[indice]

            pano_key = remove_extension(pano_key)
            pano = np.array(self.lmdb[pano_key])
            _, width, _ = pano.shape

            norths = walk.nodes[node]['north']
            north = norths[indice]

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
    # TEMPORARY IMPORT BODGE
    from yacs.config import CfgNode as CN


    """
    SET PATH VARIABLES
    """
    path = str(Path(__file__).parent.parent.absolute())

    _C = CN()

    _C.data = '/scratch/datasets/'
    _C.dataset = 'spagbol'
    _C.fov = 90

    _C.train_localisations =  ['tokyo', 'london', 'philly', 'brussels', 'chicago', 'new york', 'singapore', 'hong kong', 'guildford']
    _C.test_localisations =  ['boston']
    _C.width = 2000   # Width of graph
    _C.sat_width = 50 # Width of satellite image for each node 
    _C.roll_direction = True
    _C.yaw_threshold = 45 # 0-90
    _C.num_yaws = 8

    # I guess walk can't be more than k_hop - generally?
    _C.walk = 4
    _C.num_povs = 0 # 0 doesn't limit, then [1, 2, 3, 4]
    _C.exhaustive = False
    _C.workers = 1

    _C.path = path

    _C.loss = 'triplet'
    _C.encoder = 'sage'
    _C.enc_layers = 2
    _C.hidden_dim = 256
    _C.out_dim = 64

    cfg = _C.clone()
    cfg.merge_from_file('config/standard.yaml')
    cfg.freeze()

    # parser = args.to_parser()
    # args.merge(parser.parse_args())

    graphs = GraphData(cfg)
    dataset = GraphDataset(cfg, graphs, stage='train')
