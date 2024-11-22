from yacs.config import CfgNode as CN
from pathlib import Path

path = str(Path(__file__).parent.parent.absolute())

_C = CN()
# defaults 
_C.data = '/scratch/datasets/'
_C.dataroot = 'data/images/'
_C.savedir = 'data'


_C.lr = 1e-4
_C.dataset = 'spagbol'
_C.batch_size = 16
_C.weight_decay = 1e-5
_C.epochs = 50
_C.checkpoint_dir = 'weights/checkpoints/'

# self.coords = {'guildford': (51.246194, -0.574250), 'london': (51.51492, -0.09057), 'brussels': (50.853481, 4.355358), 
#                 'boston': (42.358191, -71.060911), 'tokyo': (35.680886, 139.777483), 'chicago': (41.883181, -87.629645),
#                 'new york': (40.7484, -73.9857), 'philly': (39.952364, -75.163616), 'singapore': (1.280999, 103.845047)   

_C.train_localisations =  ['tokyo', 'london', 'philly', 'brussels', 'chicago', 'new york', 'singapore', 'hong kong', 'guildford']
_C.test_localisations =  ['boston']
_C.train_localisation_vigor = ['newyork', 'sanfrancisco', 'seattle']
_C.test_localisation_vigor = ['chicago']

_C.width = 2000   # Width of graph
_C.sat_width = 50 # Width of satellite image for each node 

_C.embedding = 'convnext' # bevcv, resnet, vgg, yaw, convnext
_C.roll_direction = False

_C.yaw_threshold = 45 # 0-90
_C.num_yaws = 8

_C.crop_yaws = False
_C.val_on_test = False

# YAW METRICS CONFIGS
_C.yaw_type = None # None, stat, pred
_C.yaw_oris_bears = True # known ori and bear
_C.yaw_oris = True # known ori, unknown bear

# I guess walk can't be more than k_hop - generally?
_C.walk = 4
_C.limit_povs = 0 # 0 doesn't limit, then [1, 2, 3, 4]
_C.exhaustive = False

_C.layer_equal_walk = False
_C.workers = 6
_C.single_pov = False

_C.config = 'misc/standard'
_C.resume_training = False
_C.path = path
_C.fov = 90

_C.loss = 'triplet'
_C.encoder = 'sage'
_C.enc_layers = 2
_C.hidden_dim = 256
_C.out_dim = 64

_C.neg_distance = 10
_C.name = 'test'
_C.acc_interval = 1

# evaluation
_C.single_node = True # That node embedding is the start_point only

### Cofiguration about bev-cv pretrained or not
_C.feat_ext_pretrained = False
_C.bev_trained = True
_C.gnn = True
_C.unfreeze_epoch = 0

_C.debug = False
_C.tune = True

_C.triplet_mine = False
_C.double_negatives = False
_C.resume_training = False # CONDORHTC

def return_defaults():
    return _C.clone()

