import pathlib as p
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import utils

from frag_nn import constants as c


class XChemDataset(Dataset):

    def __init__(self, XChemDataFrame, mode="ED", grid_size=10, grid_step=0.5, transform=None, replace_rate=0.5,
                 network_type="classifier"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.XChemDataFrame = XChemDataFrame
        self.transform = transform
        self.grid_size = grid_size
        self.grid_step = grid_step
        self.mode = mode
        self.replace_rate = replace_rate
        self.network_type = network_type

    def __len__(self):
        return len(self.XChemDataFrame)

    def __getitem__(self, idx):

        # idx -> record
        record = self.XChemDataFrame.iloc[idx]

        if self.mode == "ED":
            sample = SampleED(record, self.grid_size, self.grid_step)
        elif self.mode == "RefMove":
            sample = SampleRefMove(record, self.grid_size, self.grid_step, replace_rate=self.replace_rate)
        elif self.mode == "RefMovev2":
            sample = SampleRefMovev2(record, self.grid_size, self.grid_step, replace_rate=self.replace_rate,
                                     network_type=self.network_type)

        # Get annotation
        xy = sample.get()



        # return sample.xy()
        #
        # xmap, annotation = get_ccp4_label(record,
        #                                   grid_size=self.grid_size,
        #                                   grid_step=self.grid_step)
        #
        # if self.transform:
        #     sample = self.transform(sample)

        return xy


class SampleED:

    def __init__(self, record, grid_size, grid_step):
        self.record = record
        self.x = None
        self.y = None
        self.grid_size = grid_size
        self.grid_step = grid_step

    def get_data(self):
        # Get Model path
        pdb_path = self.record[c.model_pdb_record_name]

        # Load the model
        model = utils.load_model(pdb_path)

        # Find the ligand
        ligand_model = utils.get_ligand_model(model)

        # # Find the ligand centroid
        # ligand_centroid_orth = utils.get_ligand_centroid(ligand_model)
        # print(ligand_centroid_orth)

        # event centroid
        event_centroid = np.array([self.record.event_centroid_x,
                                   self.record.event_centroid_y,
                                   self.record.event_centroid_z
                                   ])
        offset = np.random.randn(3) * 4 - 2
        event_centroid_perturbed = event_centroid + offset
        # print("Event centroid is: {}".format(event_centroid))

        # Get map path
        event_map_path = record[c.event_map_record_name]

        # Load the xmap
        # print("Loading ccp4 map")
        xmap = utils.load_xmap_from_ccp4(event_map_path)

        # Cut around the ligand
        xmap_cut = utils.subsample_xmap(xmap=xmap,
                                        ligand_centroid_orth=event_centroid_perturbed,
                                        grid_size=self.grid_size,
                                        grid_step=self.grid_step)

        self.x = xmap_cut

    def get_annotation(self):

        label = utils.get_label_is_hit(self.record)
        self.y = np.array(label, dtype=np.float32)

    def get_xy(self):
        return {'x': self.x,
                'y': self.y}


class SampleRefMove:

    def __init__(self, record, grid_size, grid_step, replace_rate=0.5):
        self.record = record
        self.x = None
        self.y = None
        self.grid_size = grid_size
        self.grid_step = grid_step
        self.replace_rate = 0.5
        self.replace = False



        # print("Replacing move with ref: {}".format(self.replace))

    def get(self):

        # Get annotation
        self.get_annotation()

        # Decide whether to replace: if it is a hit replace with 50% otherwise don't
        if self.y == 1:
            if np.random.rand() < self.replace_rate:
                self.replace = True
                self.y = np.array(0, dtype=np.float32)
            else:
                self.replace = False
        else:
            self.replace = False

        # Get data
        try:
            self.get_data()
        except Exception as e:
            self.x = np.zeros([2, self.grid_size, self.grid_size, self.grid_size], dtype=np.float32)
            self.y = np.array(0, dtype=np.float32)


        return self.get_xy()



    def get_data(self):

        # Get event centroid
        event_centroid = np.array([self.record.event_centroid_x,
                                   self.record.event_centroid_y,
                                   self.record.event_centroid_z
                                   ])

        # Decide random Translation
        base_offset = (np.random.randn(3) - 0.5) * 4

        # Decide random rotation
        base_grid_rotation = np.random.rand(3) * 2 * np.pi

        # Get refmap path
        map_dir = p.Path(self.record[c.event_map_record_name]).parent
        system_name = map_dir.name
        ref_map_name = system_name + "-ground-state-mean-map.native.ccp4"
        if (map_dir / ref_map_name).exists():
            ref_map_path = str((map_dir / ref_map_name))
        # elif (map_dir / "2fofc.map").exists():
        #     ref_map_path = str((map_dir / "2fofc.map"))
        # else:
        #
        #     ref_map_path = self.record[c.event_map_record_name]
        else:
            raise Exception("Didn't work!: {}".format(map_dir))


        # Load the refmap
        xmap_ref = utils.load_xmap_from_ccp4(ref_map_path)

        # Cut around the event
        event_centroid_perturbed_ref = event_centroid + base_offset
        grid_rotation_ref = base_grid_rotation
        xmap_ref_cut = utils.subsample_xmap(xmap=xmap_ref,
                                            grid_rotation=grid_rotation_ref,
                                            ligand_centroid_orth=event_centroid_perturbed_ref,
                                            grid_size=self.grid_size,
                                            grid_step=self.grid_step)

        # Get moving map path
        if self.replace is False:
            event_map_path = self.record[c.event_map_record_name]
        else:
            event_map_path = ref_map_path



        # Slightly adjust translation
        event_centroid_perturbed_move = event_centroid_perturbed_ref + (np.random.randn(3) - 0.5)

        # Slightly adjust rotation
        grid_rotation_move = grid_rotation_ref + (np.random.rand(3) - 0.5) * 2 * np.pi * 0.05


        # Load the moving map
        xmap_moving = utils.load_xmap_from_ccp4(event_map_path)

        # Cut around the event
        xmap_moving_cut = utils.subsample_xmap(xmap=xmap_moving,
                                               grid_rotation=grid_rotation_move,
                                               ligand_centroid_orth=event_centroid_perturbed_move,
                                               grid_size=self.grid_size,
                                               grid_step=self.grid_step)

        # Concatenate the maps
        xmaps_cut = np.stack([xmap_ref_cut, xmap_moving_cut], axis=0)
        # print(xmaps_cut.shape)

        self.x = xmaps_cut

    def get_annotation(self):

        # if self.replace is False:
        label = utils.get_label_is_hit(self.record)
        # else:
        #     label = np.array(0, dtype=np.float32)

        self.y = np.array(label, dtype=np.float32)

    def get_xy(self):
        return {'x': self.x,
                'y': self.y}


def get_ccp4_label(record, grid_size=10, grid_step=0.5):

    # print(record)
    # print("Model path is: {}".format(record.pandda_input_pdb))

    # print("Event map is: {}".format(record.pandda_event_map_native))


    # Get Model path
    pdb_path = record[c.model_pdb_record_name]

    # Load the model
    model = utils.load_model(pdb_path)

    # Find the ligand
    ligand_model = utils.get_ligand_model(model)

    # # Find the ligand centroid
    # ligand_centroid_orth = utils.get_ligand_centroid(ligand_model)
    # print(ligand_centroid_orth)

    # event centroid
    event_centroid = np.array([record.event_centroid_x,
                               record.event_centroid_y,
                               record.event_centroid_z
                               ])
    offset = np.random.randn(3)*4 - 2
    event_centroid_perturbed = event_centroid + offset
    # print("Event centroid is: {}".format(event_centroid))

    # Get map path
    event_map_path = record[c.event_map_record_name]

    # Load the xmap
    # print("Loading ccp4 map")
    xmap = utils.load_xmap_from_ccp4(event_map_path)

    # Cut around the ligand
    xmap_cut = utils.subsample_xmap(xmap=xmap,
                                    ligand_centroid_orth=event_centroid_perturbed,
                                    grid_size=grid_size,
                                    grid_step=grid_step)

    # # Find the sample label from the database
    label = utils.get_label_is_hit(record)

    # # Convert xmap to tensor
    # x = torch.Tensor(xmap_cut)
    #
    # # Convert hit value to tensor
    # y = torch.Tensor([label])

    #
    x = xmap_cut

    # Convert hit value to tensor
    y = np.array(label, dtype=np.float32)

    return x, y


def get_ccp4_label_v2(record, grid_size=10, grid_step=0.5):

    # print(record)
    # print("Model path is: {}".format(record.pandda_input_pdb))

    # print("Event map is: {}".format(record.pandda_event_map_native))


    # Get Model path
    pdb_path = record[c.model_pdb_record_name]

    # Load the model
    model = utils.load_model(pdb_path)

    # Find the ligand
    ligand_model = utils.get_ligand_model(model)

    # # Find the ligand centroid
    # ligand_centroid_orth = utils.get_ligand_centroid(ligand_model)
    # print(ligand_centroid_orth)

    # event centroid
    event_centroid = np.array([record.event_centroid_x,
                               record.event_centroid_y,
                               record.event_centroid_z
                               ])
    offset = np.random.randn(3)*4 - 2
    event_centroid_perturbed = event_centroid + offset
    # print("Event centroid is: {}".format(event_centroid))

    # Get map path
    event_map_path = record[c.event_map_record_name]

    # Load the xmap
    # print("Loading ccp4 map")
    xmap = utils.load_xmap_from_ccp4(event_map_path)

    # Cut around the ligand
    xmap_cut = utils.subsample_xmap(xmap=xmap,
                                    ligand_centroid_orth=event_centroid_perturbed,
                                    grid_size=grid_size,
                                    grid_step=grid_step)

    # # Find the sample label from the database
    label = utils.get_label_is_hit_v2(record)

    # # Convert xmap to tensor
    # x = torch.Tensor(xmap_cut)
    #
    # # Convert hit value to tensor
    # y = torch.Tensor([label])

    #
    x = xmap_cut

    # Convert hit value to tensor
    y = np.array(label, dtype=np.float32)

    return x, y


class SampleRefMovev2:

    def __init__(self, record, grid_size, grid_step, replace_rate=0.5, network_type="classifier"):
        self.record = record
        self.x = None
        self.y = None
        self.grid_size = grid_size
        self.grid_step = grid_step
        self.replace_rate = 0.5
        self.replace = False

        self.event_map_path = None
        self.ref_map_path = None
        self.dtag = self.record.dtag
        self.event_idx = self.record.event_idx
        self.type = network_type

    def get(self):

        # Get annotation
        self.get_annotation()

        # Decide whether to replace: if it is a hit replace with 50% otherwise don't
        if self.type == "classifier":
            if self.y[1] == 1:
                if np.random.rand() < self.replace_rate:
                    self.replace = True
                    self.y = np.array([1, 0], dtype=np.float32)

        else:
            if self.y == 1:
                if np.random.rand() < self.replace_rate:
                    self.replace = True
                    self.y = np.array(0, dtype=np.float32)


        # Get data
        try:
            self.get_data()
        except Exception as e:
            print(e)
            print("Getting data failed")
            self.x = np.zeros([2, self.grid_size, self.grid_size, self.grid_size], dtype=np.float32)
            if self.type == "classifier":
                self.y = np.array([1, 0], dtype=np.float32)
            else:
                self.y = np.array(0, dtype=np.float32)


        return self.get_xy()



    def get_data(self):

        # Get event centroid
        # print(self.record)
        event_centroid = np.array([self.record.x,
                                   self.record.y,
                                   self.record.z
                                   ])

        # Decide random Translation
        base_offset = (np.random.randn(3) - 0.5) * 4

        # Decide random rotation
        base_grid_rotation = np.random.rand(3) * 2 * np.pi

        # Get refmap path
        ref_map_path = self.record.ground_map_path
        self.ref_map_path = ref_map_path


        # Load the refmap
        # print("Loading refmap from: {}".format(ref_map_path))
        xmap_ref = utils.load_xmap_from_ccp4(ref_map_path)

        # Cut around the event
        event_centroid_perturbed_ref = event_centroid + base_offset
        grid_rotation_ref = base_grid_rotation
        xmap_ref_cut = utils.subsample_xmap(xmap=xmap_ref,
                                            grid_rotation=grid_rotation_ref,
                                            ligand_centroid_orth=event_centroid_perturbed_ref,
                                            grid_size=self.grid_size,
                                            grid_step=self.grid_step)

        # Get moving map path
        if self.replace is False:
            event_map_path = self.record.event_map_path
        else:
            event_map_path = ref_map_path

        self.event_map_path = event_map_path



        # Slightly adjust translation
        event_centroid_perturbed_move = event_centroid_perturbed_ref + 0.5*(np.random.randn(3) - 0.5)

        # Slightly adjust rotation
        grid_rotation_move = grid_rotation_ref + (np.random.rand(3) - 0.5) * 2 * np.pi * 0.025


        # Load the moving map
        # print("Loading refmap from: {}".format(event_map_path))
        xmap_moving = utils.load_xmap_from_ccp4(event_map_path)

        # Cut around the event
        xmap_moving_cut = utils.subsample_xmap(xmap=xmap_moving,
                                               grid_rotation=grid_rotation_move,
                                               ligand_centroid_orth=event_centroid_perturbed_move,
                                               grid_size=self.grid_size,
                                               grid_step=self.grid_step)

        # Concatenate the maps
        xmaps_cut = np.stack([xmap_ref_cut, xmap_moving_cut], axis=0)
        # print(xmaps_cut.shape)

        self.x = xmaps_cut

    def get_annotation(self):

        # if self.replace is False:
        label = utils.get_label_is_hit_v2(self.record)
        # else:
        #     label = np.array(0, dtype=np.float32)

        if self.type == "classifier":
            if label == 0:
                self.y = np.array([1, 0], dtype=np.float32)
            elif label == 1:
                self.y = np.array([0, 1], dtype=np.float32)
            else:
                raise Exception
        else:
            self.y = np.array(label, dtype=np.float32)

    def get_xy(self):
        return {'x': self.x,
                'y': self.y,
                "event_map_path": self.event_map_path,
                "ref_map_path": self.ref_map_path,
                "dtag": self.dtag,
                "event_idx": self.event_idx,
                "replaced": self.replace
                }


# class EDDataset:
#
#     def __init__(self, event_table, grid_size = 10, grid_step=0.5):
#         self.table = event_table
#         self.grid_size = grid_size
#         self.grid_step = grid_step
#
#     def __len__(self):
#         return len(self.table)
#
#     def __getitem__(self, idx):
#
#         # Get item from table
#         record = self.table.iloc[idx]
#
#         # Get pdb and mtz paths
#         pdb_path = record[c.model_pdb_record_name]
#         mtz_path = record[c.input_mtz_record_name]
#
#         # Load the model
#         model = utils.load_model(pdb_path)
#
#         # Find the ligand
#         ligand_model = utils.get_ligand_model(model)
#
#         # Find the ligand centroid
#         ligand_centroid = utils.get_ligand_centroid(ligand_model)
#
#         # Load the xmap
#         xmap, grid_model = utils.load_xmap(mtz_path)
#
#         # Translate the ligand centroid to grid coords
#         ligand_centroid_grid = utils.cart_to_grid(ligand_centroid,
#                                                      xmap)
#
#         # Cut around the ligand
#         xmap_trans = utils.subsample_xmap(xmap=xmap,
#                                              ligand_centroid_grid=ligand_centroid_grid,
#                                              grid_size=self.grid_size,
#                                              grid_step=self.grid_step)
#
#         # Convert to torch tensor
#         xmap_torch = torch.Tensor(xmap_trans)
#
#         # Find the sample label from the database
#         label = utils.get_label_is_hit(record)
#
#         # Define the sample to be returned
#         sample = {"density": xmap_torch,
#                   "model": model,
#                   "label": label}
#
#         return sample


class EventDataset(Dataset):

    def __init__(self, events=None, transforms_record=[], get_annotation=None, get_data=None, root=""):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.XChemDataFrame = events
        self.transforms_record = transforms_record
        self.get_annotation = get_annotation
        self.get_data = get_data
        self.root = ""

    def __len__(self):
        return len(self.XChemDataFrame)

    def __getitem__(self, idx):

        # idx -> record
        record = Record(self.XChemDataFrame.iloc[idx])

        for transform in self.transforms_record:
            record = transform(record)

        annotation = self.get_annotation(record)
        data = self.get_data(record)

        return {"data": data,
                "annotation": annotation}


class Record:
    def __init__(self, record):

        self.record = record

        self.ground_map_path = lambda: self.record.ground_map_path
        self.z_map_path = lambda: self.record.z_map_path
        self.event_map_path = lambda: self.record.event_map_path

        self.location = lambda: np.array([self.record.x, self.record.y, self.record.z])
        self.rotation = lambda: np.array([0,0,0])

        self.annotation = self.record["Ligand Confidence"]


class SetRoot:
    def __init__(self, root=""):
        self.root = root

    def __call__(self, record):
        ground_map_path = record.ground_map_path()
        record.ground_map_path = lambda: self.root + ground_map_path

        z_map_path = record.z_map_path()
        record.z_map_path = lambda: self.root + z_map_path

        event_map_path = record.event_map_path()
        record.event_map_path = lambda: self.root + event_map_path
        return record


class GetAnnotationRandomlyReplaced:

    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, *args, **kwargs):
        pass


class GetRandomisedLocation:
    def __init__(self, base_trans_max=4.0, secondary_trans_max=1.0):
        self.base_trans_max = base_trans_max
        self.secondary_trans_max = secondary_trans_max

        self.base_offset = 0

    def __call__(self, record):
        self.base_offset = (np.random.randn(3) - 0.5) * self.base_trans_max
        initial_location = record.location()

        record.location = lambda: initial_location + self.base_offset + (np.random.randn(3) - 0.5) * self.secondary_trans_max
        return record


class GetRandomisedRotation:
    def __init__(self, max_rot=0.025):
        self.max_rot = max_rot
        self.base_rotation = 0

    def __call__(self, record):
        self.base_rotation = np.random.rand(3) * 2 * np.pi
        record.rotation = lambda: self.base_rotation + (np.random.rand(3) - 0.5) * 2 * np.pi * self.max_rot
        return record


class Sample:
    def __init__(self, record, get_annotation, get_data):
        self.record = record

        self.get_annotation = get_annotation
        self.get_data = get_data

        self.data = None
        self.annotation = None

    def get(self):

        # Get annotation
        self.annotation = self.get_annotation(self.record)

        # Get data
        self.data = self.get_data(self.record, self.annotation)

        return {"data": self.data,
                "annotation": self.annotation}


# GET DATA
class GetDataRefMove:

    def __init__(self, grid):

        self.grid = grid

    def __call__(self, record):

        ref_map = self.get_map(record.ground_map_path(),
                               self.grid,
                               record.location(),
                               record.rotation())

        event_map = self.get_map(record.event_map_path(),
                                 self.grid,
                                 record.location(),
                                 record.rotation())

        return np.stack([ref_map, event_map ],
                        axis=0)

    def get_map(self, map_path, grid, centre, rotation):
        xmap_moving = utils.load_xmap_from_ccp4(map_path)

        # Cut around the event
        xmap_moving_cut = utils.subsample_xmap(xmap=xmap_moving,
                                               grid_rotation=rotation,
                                               ligand_centroid_orth=centre,
                                               grid_size=grid.size,
                                               grid_step=grid.step)
        return xmap_moving_cut


class GetDataRefMoveZ:

    def __init__(self, grid):

        self.grid = grid

    def __call__(self, record):

        ref_map = self.get_map(record.ground_map_path(),
                               self.grid,
                               record.location(),
                               record.rotation())

        z_map = self.get_map(record.z_map_path(),
                               self.grid,
                               record.location(),
                               record.rotation())

        event_map = self.get_map(record.event_map_path(),
                                 self.grid,
                                 record.location(),
                                 record.rotation())

        return np.stack([ref_map, z_map, event_map ],
                        axis=0)

    def get_map(self, map_path, grid, centre, rotation):
        xmap_moving = utils.load_xmap_from_ccp4(map_path)

        # Cut around the event
        xmap_moving_cut = utils.subsample_xmap(xmap=xmap_moving,
                                               grid_rotation=rotation,
                                               ligand_centroid_orth=centre,
                                               grid_size=grid.size,
                                               grid_step=grid.step)
        return xmap_moving_cut


# GRID
class OrthogonalGrid:
    def __init__(self, grid_size, grid_step):
        self.size = grid_size
        self.step = grid_step


# GET ANNOTATION
class GetAnnotationClassifier:

    def __init__(self):

        pass

    def __call__(self, record):

        if record.annotation in ["High", "Medium"]:
            label = 1.0
        else:
            label = 0.0

        if label == 0:
            y = np.array([1, 0], dtype=np.float32)
        elif label == 1:
            y = np.array([0, 1], dtype=np.float32)
        else:
            raise Exception

        return y

# class GetMeta:
#
#     def __init__(self):
#         pass
#
#     def __call__(self, record):
#         meta = {}
#
#         meta["ground_map_path"] =