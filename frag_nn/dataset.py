# Imports
import pandas as pd
import sqlalchemy

from torch.utils.data import Dataset

import clipper_python as clipper

# Class
class XChemDataset(Dataset):
    """
    Dataset representing the XChem database

    Items are dicts of:
        PanDDA event map
        Annotation
        Event coordinates
        Unit cell parameters
        Resolution

    Transformations include:


    """

    def __init__(self, host, port, database, user, password,
                 get_all=False):
        # Connect to XChem database

        # Database
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

        self.databases = {}

        self.con = self.connect()

        self.tables = sqlalchemy.inspect(self.con).get_table_names()

        if get_all is True: self.get_all_databases()





    pass

    def connect(self):
        engine = sqlalchemy.create_engine(
            "postgresql://{}:{}@{}:{}/{}".format(self.user, self.password, self.host, self.port, self.database))

        return engine

    def __len__(self):
        """
        Link to the XChem database
        Pull a local copy of the tables


        :return:
        """
        pass

    def __getitem__(self, item):
        """
        Load the member of the dataset keyed by item

        :param item:
        :return:
        """

        mtz_path = None
        pbd_path = None



        # Get Model


        # Get target
        target_mask = None

        # Get dict of returns


        return




    def transform_rotate(self, image):
        # Define random angles

        # Rotate
        image = scipy.ndimage.rotate(image,
                                     random[0],
                                     (0,1),
                                     reshape=False)
        image = scipy.ndimage.rotate(image,
                                     random[1],
                                     (1, 2),
                                     reshape=False)
        image = scipy.ndimage.rotate(image,
                                     random[2],
                                     (0, 2),
                                     reshape=False)

        return image

    def transform_crop(self, image):

        # Define crop parameters

        image = skimage.util.crop(image,
                                  (40, 40, 40))

        return image

    def transform_fft(self):
        mtz = clipper.CCP4MTZfile()
        xmap = clipper.Xmap(spacegroup,
                           cell,
                           grid)
        map.fft_from(mtz)

        return xmap


    def load_mtz(self, mtz_path):
        mtz = clipper.CCP4MTZfile(mtz_path)
        hkl_info = clipper.HKL_info()
        hkl_data = clipper.HKL_data_F_phi(hkl_info)
        mtz.import_hkl_info(hkl_info)
        mtz.import_hkl_data(hkl_data, "*/*/[FWT,PHWT]")
        mtz.close_read()

        return {"info": hkl_info, "data": hkl_data}


    def load_mmol(self, pdb_path):
        f = clipper.MMDBfile()
        f.read_file(pdb_path)
        mmol = clipper.MiniMol()
        f.import_minimol(mmol)
        f.close_read()

        return mmol

    def mask_protien(self, model, xmap):
        protien_atom_grid_coords = None
        protien_layer = np.zeros(xmap.shape)
        protien_layer[protien_atom_grid_coords] = 1

    def mask_lig(self):
        pass




