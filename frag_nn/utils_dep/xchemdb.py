# Imports
import pandas as pd
import sqlalchemy

class XChemDataset:
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

    def connect(self):
        engine = sqlalchemy.create_engine(
            "postgresql://{}:{}@{}:{}/{}".format(self.user, self.password, self.host, self.port, self.database))

        return engine

    def get_event_data_paths(self):
        event_table = self.get_database("pandda_event")

        event_records = event_table.to_dict('records')

        valid_events = []
        for record in event_records:
            try:
                pth = p.Path(record[c.input_pdb_record_name])
                if pth.exists():
                    valid_events.append(record)
                else:
                    continue
            except Exception as e:
                print(e)

        return valid_events

    def get_database(self, db_name):

        db = pd.read_sql_query("SELECT * FROM {}".format(db_name), con=self.con)

        self.databases[db_name] = db

        return db

    def get_all_databases(self):

        for db in self.tables:
            self.get_database(db)

        return self.databases










