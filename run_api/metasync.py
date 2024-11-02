from run_api.database import Base
from metasync.shared import create_metagraph_node_class

MetagraphNode = create_metagraph_node_class(Base)
