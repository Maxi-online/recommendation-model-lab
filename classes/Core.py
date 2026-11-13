"""Small mutable state container used by the serialized legacy HTTP adapter."""
class Core:
    def __init__(self):
        self.db = None
        self.req = None
        self.model = None
        self.config = None
