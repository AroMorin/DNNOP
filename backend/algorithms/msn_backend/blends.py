"""Class that defines all blend operations."""

class Blends:
    def __init__(self, hp):
        self.nb_blends = hp.pool_size-((hp.nb_anchors*hp.nb_probes)+1)
        self.blends = ''
