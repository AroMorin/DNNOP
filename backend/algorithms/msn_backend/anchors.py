"""Base class for anchors"""

class Anchors:
    def __init__(self, hp):
        self.nb_anchors = hp.nb_anchors
        self.anchors = ''
