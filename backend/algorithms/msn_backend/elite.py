"""base class for elite"""

class Elite:
    def __init__(self, hp):
        self.elite = ''
        self.backtracking = False
        self.patience = hp.patience
