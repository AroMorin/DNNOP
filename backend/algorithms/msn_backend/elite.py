"""base class for elite"""

class Elite:
    def __init__(self, hp):
        self.model = None
        self.elite_score = hp.initial_score
        self.minimizing = hp.minimizing

    def set_elite(self, pool, analyzer):
        idx = analyzer.top_idx
        top_score = analyzer.new_top
        if self.replace(top_score):
            print ("------Setting new Elite-------")
            elite = pool[idx]
            self.clone_model(elite)
            self.elite_score = top_score
        print ("Elite Score: %f" %self.elite_score)

    def replace(self, top_score):
        if self.minimizing:
            return top_score < self.elite_score
        else:
            return top_score > self.elite_score

    def clone_model(self, elite):
        """We clone the elite in order to have our own copy of it, not just a
        pointer to the object. This will be important because we want to
        keep the elite outside of the pool. Only when backtracking do we insert
        the elite into the pool.
        """
        self.model = elite.clone()

    def query_elite(self, observation):
        return self.model(observation)
