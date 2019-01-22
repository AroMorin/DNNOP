"""base class for elite"""

class Elite:
    def __init__(self, hp):
        self.elite = None
        self.elite_score = hp.initial_score
        self.minimizing = hp.minimizing

    def set_elite(self, pool, scores):
        idx = get_top_score_idx(scores)
        top_score = scores[idx]
        if self.replace(top_score):
            print ("Setting new elite")
            elite = pool[idx]
            self.clone_elite(elite)
            self.elite_score = top_score
            return
        print ("Elite Score: %f" %self.elite_score)

    def get_top_score_idx(self, scores):
        if self.minimizing:
            return argmin(scores)
        else:
            return argmax(scores)

    def replace(self, top_score):
        if self.minimizing:
            return top_score < self.elite_score
        else:
            return top_score > self.elite_score

    def clone_elite(self, elite):
        """We clone the elite in order to have our own copy of it, not just a
        pointer to the object. This will be important because we want to
        keep the elite outside of the pool. Only when backtracking do we insert
        the elite into the pool.
        """
        self.elite = elite.clone()
