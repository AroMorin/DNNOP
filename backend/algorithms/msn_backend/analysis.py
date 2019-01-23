"""Class for analysis operations on the scores."""

class Analysis:
    def __init__(self, scores, hyper_params):
        self.scores = scores
        self.hp = hyper_params
        self.current_top = self.hyper_params.initial_score
        self.new_top = self.hyper_params.initial_score
        self.top_idx = 0
        self.sorted_scores = []
        self.sorted_idxs = []
        self.backtracking = False

    def analyze(self, scores):
        self.clean_list(scores)
        self.review()
        self.sort_scores()
        self.set_sorted_idxs()
        self.set_integrity()

    def clean_list(self, mylist):
        # Remove NaNs
        self.scores = [x for x in mylist if not math.isnan(x)]

    def sort_scores(self):
        """This function sorts the values in the list. Duplicates are removed
        also.
        """
        if self.hp.minimizing:
            self.sorted_scores = sorted(set(scores))
        else:
            self.sorted_scores = sorted(set(scores), reverse=True)
        self.new_top = self.sorted_scores[0]
        print("New top score: %f" %self.new_top)

    def get_sorted_idxs(self):
        """This function checks each element in the sorted list to retrieve
        all matching indices in the original scores list. This preserves
        duplicates. It reduces the likelihood that anchor slots will be
        unfilled.
        """
        self.sorted_idxs = []
        for i in range(len(self.scores)):
            score = self.sorted_scores[i]
            idxs = [idx for idx, value in enumerate(self.scores) if value == score]
            self.sorted_idxs.append(idxs)
        # Sanity checks
        assert len(self.sorted_idxs) == len(self.scores)  # No missing elements
        assert len(set(self.sorted_idxs)) == len(self.sorted_idxs)  # No duplicates
        self.top_idx = self.sorted_idxs[0]

    def set_integrity(self):
        if not self.improved():
            # Reduce integrity, but not below the minimum allowed level
            self.integrity = max(self.hp.step_size, self.hp.min_integrity)


    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.current_top != self.hp.initial_score:
            self.set_entropy()
            return self.entropy < self.hp.min_entropy
        else:
            # Improved over the initial score
            return True

    def set_entropy(self):
        if self.current_top != 0:
            self.entropy = ((self.new_top - self.current_top)./abs(self.current_top))*100
        else:
            self.entropy = 0
        print("Entropy: %f" %self.entropy)

    def review(self):
        if self.elapsed_steps > self.hp.patience:
            self.backtracking = True
        else:
            self.elapsed_steps += 1
















#
