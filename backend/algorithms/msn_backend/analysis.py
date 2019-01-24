"""Class for analysis operations on the scores."""

from __future__ import division
import math

class Analysis:
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.current_top = self.hp.initial_score
        self.new_top = self.hp.initial_score
        self.top_idx = 0
        self.scores = []
        self.sorted_scores = []
        self.sorted_idxs = []
        self.backtracking = False
        self.elapsed_steps = 0  # Counts steps without improvement
        self.reset_integrity = False

    def analyze(self, scores):
        self.clean_list(scores)
        self.sort_scores()
        self.sort_idxs()
        self.set_integrity()
        self.review()

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

    def sort_idxs(self):
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
        """Once an improvement is detected, the flag "reset_integrity" is set
        to True. This means that if later there wasn't an improvement, integrity
        would be reset. Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        if not self.improved():
            if not self.search_start:
                # Reduce integrity, but not below the minimum allowed level
                self.integrity = max(self.hp.step_size, self.hp.min_integrity)
                self.elapsed_steps += 1
            else:
                self.integrity = self.hp.def_integrity  # Reset integrity
                self.search_start = False
        else:  # Improved
            self.search_start = True
            self.elapsed_steps = 0

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
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        The integrity needs to be reset at some point, however. Maybe backtracking
        is just enough for now? I want to reset integrity once no improvement
        was detected (but only the first instance of such occasion).
        """
        if self.current_top != 0:
            # Percentage change
            self.entropy = ((self.new_top-self.current_top)/abs(self.current_top))*100
        else:
            # Prevent division by zero
            self.entropy = ((self.new_top-self.current_top)/abs(self.hp.epsilon))*100
        print("Entropy: %f" %self.entropy)

    def review(self):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter.
        """
        self.backtracking = False
        if self.elapsed_steps > self.hp.patience:
            self.backtracking = True
            self.elapsed_steps = 0















#
