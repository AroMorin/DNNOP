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
        self.integrity = self.hp.initial_integrity
        self.lr = self.hp.lr
        self.alpha = self.hp.alpha
        self.lambda_ = self.hp.lambda_
        self.search_start = True

    def analyze(self, scores, nb_anchors):
        self.clean_list(scores)
        print("Cleaned scores: ", self.scores)
        self.sort_scores()
        print("Sorted scores: ", self.sorted_scores)
        self.sort_idxs()
        self.set_integrity()
        self.review(nb_anchors)
        self.set_num_selections()
        self.set_search_radius()
        print("Integrity: %f" %self.integrity)

    def clean_list(self, mylist):
        # Check for type
        if int(mylist[0]) != mylist[0]:
            # Tensor
            mylist = [i.item() for i in mylist]
        elif int(mylist[0]) == mylist[0]:
            # Integer
            mylist = [i for i in mylist]
        else:
            print("Error in score variable type")
            exit()
        print("Raw scores: ", mylist)
        # Remove NaNs and infinities
        #self.scores = [x for x in mylist if not math.isnan(x)]
        self.scores = [x for x in mylist if not math.isnan(x) and not math.isinf(x)]

    def sort_scores(self):
        """This function sorts the values in the list. Duplicates are removed
        also.
        """
        self.current_top = self.new_top  # Inheritance
        if self.hp.minimizing:
            self.sorted_scores = sorted(set(self.scores))
        else:
            self.sorted_scores = sorted(set(self.scores), reverse=True)
        self.new_top = self.sorted_scores[0]
        print("Pool top score: %f" %self.new_top)

    def sort_idxs(self):
        """This function checks each element in the sorted list to retrieve
        all matching indices in the original scores list. This preserves
        duplicates. It reduces the likelihood that anchor slots will be
        unfilled.
        """
        self.sorted_idxs = []
        for i in range(len(self.sorted_scores)):
            score = self.sorted_scores[i]
            idxs = [idx for idx, value in enumerate(self.scores) if value == score]
            for idx in idxs:
                self.sorted_idxs.append(idx)
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
            print ("No improvement")
            if not self.search_start:
                # Reduce integrity, but not below the minimum allowed level
                a = self.integrity-self.hp.step_size
                b = self.hp.min_integrity
                self.integrity = max(a, b)
                self.elapsed_steps += 1
            else:
                print("Start searching")
                self.integrity = self.hp.def_integrity  # Reset integrity
                self.search_start = False
        else:  # Improved
            print ("Improved")
            self.elapsed_steps = 0
            self.search_start = True

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

    def review(self, nb_anchors):
        self.set_backtracking()
        self.set_radial_expansion(nb_anchors)


    def set_backtracking(self):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter."""
        if self.elapsed_steps > self.hp.patience:
            print ("Waited %d steps" %self.elapsed_steps)
            self.backtracking = True
            self.elapsed_steps = 0
            self.integrity = self.hp.def_integrity  # Reset integrity
        else:
            self.backtracking = False

    def set_radial_expansion(self, nb_anchors):
        """Triggers radial expansion only if the condition is met. Then in the
        new turn it switches it off again.
        It compares the actual number of anchors with the desired number of
        anchors
        """
        if nb_anchors < self.hp.nb_anchors:
            print("--Expanding Search Radius!--")
            self.radial_expansion = True
            self.lr = self.lr * self.hp.expansion_factor
            self.alpha = self.alpha * self.hp.expansion_factor
            self.lambda_ = self.lambda_ * self.hp.expansion_factor
        else:
            self.radial_expansion = False
            self.lr = self.hp.lr
            self.alpha = self.hp.alpha
            self.lambda_ = self.hp.lambda_

    def set_num_selections(self):
        p = 1-self.integrity
        numerator = self.hp.alpha
        denominator = 1+(self.hp.beta/p)
        self.num_selections = numerator/denominator

    def set_search_radius(self):
        p = 1-self.integrity
        argument = (self.hp.lambda_*p)-2.5
        exp1 = math.tanh(argument)+1
        self.search_radius = exp1*self.hp.lr
        print ("Search Radius: %f" %self.search_radius)














#
