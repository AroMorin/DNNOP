"""A Solver class for robot problems.
"""

from .solver import Solver

class Robo_Solver(Solver):
    def __init__(self, slv_params):
        super(Robo_Solver, self).__init__(slv_params)
        self.current_iteration = 0

    def solve(self, iterations):
        """A solver method."""
        print("Training Robot solver \n")
        for _ in range(iterations):
            print("Iteration: %d\n" %self.current_iteration)
            self.env.step()
            self.forward()
            self.backward()
            self.log(score=None, iteration=self.current_iteration)
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def demonstrate_env(self):
        """In cases where training is needed."""
        self.alg.eval()
        for _ in range(episodes):
            self.roll_and_render()
        self.env.close()

    def log(self, score=None, iteration=0):
        if self.logger == None:
            return
        else:
            if score is None:
                score = self.evaluator.score.item()
            if not isinstance(score, float):
                score = score.item()
            top_score = self.alg.top_score.item()
            self.logger.log_metric("Score", score, step=iteration)
            self.logger.log_metric("Top Score", top_score, step=iteration)


#
