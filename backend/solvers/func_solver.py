"""Base Class for a Solver. This class contains the different methods that
"""

from .solver import Solver

class Func_Solver(Solver):
    """This class makes absolute sense because there are many types of training
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        super(Func_Solver, self).__init__(slv_params)
        self.current_iteration = 0

    def solve(self, iterations):
        """In cases where training is needed."""
        print("Training regular solver \n")
        for _ in range(iterations):
            print("Iteration: %d\n" %self.current_iteration)
            self.env.step()
            self.forward()
            self.backward()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def solve_and_plot(self, iterations):
        """In cases where training is needed."""
        print("Training regular solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            self.env.step()
            self.forward()
            self.backward()
            self.env.make_plot(self.alg)
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

#
