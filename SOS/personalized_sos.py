import numpy as np
from mealpy import Optimizer

class Clustering_SOS(Optimizer):

    """
    The Personalized version for Data Clustering: Symbiotic Organisms Search (DC-SOS)
    """

    def __init__(self, epoch=1000, pop_size=100, **kwargs):
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])

        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])

        self.set_parameters(["epoch", "pop_size"])

        self.is_parallelizable = False

        self.sort_flag = False


    def initialize_variables(self):
        """
        This is method is called before initialization() method.
        """
        ## Support variables
        self.space = self.problem.ub - self.problem.lb

    

    def initialization(self):
        """
        Override this method if needed. But the first 2 lines of code is required.
        """
        ### Required code
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        ### Your additional code can be implemented here
            
    
            
    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        This function is based on optimizer's strategy.
        In each optimizer, this function can be overridden

        Args:
            solution: The position

        Returns:
            The valid solution based on optimizer's strategy
        """
        return solution
        
    

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):

            ## Mutualism Phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            mutual_vector = (self.pop[idx].solution + self.pop[jdx].solution) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            xi_new = self.pop[idx].solution + self.generator.random() * (self.g_best.solution - bf1 * mutual_vector)
            xj_new = self.pop[jdx].solution + self.generator.random() * (self.g_best.solution - bf2 * mutual_vector)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
            if self.compare_target(xj_target, self.pop[jdx].target, self.problem.minmax):
                self.pop[jdx].update(solution=xj_new, target=xj_target)
            

            ## Commensalism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            xi_new = self.pop[idx].solution + self.generator.uniform(-1, 1) * (self.g_best.solution - self.pop[jdx].solution)
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)

            
            ## Parasitism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            temp_idx = self.generator.integers(0, self.problem.n_dims)
            xi_new = self.pop[jdx].solution.copy()
            xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)


