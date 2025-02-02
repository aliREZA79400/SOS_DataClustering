from mealpy import Optimizer
import random


class mySOS(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False
        self.maxit = self.validator.check_int("epoch", epoch, [1, 100000])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # buil a shuffle list of pop indexes
        shuffle_pop = list(range(0, self.pop_size))
        random.shuffle(shuffle_pop)

        operators = ("mutual", "parasite")
        counts = [1, 2.5]
        for i in range(0, self.pop_size, 2):
            pair = shuffle_pop[i : i + 2]
            # choose operator
            weights = [count / sum(counts) for count in counts]
            op = random.choices(operators, weights=weights)[0]
            match op:
                case "mutual":
                    ## Mutualism Phase
                    idx, jdx = pair[0], pair[1]
                    mutual_vector = (
                        self.pop[idx].solution + self.pop[jdx].solution
                    ) / 2
                    bf1, bf2 = self.generator.integers(1, 3, 2)
                    xi_new = self.pop[idx].solution + self.generator.random() * (
                        self.g_best.solution - bf1 * mutual_vector
                    )
                    xj_new = self.pop[jdx].solution + self.generator.random() * (
                        self.g_best.solution - bf2 * mutual_vector
                    )
                    xi_new = self.correct_solution(xi_new)
                    xj_new = self.correct_solution(xj_new)
                    xi_target = self.get_target(xi_new)
                    xj_target = self.get_target(xj_new)
                    if self.compare_target(
                        xi_target, self.pop[idx].target, self.problem.minmax
                    ):
                        self.pop[idx].update(solution=xi_new, target=xi_target)
                        counts[0] = +1
                    if self.compare_target(
                        xj_target, self.pop[jdx].target, self.problem.minmax
                    ):
                        self.pop[jdx].update(solution=xj_new, target=xj_target)
                        counts[0] = +1

                case "parasite":
                    ## Parasitism phase
                    idx, jdx = pair[0], pair[1]
                    temp_idx = self.generator.integers(0, self.problem.n_dims)
                    xi_new = self.pop[jdx].solution.copy()
                    xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
                    xi_new = self.correct_solution(xi_new)
                    xi_target = self.get_target(xi_new)
                    ##changed here from idx to jdx
                    ## compare parasite vector with her origin not another organism
                    if self.compare_target(
                        xi_target, self.pop[jdx].target, self.problem.minmax
                    ):
                        self.pop[idx].update(solution=xi_new, target=xi_target)
                        counts[1] = +1
