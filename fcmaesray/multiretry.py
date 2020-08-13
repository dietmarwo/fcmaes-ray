# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# parallel retry of a list of problems. Uses retry to filter the best ones, 
# then uses coordinated retry to evaluate these.

import ray
import numpy as np
import multiprocessing as mp
from scipy.optimize import OptimizeResult
from fcmaes.optimizer import logger, de_cma
from fcmaes import advretry

def minimize(problems, ids=None, num_retries = min(256, 8*mp.cpu_count()), 
             keep = 0.7, optimizer = de_cma(1500), logger = logger()):
      
    """Minimization of a list of optimization problems by first applying parallel retry
    to filter the best ones and then applying coordinated retry to evaluate these further. 
    Can replace mixed integer optimization if the integer variables are narrowly bound. 
    In this case all combinations of these integer values can be enumerated to generate a
    list of problem instances each representing one combination. See for instance 
    https://www.esa.int/gsp/ACT/projects/gtop/tandem where there is a problem instance for each
    planet sequence.
     
    Parameters
    ----------
    
    problems: list
        list of objects providing name, fun and bounds attributes like fcmaes.astro.Astrofun

    ids:  list, optional
        list of objects corresponding to the list of problems used in logging to identify the 
        problem variant currently logged. If None, the index of the problem 
        variant is used instead.
    
    num_retries:  int, optional
        number of coordinated retries applied in the problem filter for each problem 
        in each iteration.
 
    keep:  float, optional
        rate of the problems kept after each iteration. 100*(1 - keep) % will be deleted. 
                        
    optimizer: optimizer.Optimizer, optional
        optimizer to use for the problem filter.
        
    logger, optional
        logger for log output. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.    
     
    Returns
    -------
    dictionary( optimizer -> ret): scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    solver = multiretry()
    n = len(problems)
        
    for i in range(n):    
        id = str(i+1) if ids is None else ids[i]   
        rps = problem_stats.remote(problems[i], id, num_retries, logger)
        solver.add(problems[i], rps)
        
    while solver.size() > 1:    
        solver.retry(optimizer)
        to_remove = int(round((1.0 - keep) * solver.size()))
        if to_remove == 0:
            to_remove = 1
        solver.remove_worst(to_remove)
        solver.dump_all()
    return solver.all_stats

@ray.remote        
class problem_stats:

    def __init__(self, prob, id, num_retries = 64, logger = logger()):
        self.store = advretry.Store(prob.bounds, logger = logger)
        self.prob = prob
        self.name = prob.name
        self.fun = prob.fun
        self.num_retries = num_retries
        self.retries = 0
        self.value = 0
        self.id = id
        self.ret = None

    def retry(self, optimizer):
        retries = self.retries + self.num_retries
        self.ret = advretry.retry(self.fun, self.store, optimizer.minimize, retries)
        self.retries = retries
        self.value = self.store.get_y_best()
        return self.value
        
    def name(self):
        return self.name

    def id(self):
        return self.id

    def value(self):
        return self.value
    
    def get_x_best(self):
        self.store.get_x_best(), 
    
    def get_y_best(self):
        return self.store.get_y_best(), 
    
    def get_count_evals(self):
        return self.store.get_count_evals()

class multiretry:
    
    def __init__(self):
        self.problems = []
        self.problems_stats = []
        self.all_stats = []
    
    def add(self, problem, problem_stats):
        self.problems.append(problem)        
        self.problems_stats.append(problem_stats)
        self.all_stats.append(problem_stats)
    
    def retry(self, optimizer):
        ys = []
        for ps in self.problems_stats:
            logger().info("problem " + ray.get(ps.name.remote()) + ' ' + str(ray.get(ps.id.remote())))
            ys.append(ps.retry.remote(optimizer))
        for y in ys:
            ray.get(y)
    
    def values(self):
        return np.array([ray.get(ps.value.remote()) for ps in self.problems_stats])
     
    def remove_worst(self, n = 1):
        idx = self.values().argsort()
        self.problems_stats = list(np.asarray(self.problems_stats)[idx])
        for _ in range(n):
            self.problems_stats.pop(-1)

    def size(self):
        return len(self.problems_stats)
                    
    def dump(self):
        for i in range(self.size()):
            ps = self.problems_stats[i]
            logger().info(str(ray.get(ps.id.remote())) + ' ' + str(ray.get(ps.value.remote())))
                
    def values_all(self):
        return np.array([ray.get(ps.value.remote()) for ps in self.all_stats])
 
    def dump_all(self):
        idx = self.values_all().argsort()
        self.all_stats = list(np.asarray(self.all_stats)[idx])
        for i in range(len(self.all_stats)):
            ps = self.all_stats[i]
            logger().info(str(ray.get(ps.id.remote())) + ' ' + str(ray.get(ps.value.remote())))
        
    def result(self):
        idx = self.values_all().argsort()
        self.all_stats = list(np.asarray(self.all_stats)[idx])
        self.problems = list(np.asarray(self.problems)[idx])
        ret = []
        for i in range(len(self.all_stats)):
            ps = self.all_stats[i]
            problem = self.problems[i]
            ret.append([problem, 
                        OptimizeResult(
                            x = ray.get(ps.get_x_best().remote()), 
                            fun = ray.get(ps.get_y_best().remote()), 
                            nfev = ray.get(ps.get_count_evals().remote()), 
                            success=True)])
        