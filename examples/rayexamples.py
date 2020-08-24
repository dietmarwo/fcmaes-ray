# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Examples for fcmaes multinode coordinated retry from https://www.esa.int/gsp/ACT/projects/gtop/

import ray

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Tandem, Sagas, Cassini1minlp
from fcmaes.optimizer import logger, de_cma, de2_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Hh_cpp, Dual_annealing, Differential_evolution, GCLDE_cpp, LCLDE_cpp, LDe_cpp, Sequence
from fcmaesray.rayretry import minimize

min_evals = 1500
max_nodes = 100

problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull(), Sagas(), Cassini1minlp()]
algos = [de2_cma(min_evals), de_cma(min_evals), da_cma(min_evals), Cma_cpp(min_evals), De_cpp(min_evals), Hh_cpp(min_evals),
         Da_cpp(min_evals), Dual_annealing(min_evals), Differential_evolution(min_evals)]

def messengerFullLoop(opt, num, max_time = 1200, log = logger()):    
    problem = MessFull()
    minimizers = None # remote actors created by minimize will be reused
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):    
        ret, minimizers = minimize(problem.fun, problem.bounds, max_nodes, None, num_retries = 20000, 
            value_limit = 12.0, logger = log, optimizer=opt, max_time=max_time, minimizers=minimizers)
    print("solution: ", i+1, ret.fun, str(ret.x))
    for minimizer in minimizers:
        ray.get(minimizer.terminate.remote())
            
def test_all(max_time = 600, num_retries = 10000, num = 20):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, max_time, num_retries, num, value_limit = 1E99) 

def _test_optimizer(opt, problem, max_time = 600, num_retries = 10000, num = 1, value_limit = 100.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    minimizers = None # remote actors created by minimize will be reused
    for i in range(num):
        ret, minimizers = minimize(problem.fun, problem.bounds, max_nodes, None, num_retries, value_limit, log, 
                       optimizer=opt, max_time=max_time, minimizers=minimizers)
        print("solution: ", i+1, ret.fun, str(ret.x))
    for minimizer in minimizers:
        ray.get(minimizer.terminate.remote())

def main():
    # do 'pip install fcmaesray'
    # see https://docs.ray.io/en/master/cluster/index.html 
    # call 'ray start --head --num-cpus=1' on the head node and
    # the ip-adress logged needs to be replaced in the following commands executed at the worker nodes:
    # 'ray start --address=192.168.0.67:6379 --num-cpus=1'
    # adapt ip-adress also in the following ray.init command 
    ray.init(address = "192.168.0.67:6379")#, include_webui=True)
    #ray.init() # for single node tests
    #test_all() # test all problems
    messengerFullLoop(de2_cma(min_evals), 1000) # test messenger full

if __name__ == '__main__':
    main()
        

    