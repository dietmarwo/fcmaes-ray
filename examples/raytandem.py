# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Multi node variant of the determination of the
# TandEM https://www.esa.int/gsp/ACT/projects/gtop/tandem/ problem
# planet sequence, see https://github.com/dietmarwo/fcmaes-ray/blob/master/MULTI.adoc

import ray
import time
import multiprocessing as mp
from fcmaes.astro import Tandem
from fcmaes.optimizer import logger, de2_cma, dtime
from fcmaesray import multiretry
from fcmaesray import rayretry
        
def test_multiretry(num_retries = min(256, 8*mp.cpu_count()), 
             keep = 0.7, optimizer = de2_cma(1500), logger = logger(), repeat = 10):
    seqs = Tandem(0).seqs
    n = len(seqs)
    problems = [Tandem(i) for i in range(n)]
    ids = [str(seqs[i]) for i in range(n)]
    t0 = time.perf_counter()
    for _ in range(repeat):
        # check all variants
        problem_stats = multiretry.minimize(problems, ids, num_retries, keep, optimizer, logger)
        ps = problem_stats[0]
        
#         for _ in range(10):
#             # improve the best variant using only one node
#             fval = ray.get(ps.retry.remote(optimizer))
#             logger.info("improve best variant " + ray.get(ps.name.remote()) 
#                         + ' ' + str(ray.get(ps.id.remote()))
#                         + ' ' + str(ray.get(ps.value.remote())) 
#                         + ' time = ' + str(dtime(t0)))
#             if fval < -1490:
#                 break           
            
        # optimize best variant starting from scratch using all nodes
        logger.info("improve best variant " + ray.get(ps.name.remote()) 
                    + ' ' + str(ray.get(ps.id.remote()))
                    + ' ' + str(ray.get(ps.value.remote())) 
                    + ' time = ' + str(dtime(t0)))
        problem = problems[ray.get(ps.index.remote())]
        _rayoptimizer(optimizer, problem, 1, max_time = 1200, log = logger)

def _rayoptimizer(opt, problem, num, max_time = 1200, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    minimizers = None # remote actors created by minimize will be reused
    for i in range(num):
        ret, minimizers = rayretry.minimize(problem.fun, problem.bounds, 100, None, 20000, 0, log, 
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
    test_multiretry(repeat = 1)
    
if __name__ == '__main__':
    main()
    