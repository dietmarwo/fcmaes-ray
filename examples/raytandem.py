# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Multi node variant of the determination of the
# TandEM https://www.esa.int/gsp/ACT/projects/gtop/tandem/ problem
# planet sequence, see https://github.com/dietmarwo/fast-cma-es/blob/master/MINLP.adoc

import ray
import multiprocessing as mp
from fcmaes.astro import Tandem
from fcmaes.optimizer import logger, de_cma
from fcmaesray import multiretry
        
def test_multiretry(num_retries = min(256, 8*mp.cpu_count()), 
             keep = 0.7, optimizer = de_cma(1500), logger = logger(), repeat = 50):
    seqs = Tandem(0).seqs
    n = len(seqs)
    problems = [Tandem(i) for i in range(n)]
    ids = [str(seqs[i]) for i in range(n)]
    for _ in range(10):
        # check all variants
        problem_stats = multiretry.minimize(problems, ids, num_retries, keep, optimizer, logger)
        ps = problem_stats[0]
        for _ in range(repeat):
            # improve the best variant
            logger().info("problem " + ray.get(ps.name.remote()) + ' ' + str(ray.get(ps.id.remote())))
            fval = ray.get(ps.retry.remote(optimizer))
            if fval < -1490:
                break           

def main():

    # do 'pip install fcmaesray'
    # see https://docs.ray.io/en/master/cluster/index.html 
    # call 'ray start --head --num-cpus=1' on the head node and
    # the ip-adress logged needs to be replaced in the following commands executed at the worker nodes:
    # 'ray start --address=192.168.0.67:6379 --num-cpus=1'
    # adapt ip-adress also in the following ray.init command 
    #ray.init(address = "192.168.0.67:6379")#, include_webui=True)
    ray.init() # for single node tests
    test_multiretry(repeat = 30)
    
if __name__ == '__main__':
    main()
    