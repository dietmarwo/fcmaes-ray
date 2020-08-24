# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# MINLP variant of the determination of the
# TandEM https://www.esa.int/gsp/ACT/projects/gtop/tandem/ problem
# planet sequence, see https://github.com/dietmarwo/fcmaes-ray/blob/master/MULTI.adoc

import ray

from fcmaes.astro import Tandem_minlp
from fcmaes.optimizer import logger, de2_cma
from fcmaesray.rayretry import minimize

max_nodes = 100

def test_tandem_minlp(opt, num, max_time = 1500, log = logger()):    
    problem = Tandem_minlp()
    minimizers = None # remote actors created by minimize will be reused
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):    
        ret, minimizers = minimize(problem.fun, problem.bounds, max_nodes, None, num_retries = 20000, 
            value_limit = 12.0, logger = log, optimizer=opt, max_time=max_time, minimizers=minimizers)
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
    test_tandem_minlp(de2_cma(1500), 10)
    
if __name__ == '__main__':
    main()
    