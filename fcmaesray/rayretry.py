# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import ray
import os
import sys
import math
import random
import time

import ctypes as ct
import multiprocessing as mp
from multiprocessing import Process
from numpy.random import Generator, MT19937, SeedSequence
from scipy.optimize import OptimizeResult, Bounds

from fcmaes.optimizer import fitting, de_cma, logger
from fcmaes.advretry import Store

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(fun, 
             bounds, 
             max_nodes,
             workers = None,
             num_retries = 5000,
             value_limit = math.inf,
             logger = None,
             popsize = 31, 
             min_evaluations = 1500, 
             max_eval_fac = None, 
             check_interval = 100,
             capacity = 500,
             stop_fittness = None,
             optimizer = None,
             statistic_num = 0,
             max_time = 100000,
             minimizers = None # reused distributed ray actors
             ):   
    """Minimization of a scalar function of one or more variables using 
    coordinated parallel CMA-ES retry.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    max_nodes : int
        maximal number of distributed nodes (physical CPUs) used.
    workers : int
        number of parallel processes used.
        Use None if using CPUs with different number of cores, then mp.cpu_count() is used at each node. 
    num_retries : int, optional
        Number of optimization retries, only used to calculate the evaluation limit increment.
        Because nodes may have different CPU-power, max_time is used to terminate optimization.   
    value_limit : float, optional
        Upper limit for optimized function values to be stored. 
        This limit needs to be carefully set to a value which is seldom
        found by optimization retry to keep the store free of bad runs.
        The crossover offspring of bad parents can
        cause the algorithm to get stuck at local minima.   
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    popsize = int, optional
        CMA-ES population size used for all CMA-ES runs. 
        Not used for differential evolution. 
        Ignored if parameter optimizer is defined. 
    min_evaluations : int, optional 
        Initial limit of the number of function evaluations. Only used if optimizer is undefined, 
        otherwise this setting is defined in the optimizer. 
    max_eval_fac : int, optional
        Final limit of the number of function evaluations = max_eval_fac*min_evaluations
    check_interval : int, optional
        After ``check_interval`` runs the store is sorted and the evaluation limit
        is incremented by ``evals_step_size``
    capacity : int, optional
        capacity of the evaluation store. Higher value means broader search.
    stop_fittness : float, optional 
         Limit for fitness value. optimization runs terminate if this value is reached. 
    optimizer : optimizer.Optimizer, optional
        optimizer to use. Default is a sequence of differential evolution and CMA-ES.
        Since advanced retry sets the initial step size it works best if CMA-ES is 
        used / in the sequence of optimizers. 
    max_time : int, optional
        Time limit in seconds
    minimizers : list of Minimizer, optional
        remote ray actors for reuse. Need to be terminated finally - call terminate.remote().
   
    Returns
    -------
    res : scipy.OptimizeResult, list of Minimizer
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
        
    if minimizers is None:
        # determine the number of nodes connected
        # len(ray.nodes()) seems not not work for ray 0.8.6
        ipadrs = set(ray.get([remote_ipadr.remote() for _ in range(2000)]))
        if not logger is None:
            logger.info("cluster optimization on nodes: " + str(ipadrs))
        nodes = min(max_nodes, len(ipadrs))        
        minimizers = []   
        ips = {}
        master_ip = ipadr()
        # start the ray remote minimizers
        for rid in range(nodes):
            minimizer = None
            while minimizer is None:
                minimizer = Minimizer.remote(
                    master_ip,
                    rid, 
                    fun)
                ip = ray.get(minimizer.ip.remote())
                if ip in ips:
                    minimizer.__ray_terminate__.remote()
                    minimizer = None
                else:
                    ips[ip] = minimizer
                    minimizers.append(minimizer)
    # else reuse minimizers
    for minimizer in minimizers:
        ray.get(minimizer.init.remote(
                    bounds, 
                    workers,
                    value_limit,
                    num_retries,
                    popsize, 
                    min_evaluations, 
                    max_eval_fac, 
                    check_interval,
                    capacity,
                    stop_fittness,
                    optimizer))
        ray.get(minimizer.retry.remote())
    
    store = Store(bounds, max_eval_fac, check_interval, capacity, logger, num_retries, statistic_num)
    # each second exchange results
    for _ in range(max_time):
        time.sleep(1)
        improved = False
        for minimizer in minimizers:
            # polling is ugly but experiments with a coordinator actor failed
            # should be refactored when ray improves
            if ray.get(minimizer.is_improved.remote()):
                improved = True
                y, xs, lower, upper = ray.get(minimizer.best.remote(0))
                store.add_result(y, xs, lower, upper, 0)# 
                for recipient in minimizers:
                    if recipient != minimizer:
                        recipient.add_result.remote(y, xs, lower, upper)
        if improved:
            store.sort()
    time.sleep(10) # time to transfer all messages
    # terminate subprocesses
    for minimizer in minimizers:
        ray.get(minimizer.terminate_procs.remote())
          
    return OptimizeResult(x=store.get_x_best(), 
                          fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True), minimizers 

def ipadr():
    return str(ray.services.get_node_ip_address())

@ray.remote
def remote_ipadr():
    return ipadr()
        
@ray.remote
class Minimizer(object):   
      
    def __init__(self, 
                 master_ip,
                 rid,
                 fun, 
                 ):      
        self.master_ip = master_ip
        self.rid = rid
        self.fun = fun
     
    def init(self,
             bounds, 
             workers,
             value_limit,
             num_retries,
             popsize, 
             min_evaluations, 
             max_eval_fac, 
             check_interval,
             capacity,
             stop_fittness,
             optimizer
             ):
        if optimizer is None:
            optimizer = de_cma(self.min_evaluations, popsize, stop_fittness)     
        if max_eval_fac is None:
            max_eval_fac = int(min(50, 1 + num_retries // check_interval))
        self.store = Store(bounds, max_eval_fac, check_interval, capacity, None, num_retries)                       
        self.improved = mp.RawValue(ct.c_bool, False)
        self.bounds = bounds
        self.workers = mp.cpu_count() if workers is None else workers
        self.value_limit = value_limit
        self.num_retries = num_retries
        self.popsize = popsize
        self.min_evaluations = min_evaluations
        self.max_eval_fac = max_eval_fac
        self.check_interval = check_interval
        self.capacity = capacity
        self.stop_fittness = stop_fittness
        self.optimizer = optimizer
        self.procs = []
                      
    def ip(self):
        return ipadr()
     
    def retry(self):        
        self.procs = _retry(self)
    
    def terminate_procs(self):
        for p in self.procs: 
            p.terminate() # creates some noisy ray logs
       
    def terminate(self):
        self.terminate_procs()
        time.sleep(0.5)
        self.__ray_terminate__()
        return True
    
    def is_improved(self):
        improved = self.improved.value # use only once
        self.improved.value = False # wait for next improvement
        return improved
    
    def best(self, pid):
        with self.store.add_mutex:
            self.store.sort()
        return self.store.get_y(pid), self.store.get_x(pid), self.store.get_lower(pid), self.store.get_upper(pid)

    def get_upper(self, pid):
        return self.uppers[pid*self.dim:(pid+1)*self.dim]

    def add_result(self, y, xs, lower, upper):
        """registers an optimization result at local store."""
        self.store.add_result(y, xs, lower, upper, 0)
            
def _retry(minimizer):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(minimizer.workers)]
    procs = [Process(target=_retry_loop,
            args=(pid, rgs, minimizer)) for pid in range(minimizer.workers)]
    [p.start() for p in procs]
    return procs

def _retry_loop(pid, rgs, minimizer):
    fun = minimizer.fun
    store = minimizer.store
    optimize = minimizer.optimizer.minimize
    num_retries = minimizer.num_retries
    value_limit = minimizer.value_limit
    
    #reinitialize logging config for windows -  multi threading fix
    if 'win' in sys.platform and not store.logger is None:
        store.logger = logger()
    
    while True:   
        try:
            store.get_runs_compare_incr(num_retries)        
            if _crossover(fun, store, optimize, rgs[pid], minimizer):
                continue
            dim = len(store.lower)
            sol, y, evals = optimize(fun, Bounds(store.lower, store.upper), None, 
                                     [random.uniform(0.05, 0.1)]*dim, rgs[pid], store)
            _add_result(y, sol, store.lower, store.upper, evals, value_limit, store, minimizer)
        except Exception as ex:
            continue
 
def _crossover(fun, store, optimize, rg, minimizer):
    if random.random() < 0.5:
        return False
    y0, guess, lower, upper, sdev = store.limits()
    if guess is None:
        return False
    guess = fitting(guess, lower, upper) # take X from lower
    try:       
        sol, y, evals = optimize(fun, Bounds(lower, upper), guess, sdev, rg, store)
        _add_result(y, sol, lower, upper, evals, y0, store, minimizer) 
    except:
        return False   
    return True       

def _add_result(y, xs, lower, upper, evals, limit, store, minimizer):
    if y < minimizer.value_limit and y < store.best_y.value:
        minimizer.improved.value = True # register improvement         
    store.add_result(y, xs, lower, upper, evals, limit)  
            