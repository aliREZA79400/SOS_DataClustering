import os
import sys
import argparse
import time

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("SOS"))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

parent_dir_ = os.path.dirname(parent_dir)
sys.path.append(parent_dir_)

import mealpy 
import numpy as np
from mealpy.bio_based import SOS

from utils import datasets , generate_bound , visualize , saver_model
from SOS.problem import Data_Clustering

parser = argparse.ArgumentParser(description="Runner of Algorithm")

parser.add_argument("k",type=int, help="number of clusters (int)")
parser.add_argument("epoch" ,type=int, help="number of itrations(int)")
parser.add_argument("pop_size",type=int, help="number of population(int)")
parser.add_argument("times_run",type=int,help="number of times that each dataset runs")

args = parser.parse_args()

def main_runner(k:int , epoch:int ,pop_size:int , times:int):

    dataset_names = ("Iris_Dataset" , "Breast_Cancer" , "Balance_Scale" , "Seeds" , "Statlog" , "Contraceptive_Method_Choice" , "Haberman_s_Survival", "Wine")
    
    all_results = {}

    for dt_name in dataset_names :

        dataset_results = {}

        print(dt_name)
        
        for run in range(times):
            ### define datatset and target

            dataset , target = datasets.retu_dataset(name=dt_name)

            ### generate bound

            bound = generate_bound.generate(K=k,dataset=dataset)

            ### define problem with class {log training process}

            problem_ins = Data_Clustering(bounds=bound,
                                        K=k,
                                        dataset=dataset)

            ### define model parameters dict 

            ### build model (instance)

            model = SOS.OriginalSOS(epoch=epoch,pop_size=pop_size)

            ### train model  with solve ( training modes )
            start_time = time.time()
            g_best  = model.solve(problem=problem_ins)
            end_time = time.time()

            # save model
            # saver_model.saver_model(model=model , save_path = os.getcwd()+ "/results" + f"_{dt_name}"+ f"_{run}" +".pkl")
            
            dataset_results[str(run)] = {
                "solution" : g_best.solution,
                "fitness" : g_best.target.fitness,
                "elapsed_time" : end_time - start_time
            }

        all_results[dt_name] = dataset_results
        print(all_results)

    return all_results

###Extra
# tuning (not for now)
# define termination condition (not for now)

# Visualize
# vis = visualize.Visualize(args.k,dataset,target=target,g_best=g_best)

if __name__ == "__main__":

    main_runner(k=args.k , epoch=args.epoch,pop_size=args.pop_size,times=args.times_run)

    # print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    
    # visualize.saver_figurs(model=model , saver_path_figur=os.getcwd()+ "/figures")

    # vis.draw_all()

