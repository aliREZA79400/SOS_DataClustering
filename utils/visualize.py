import sys
import os
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("utils"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.validators import ValidType , Cluster_Number_Range
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def saver_figurs(model :Any,saver_path_figur:str):
    """
    Args:
        model

        saver_path_figur (str): The destination where the figures are saved
    """


    model.history.save_global_objectives_chart(filename = saver_path_figur + "/goc")

    model.history.save_local_objectives_chart(filename = saver_path_figur  + "/loc")

    model.history.save_global_best_fitness_chart(filename = saver_path_figur + "/gbfc")

    model.history.save_local_best_fitness_chart(filename = saver_path_figur + "/lbfc")

    model.history.save_runtime_chart(filename = saver_path_figur + "/rtc")

    model.history.save_exploration_exploitation_chart(filename = saver_path_figur + "/eec")

    model.history.save_diversity_chart(filename = saver_path_figur + "/dc")






class Visualize:

    K = Cluster_Number_Range(2,10)

    colors = ["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
    labels = ["cluster_A" , "cluster_B","cluster_C","cluster_D","cluster_E","cluster_F","cluster_G","cluster_H","cluster_X","cluster_W"]

    def __init__(self ,K :int , dataset:None , target:None,g_best:None):
        '''
        Args :
        K  = number of clusters
        '''
        self.K = K
        self.dataset = dataset
        self.target = target
        self.g_best = g_best

    @staticmethod                                                      
    def create_clusters(num_clusters : None , g_best:None, dataset:None): 
        distance_from_centers = np.array([[np.linalg.norm(row - center) # distance of row from centers
                                            for center in np.reshape(g_best.solution , (num_clusters,dataset.shape[1]))]
                                            for row in dataset
                                            ]) #c1r1 , c2r1 , cnr1 
                                                #..............ckrn  k = number of clusters  , n = number of data samples
        
        clustered = [[] for i in range(num_clusters)]
        for c in range(num_clusters) :
            for i , row_dis in enumerate(distance_from_centers) :
                    if np.where(row_dis==np.min(row_dis))[0][0] == c:
                        clustered[c].append(dataset[i].tolist())
    
        return clustered



    def draw_clustered_2D(self,save_fig = False):
        clutered_list = Visualize.create_clusters(self.K,self.g_best,self.dataset)
        
        # Create a scatter plot
        plt.figure(figsize=(8, 6))

        colors = Visualize.colors
        labels = Visualize.labels

        for i in range(self.K):
            plt.scatter(np.array(clutered_list[i])[:,0],
                        np.array(clutered_list[i])[:,1],
                        color=colors[i], label=labels[i])

        # Customize the plot
        plt.title('2D Scatter Plot of *Clusterd Data*')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        # Show the plot
        plt.grid(True)
        if save_fig:
            plt.savefig(fname = "Clustered_2D")
        plt.show()



    def draw_clustered_3D(self,save_fig = False):
        clutered_list = Visualize.create_clusters(self.K,self.g_best,self.dataset)
        
        # Create a scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = Visualize.colors
        labels = Visualize.labels

        for i in range(self.K):
            ax.scatter(np.array(clutered_list[i])[:,0],
                        np.array(clutered_list[i])[:,1],
                        np.array(clutered_list[i])[:,2],
                        color=colors[i], label=labels[i])

        # Customize the plot
        ax.set_title('3D Scatter Plot of *Clusterd Data*')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()

        # Show the plot
        plt.grid(True)
        if save_fig :
            plt.savefig(fname = "Clustered_3D")
        plt.show()


    def draw_original_data_2D(self,save_fig = False):

        pd_data = pd.DataFrame(self.dataset)

        pd_data.insert(self.dataset.shape[1],"target", self.target)

        # Create a scatter plot
        plt.figure(figsize=(8, 6))

        colors = Visualize.colors
        labels = Visualize.labels

        for i in range(self.K):#ghabel eslah be tedad original class
            plt.scatter(pd_data[pd_data.iloc[:,self.dataset.shape[1]] == i].iloc[:, 0],
                        pd_data[pd_data.iloc[:,self.dataset.shape[1]] == i].iloc[:, 1],
                        color=colors[i], label=labels[i])


        # Customize the plot
        plt.title('2D Scatter Plot of Original Data')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        # Show the plot
        plt.grid(True)
        if save_fig:
            plt.savefig(fname = "Original_2D")
        plt.show()



    def draw_original_data_3D(self,save_fig = False):

        pd_data = pd.DataFrame(self.dataset)
        pd_data.insert(self.dataset.shape[1],"target", self.target)

        # Create a scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = Visualize.colors
        labels = Visualize.labels

        for i in range(self.K): #ghabel eslah be tedad original class
            ax.scatter(pd_data[pd_data.iloc[:,self.dataset.shape[1]] == i].iloc[:, 0],
                        pd_data[pd_data.iloc[:,self.dataset.shape[1]] == i].iloc[:, 1],
                        pd_data[pd_data.iloc[:,self.dataset.shape[1]] == i].iloc[:, 2],
                        color=colors[i], label=labels[i])


        # Customize the plot
        ax.set_title('3D Scatter Plot of Original Data')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()

        # Show the plot
        plt.grid(True)
        if save_fig:
            plt.savefig(fname = "Original_3D")
        plt.show()

    def draw_all(self,save_fig = False):
        self.draw_clustered_2D(save_fig=save_fig)
        self.draw_original_data_2D(save_fig=save_fig)
        self.draw_clustered_3D(save_fig=save_fig)
        self.draw_original_data_3D(save_fig=save_fig)
                
                              
                          
            
            
         
