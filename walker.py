import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


class Page:
    def __init__(self, width, height, margin, stepsize):
        self.width = width
        self.height = height
        self.margin = margin
        self.min_x = margin
        self.max_x = width - margin
        self.min_y = margin
        self.max_y = height - margin
        self.stepsize = stepsize
        self.npts_x = int((self.width - 2*self.margin)/self.stepsize)
        self.npts_y = int((self.height - 2*self.margin)/self.stepsize)
        self.pt_available = np.full((self.npts_y,self.npts_x), True)

class Walker:
    def __init__(self, page, idx_x, idx_y, lifetime):
        self.page = page            # Page object
        if idx_x <= self.page.npts_x-1:
            self.idx_x = idx_x
        else:
            self.idx_x = random.randint(0,self.page.npts_x-1)
        if idx_y <= self.page.npts_y-1:
            self.idx_y = idx_y
        else:
            self.idx_y = random.randint(0,self.page.npts_y-1)
            
        self.idx_x_prev = self.idx_x
        self.idx_y_prev = self.idx_y
        self.page.pt_available[self.idx_x,self.idx_y] = False
            
        self.x_current = self.page.margin + self.idx_x*self.page.stepsize
        self.y_current = self.page.margin + self.idx_y*self.page.stepsize
        self.x_prev = self.x_current
        self.y_prev = self.y_current
        
        
        self.alive = True
        self.lifetime = lifetime        # Maximum number of steps the walker can do if not constrained too soon
        
        self.available_directions = {"UP":True, "DOWN":True, "LEFT":True, "RIGHT":True}
        
        self.path_pts = [[self.x_current, self.y_current]]
            
    def plot_initial_position(self):
        fig, ax = plt.subplots(figsize=(self.page.width, self.page.height), dpi=300)
        ax.scatter(self.x_current, self.y_current)
        plt.xlim([0,self.page.width])
        plt.ylim([0,self.page.height])
        
    def determine_available_steps(self):
        # UP
        if self.idx_y == (self.page.npts_y-1):
            self.available_directions["UP"] = False
        elif self.page.pt_available[self.idx_x, self.idx_y+1] == True:
            self.available_directions["UP"] = True
        else:
            self.available_directions["UP"] = False
            
        # DOWN
        if self.idx_y == 0:
            self.available_directions["DOWN"] = False
        elif self.page.pt_available[self.idx_x, self.idx_y-1] == True:
            self.available_directions["DOWN"] = True
        else:
            self.available_directions["DOWN"] = False
        
        # LEFT
        if self.idx_x == 0:
            self.available_directions["LEFT"] = False
        elif self.page.pt_available[self.idx_x-1, self.idx_y] == True:
            self.available_directions["LEFT"] = True
        else:
            self.available_directions["LEFT"] = False
                
        # RIGHT
        if self.idx_x == (self.page.npts_x-1):
            self.available_directions["RIGHT"] = False
        elif self.page.pt_available[self.idx_x+1, self.idx_y] == True:
            self.available_directions["RIGHT"] = True
        else:
            self.available_directions["RIGHT"] = False
            
        print(self.available_directions)
    
    def take_step(self):
        pass
    
        
        
if __name__ == "__main__":
    page = Page(8.5, 11, 0.5, 0.25)
    walker = Walker(page, 6, 5, 11)
    walker.plot_initial_position()
    walker.determine_available_steps()
        