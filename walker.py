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
    def __init__(self, page, idx_x, idx_y, lifetime, direction_prefs):
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
        self.direction_prefs = direction_prefs      # list of 4 values from 0 to 1 to determine directional preference
        
        self.available_directions = {"UP":True, "DOWN":True, "LEFT":True, "RIGHT":True}
        
        self.path_pts_x = [self.x_current]
        self.path_pts_y = [self.y_current]
            
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
            
        # print(self.available_directions)
    
    def take_step(self):
        while True:
            # Check if there are any available directions
            self.determine_available_steps()
            if (self.available_directions["UP"] == False) and (self.available_directions["DOWN"] == False) and (self.available_directions["LEFT"] == False) and (self.available_directions["RIGHT"] == False):
                self.alive = False
                print("Walker died from getting trapped")
                break                
            
            while self.alive:
                # Generate random numbers
                val = random.uniform(0, 1)
                thresh_met = []
                
                for i,thresh in enumerate(self.direction_prefs):
                    if val < thresh:
                        thresh_met.append(thresh)
                
                while True:
                    if len(thresh_met) == 0:
                        break
                    # find the index of the highest met threshold
                    idx = self.direction_prefs.index(np.max(thresh_met))
                    
                    # Check if the chosen direction is available. If not, remove
                    # that direction from the running and choose the next highest
                    # threshold
                    if idx == 0:
                        key = "UP"
                    elif idx == 1:
                        key = "DOWN"
                    elif idx == 2:
                        key = "LEFT"
                    elif idx == 3:
                        key == "RIGHT"
                    if self.available_directions[key] == True:
                        break
                    else:
                        thresh_met.remove(np.max(thresh_met))
                        if len(thresh_met) > 0:
                            continue
                        else:
                            break
                
                if len(thresh_met) == 0:
                    continue
                
                # Save previous locations and indices
                self.x_prev = self.x_current
                self.y_prev = self.y_current
                self.idx_x_prev = self.idx_x
                self.idx_y_prev = self.idx_y
                
                # Based on idx, take a step in the chosen direction
                # UP
                if idx == 0:
                    self.idx_y += 1
                    # print("UP")
                    
                # DOWN
                elif idx == 1:
                    self.idx_y -= 1
                    # print("DOWN")
                
                # LEFT
                elif idx == 2:
                    self.idx_x -= 1
                    # print("LEFT")
                    
                # RIGHT
                elif idx == 3:
                    self.idx_x += 1
                    # print("RIGHT")
                    
                self.x_current = self.page.margin + self.idx_x*self.page.stepsize
                self.y_current = self.page.margin + self.idx_y*self.page.stepsize
                
                # Append the new step position
                self.path_pts_x.append(self.x_current)
                self.path_pts_y.append(self.y_current)
                
                # Adjust the available points array
                self.page.pt_available[self.idx_x,self.idx_y] = False
                
                break
            break
        
    
    def take_random_step(self):
        while True:
            # Check if there are any available directions
            self.determine_available_steps()
            if (self.available_directions["UP"] == False) and (self.available_directions["DOWN"] == False) and (self.available_directions["LEFT"] == False) and (self.available_directions["RIGHT"] == False):
                self.alive = False
                # print("Walker died from getting trapped")
                break                
            else:
                direction_indices = []
                if self.available_directions["UP"] == True:
                    direction_indices.append(0)
                if self.available_directions["DOWN"] == True:
                    direction_indices.append(1)
                if self.available_directions["LEFT"] == True:
                    direction_indices.append(2)
                if self.available_directions["RIGHT"] == True:
                    direction_indices.append(3)
                    
                # Randomly choose the direction index
                idx = random.choice(direction_indices)
                # Based on idx, take a step in the chosen direction
                # UP
                if idx == 0:
                    self.idx_y += 1
                    # print("UP")
                    
                # DOWN
                elif idx == 1:
                    self.idx_y -= 1
                    # print("DOWN")
                
                # LEFT
                elif idx == 2:
                    self.idx_x -= 1
                    # print("LEFT")
                    
                # RIGHT
                elif idx == 3:
                    self.idx_x += 1
                    # print("RIGHT")
                    
                self.x_current = self.page.margin + self.idx_x*self.page.stepsize
                self.y_current = self.page.margin + self.idx_y*self.page.stepsize
                
                # Append the new step position
                self.path_pts_x.append(self.x_current)
                self.path_pts_y.append(self.y_current)
                
                # Adjust the available points array
                self.page.pt_available[self.idx_x,self.idx_y] = False
            

            
    def walk(self):
        steps = 1
        
        # Walk until the walker is dead
        while self.alive:
            # self.take_step()
            self.take_random_step()
            steps += 1
            if steps > self.lifetime:
                self.alive = False
                print("Walker reached full life")
                
    def plot_path(self, fig, ax):
        # fig, ax = plt.subplots(figsize=(self.page.width, self.page.height), dpi=300)
        ax.plot(self.path_pts_x, self.path_pts_y, 'k')
        plt.xlim([0,self.page.width])
        plt.ylim([0,self.page.height])
        
        
if __name__ == "__main__":
    page = Page(11, 8.5, 0.5, 0.05)
    direction_prefs = [0.51, 0.7, 0.49, 0.39]
    
    n_walkers = 400
    fig, ax = plt.subplots(figsize=(page.width, page.height), dpi=300)
    
    
    for i in range(n_walkers):
        idx_x = random.randint(20,150)
        idx_y = random.randint(20,150)
        
        walker = Walker(page, idx_x, idx_y, 100, direction_prefs)
        # walker.plot_initial_position()
        # walker.determine_available_steps()
        try:
            walker.walk()
            walker.plot_path(fig, ax)
        except IndexError:
            continue
        