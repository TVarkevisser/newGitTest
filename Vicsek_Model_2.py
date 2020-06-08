# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:10:24 2020

@author: Colin Meulblok & Thijs Varkevisser
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

"""
In 2d
noise values between 0 and 2pi;
noise of 0 -> system is deterministic

system_size has to be dividable by interaction_radius
Only works for systems larger then 3 interaction_radius
"""

matplotlib.rcParams.update({'font.size': 14})

class Active_System():
    def __init__(self, number_particles, density, interaction_radius, step_size, noise_value):
        
        self.number_particles = number_particles
        self.density = density
        self.interaction_radius = interaction_radius
        self.step_size = step_size
        self.noise_value = noise_value
        
        self.system_size = np.sqrt(number_particles / density)
        
        
        self.positions = self.system_size * np.random.rand(number_particles, 2)
        #2 is the number of dimensions of the system
        
        velocity_angles = np.random.uniform(-1*np.pi, np.pi, number_particles)
        self.velocities = np.transpose(np.array([np.cos(velocity_angles), np.sin(velocity_angles)]))
                
        #A straight forward calculation would be to just loop over all particles
        #This is very inefficient for large number of particles, thus each
        #particle is assigned to a box of size interaction_radius, so we only
        #have to check the particles inside the neighboring boxes
        self.number_boxes_on_axis = self.system_size // self.interaction_radius +1
        self.boxes_of_particles = self.positions // self.interaction_radius
        
        
    def Update_Positions(self):
        #first the positions are updated for to the new positions and the boxes 
        #are reassigned
        
        
        #now using the velocity angles the new positions are calculated
        velocity = self.step_size*self.velocities
        self.positions = (self.positions + velocity) % self.system_size
        
        self.boxes_of_particles = self.positions // self.interaction_radius
        
    def Pos_VelAngle_Accepted_BoxArgument(self, box_x, box_y):
        """Give the particles that we are going to check"""
        #thus the box looked at is box_x, box_y
        #the particles we want to check then are in box (x,y) and the eight
        #surrounding boxes
        box_left = (box_x-1) % self.number_boxes_on_axis
        box_right = (box_x+1) % self.number_boxes_on_axis
        boxes_on_x_accepted = (1*(self.boxes_of_particles[:,0] == box_x) + 
                               1*(self.boxes_of_particles[:,0] == box_left) + 
                               1*(self.boxes_of_particles[:,0] == box_right))
        
        box_down = (box_y-1) % self.number_boxes_on_axis
        box_up = (box_y+1) % self.number_boxes_on_axis
        boxes_on_y_accepted = (1*(self.boxes_of_particles[:,1] == box_y) + 
                               1*(self.boxes_of_particles[:,1] == box_down) + 
                               1*(self.boxes_of_particles[:,1] == box_up))
        
        accepted_particles = boxes_on_x_accepted * boxes_on_y_accepted
        accepted_positions = self.positions[accepted_particles==1]
        accepted_velocities = self.velocities[accepted_particles==1]
        return accepted_positions, accepted_velocities
    
    def Particles_Within_Interaction_radius(self, position, accepted_positions):
        """Checks the particles that were determined to be checked
        Returns a Boolean array with the positions of the particles within 
        interation radius."""
        pos_difference = accepted_positions-position
        pos_difference = np.abs(pos_difference)
        pos_through_boundary = pos_difference > 2*self.interaction_radius
        pos_difference -= pos_through_boundary * self.system_size
        
        distances_squared = pos_difference[:,0]**2 + pos_difference[:,1]**2
        particles_inside_interaction = (distances_squared <= self.interaction_radius**2)
        return particles_inside_interaction
        
    #def Polarization(self):
        #order parameter
        #velocity = np.array([np.cos(self.velocity_angles), np.sin(self.velocity_angles)])
        #current_polarization = np.sqrt(np.dot(np.sum(velocity,1), np.sum(velocity,1))) / self.number_particles
        #return current_polarization
    
    def White_Noise(self, N):
        angles = np.random.uniform(-1*self.noise_value/2, self.noise_value/2, N)
        return np.transpose(np.array([np.cos(angles), np.sin(angles)]))   

    def Add_White_Noice(self, velocities)  :
        velocity_angles = np.arccos(velocities[:,0]) # [:,0] are all x values
        minus_mask = 2*(velocities[:,1] > 0)-1 # if y-value is < 0, theta -> -theta
        velocity_angles *= minus_mask
        
        noice_angles = np.random.uniform(-1*self.noise_value/2, self.noise_value/2, self.number_particles)
        velocity_angles += noice_angles
        velocities = np.transpose(np.array([np.cos(velocity_angles), np.sin(velocity_angles)]))
        return velocities
    
    
    
    def Update_Velocity(self):
        new_velocities = np.zeros((self.number_particles,2))
        for n in np.arange(self.number_particles):
            box_x = self.boxes_of_particles[n,0]
            box_y = self.boxes_of_particles[n,1]
            
            position = self.positions[n,:]
            
            accepted_positions, accepted_velocities = self.Pos_VelAngle_Accepted_BoxArgument(box_x,box_y)
            particles_inside_interaction = self.Particles_Within_Interaction_radius(position, accepted_positions)
            
            velocities_inside_interaction = accepted_velocities[particles_inside_interaction==1]
            
            new_velocity = np.sum(velocities_inside_interaction,0) 
            new_velocity /= np.linalg.norm(new_velocity)
            #new_velocity += white_noise[n,:]
            #new_velocity /= np.linalg.norm(new_velocity)
            new_velocities[n,:] = new_velocity
        
        new_velocities = self.Add_White_Noice(new_velocities)
        self.velocities = new_velocities
        
    def Run_System(self, time):
        
        all_positions = np.zeros((self.number_particles,2,time))
        all_velocities = np.zeros((self.number_particles,2,time))
        for t in np.arange(time):
            all_positions[:,:,t] = self.positions
            all_velocities[:,:,t] = self.velocities
            
            self.Update_Positions()
            self.Update_Velocity()
        return all_positions, all_velocities
            
    def Show_Current_configuration(self, show = False, save_at = os.getcwd()+'\\', save_as = 'Current_Configuration.png'):
        theta=np.arange(-np.pi,np.pi+1, np.pi/24)

        plt.figure(figsize=(8,8))
        plt.scatter(self.positions[:,0], self.positions[:,1], marker='.', color='blue')
        plt.plot(self.interaction_radius*np.sin(theta)+self.system_size/2, self.interaction_radius*np.cos(theta)+self.system_size/2,
                 color='darkgoldenrod', lw=2, ls='-')
        plt.plot(np.array([self.system_size/2, self.system_size/2 + self.interaction_radius]), 
                 np.array([self.system_size/2, self.system_size/2]), lw=2, color='darkgoldenrod', ls='--')
        plt.text(self.system_size/2, self.system_size/2, r'$\Delta r$', color='darkgoldenrod', fontsize=14)
        
        plt.title(r'Configuration, where $\eta=$' + str(round(self.noise_value,2)))
        plt.xlim(0 ,self.system_size)
        plt.ylim(0, self.system_size)
        plt.xticks([])
        plt.yticks([])
        
        if show:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(save_at + save_as,  dpi=300)
        



