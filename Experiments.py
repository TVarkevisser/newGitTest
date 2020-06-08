# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:06:56 2020

@author: Colin Meulblok
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import time
from matplotlib import cm # for colors in video
from celluloid import Camera

from Vicsek_Model_2 import Active_System
matplotlib.rcParams.update({'font.size': 14})

def Experiment_Critical_noise_value(all_number_particles, density, interaction_radius, step_size, 
                                    noise_values, equalibration_time, running_time):
    
    polarizations = np.zeros((np.size(noise_values), np.size(all_number_particles)))
    
    print('starting experiment 1')
    t1 = time.time()
    
    loop1_counter = 0; 
    for number_particles in all_number_particles:
        loop2_counter = 0
        for noise_value in noise_values:
            Model = Active_System(number_particles, density, interaction_radius, step_size, noise_value)
            
            Model.Run_System(equalibration_time)
            all_positions, all_velocities, polarization = Model.Run_System(running_time)
            
            polarizations[loop2_counter, loop1_counter] = np.mean(polarization)
            loop2_counter += 1
            
            t2 = time.time()
            print(str(loop2_counter) + ' is done of the ' + np.size(noise_values) + '; Elapsed time is ' 
                  + str(round((t2-t1)/60, 2)) + ' minutes')
        loop1_counter += 1
           
    np.savetxt(os.getcwd()+'\\Results\\Exp1\\polarizations.txt', polarizations)
    
    ### Now, make the two plots ###
    critical_noise = 3.1
    
    ### plot 1:
    clrs = ['salmon', 'orangered', 'indianred', 'darkred']
    for i in np.arange(np.size(all_number_particles)):
        plt.scatter(noise_values, polarizations[:,i], marker='d', color = clrs[i % 4], 
                    label=r'N = ' + str(round(all_number_particles[i])) )
    
    plt.plot(critical_noise*np.array([1., 1.]), np.array([-1, 5]), lw=2, ls='--', color = 'darkgoldenrod')
    plt.text(3, 0, r'$\eta_c$', color='darkgoldenrod')
    
    plt.xlim(-0.2, noise_values[-1]+0.2)
    plt.ylim(-0.05, 1.05)
    
    plt.xticks(np.arange(0, 2*np.pi + 0.1, np.pi/4), [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$', r'$2\pi$'] )
    plt.yticks(np.arange(0,1.1, 0.2))
    
    plt.grid()
    plt.legend()
    
    plt.xlabel(r'noise value $\eta$')
    plt.ylabel(r'polarization $v_0$')
    
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\Results\\Exp1\\OrderParameterPlot.png', dpi=300)
    plt.close()
    
    ### plot 2:
    
    noise = (1. - noise_values/critical_noise)
    noise_OrderedPhase = noise[noise>0] #only the positive values are in the ordered phase
    polarization = polarizations[noise>0, -1]
    
    P = np.polyfit(np.log(noise_OrderedPhase), np.log(polarization), 1)
    
    plt.scatter(noise_OrderedPhase, polarization, marker='d', color = 'maroon', label = 'Experimental value')
    plt.plot(noise_OrderedPhase, np.exp(P[1]) * noise_OrderedPhase**P[0], ls='--', lw=2, 
             color = 'darkgoldenrod', label = r'Fitted line, where $\beta=$' + str(round(P[0],2)) )
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(0.01, 10)
    plt.ylim(0.1, 10)
    
    plt.xlabel(r'$(1-\eta/\eta_c)$')
    plt.ylabel(r'polarization $v_0$')
    
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\Results\\Exp1\\Fit_CE_beta.png', dpi=300)
    plt.close()


def Experiment_Critical_Density(system_size, densities, interaction_radius, step_size, 
                                noise_value, equalibration_time, running_time):
    
    polarizations = np.zeros(np.size(densities))
    
    loop_counter = 0
    for density in densities:
        number_particles = density * system_size**2
        Model = Active_System(number_particles, density, interaction_radius, step_size, noise_value)
            
        Model.Run_System(equalibration_time)
        
        all_positions, all_velocities, polarization = Model.Run_System(running_time)
        
        polarizations[loop_counter] = np.mean(polarization)
        loop_counter += 1
    
    np.savetxt(os.getcwd()+'\\Results\\Exp2\\polarizations.txt', polarizations)
    
    ### Now again 2 plots are made ###
    
    critical_density = 1.5
    ### plot 1:
    plt.scatter(densities, polarizations, marker='d', color='mediumblue')
    
    plt.plot(critical_density*np.ones(2), np.array([-1,5]), color='darkgoldenrod', lw=2, ls='--')
    plt.plot(critical_density + 0.1 , 0, r'$\rho_c$', color='darkgoldenrod')
    
    plt.xlim(-.1, densities[-1])
    plt.ylim(-0.05, 1.05)
    
    plt.xlabel(r'density $\rho_0$')
    plt.ylabel(r'polarization $v_0$')
    
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\Results\\Exp2\\DensityPlot.png', dpi=300)
    plt.close()
    
    ### plot 2:
    
    density = (1. - densities/critical_density)
    density_OrderedPhase = density[density>0] #only the positive values are in the ordered phase
    polarization = polarizations[density>0]
    
    P = np.polyfit(np.log(density_OrderedPhase), np.log(polarization), 1)
    
    plt.scatter(density_OrderedPhase, polarization, marker='d', color = 'navy', label = 'Experimental value')
    plt.plot(density_OrderedPhase, np.exp(P[1]) * density_OrderedPhase**P[0], ls='--', lw=2, 
             color = 'darkgoldenrod', label = r'Fitted line, where $\gamma=$' + str(round(P[0],2)) )
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(0.01, 10)
    plt.ylim(0.1, 10)
    
    plt.xlabel(r'$(1-\rho_0/\rho_c)$')
    plt.ylabel(r'polarization $v_0$')
    
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\Results\\Exp2\\Fit_CE_delta.png', dpi=300)
    plt.close()

def Main():
    equalibration_time = 500
    running_time = 2000
    interaction_radius = 1.
    step_size = 0.05
    
    density = 2.
    
    all_number_particles = np.array([64, 512, 2048, 4096])
    
    maximal_noise = 2*np.pi; steps_noise = np.pi/6
    noise_values = np.arange(0, maximal_noise + 0.01, steps_noise)
    
    Experiment_Critical_noise_value(all_number_particles, density, interaction_radius, step_size, 
                                    noise_values, equalibration_time, running_time)
    """
    system_size = 20
    noise_value = 2.0
    densities = np.array([np.arange(0.2, 2, 0.2), np.arange(2, 10, 1)])
    densities = densities.reshape(np.size(densities))
    
    Experiment_Critical_Density(system_size, densities, interaction_radius, step_size, 
                                noise_value, equalibration_time, running_time)
    """
    
def Create_Data(number_particles, density, interaction_radius, step_size, noise_value, equalibration_time, running_time):
    """
    Saves positions and velocities during the equilibration and run phase of 
    the system as numpy arrays in a folder called 'Data' which must be 
    contained in the working directory.

    Parameters
    ----------
    number_particles : int
    density : int
    interaction_radius : int
        Often chosen to be 1.
    step_size : float
        Must be smaller than interaction_radius, typically 0.03.
    noise_value : float
        Number in [0,1], determines the noise angle added to the particles.
    equalibration_time : int
    running_time : int

    Returns
    -------
    None.

    """
    Model = Active_System(number_particles, density, interaction_radius, step_size, noise_value)
    equ_positions, equ_velocities = Model.Run_System(equalibration_time)
    run_positions, run_velocities = Model.Run_System(running_time)
    loc = os.getcwd() + '\\Data\\'
    params = str(number_particles)+'_'+str(density)+'_'+str(interaction_radius)+'_'+str(step_size)+'_'+str(noise_value)+'_'+str(equalibration_time)+'_'+str(running_time)
    np.save(loc+'equ_positions_'+params+'.npy', equ_positions)
    np.save(loc+'equ_velocities_'+params+'.npy', equ_velocities)
    np.save(loc+'run_positions_'+params+'.npy', run_positions)
    np.save(loc+'run_velocities_'+params+'.npy', run_velocities)
    
def Load_Data(number_particles, density, interaction_radius, step_size, noise_value, equalibration_time, running_time):
    #loc = os.getcwd() + '\\Data\\'
    loc = r"C:\Users\thijs\Documents\COP_Data\\"
    params = str(number_particles)+'_'+str(density)+'_'+str(interaction_radius)+'_'+str(step_size)+'_'+str(noise_value)+'_'+str(equalibration_time)+'_'+str(running_time)    
    #equ_positions = np.load(loc+'equ_positions_'+params+'.npy')
    #equ_velocities = np.load(loc+'equ_velocities_'+params+'.npy')
    print(loc+'run_positions_'+params+'.npy')
    run_positions = np.load(loc+'run_positions_'+params+'.npy')
    #run_velocities = np.load(loc+'run_velocities_'+params+'.npy')
    return run_positions

def Polarization(velocities):
    magnitude = np.sqrt(np.dot(np.sum(velocities,0), np.sum(velocities,0))) / np.size(velocities[:,0])
    direction = np.sum(velocities,0)
    direction /= np.linalg.norm(direction)
    return magnitude, direction


def Create_Video():
    tic = time.time()
    
    number_particles = 100
    density = 1
    interaction_radius = 1
    step_size = 0.05
    noise_value = 0.5*np.pi
    system_size = np.sqrt(number_particles/density) # for plot
    
    Model = Active_System(number_particles, density, interaction_radius, step_size, noise_value)
    
    
    equalibration_time = 500
    running_time = 300
    
    all_positions, all_velocities = Model.Run_System(running_time)
    
    
    """Create Video"""
    """
    fig = plt.figure(figsize=(10,10))
    camera = Camera(fig)
    #colors = cm.rainbow(np.linspace(0, 1, number_particles))
    
    for t in np.arange(1,running_time): #first one not to fit velocity bars
        plt.scatter(all_positions[:,0,t], all_positions[:,1,t], c='black')
        for n in np.arange(np.size(all_velocities[:,0,t])):
            plt.plot([all_positions[n,0,t], all_positions[n,0,t]-all_velocities[n,0,t-1]/(8*density)], 
                     [all_positions[n,1,t], all_positions[n,1,t]-all_velocities[n,1,t-1]/(8*density)], c='red')
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0,system_size])
        plt.ylim([0,system_size])
        camera.snap()
    
    anim = camera.animate(blit=True)
    filename = 'Vicsek_7_density'+str(density)+'_noise'+str(noise_value/2/np.pi)+'_step'+str(step_size)+'.mp4'
    print('done')
    anim.save(filename, fps=60, dpi=300)
    """
    fig, ax1 = plt.subplots(figsize=(12,12))
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.148, 0.71, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])
    #ax1.set_title('test') works fine
    camera = Camera(fig)
    #colors = cm.rainbow(np.linspace(0, 1, number_particles))
    circle_angles = np.linspace(0, 2*np.pi, 100)
    #circle = plt.Circle((0,0), radius=1, color='gray', alpha=0.5)
    
    for t in np.arange(1,running_time): #first one not to fit velocity bars
    
        polarization_magnitude, polarization_direction = Polarization(all_velocities[:,:,t-1])
        polarization_coords = polarization_magnitude * polarization_direction
        ax1.scatter(all_positions[:,0,t], all_positions[:,1,t], c='black', zorder=2) 
        for n in np.arange(np.size(all_velocities[:,0,t])):
            ax1.plot([all_positions[n,0,t], all_positions[n,0,t]-all_velocities[n,0,t-1]/(6*density)], 
                     [all_positions[n,1,t], all_positions[n,1,t]-all_velocities[n,1,t-1]/(6*density)], c='red', zorder=1)       
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlim([0,system_size])
        ax1.set_ylim([0,system_size])
    
        ax2.plot(np.cos(circle_angles), np.sin(circle_angles), c='black', zorder=3)
        #ax2.add_artist(circle)
        ax2.arrow(0, 0, 0.9*polarization_coords[0], 0.9*polarization_coords[1], width=0.05, head_width=0.2, head_length=0.1, color='black', zorder=4)
        ax2.scatter([0], [0], c='black', alpha=0.2, s=14200, zorder=0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim([-1.1,1.1])
        ax2.set_ylim([-1.1,1.1])
        ax2.axis('off')
        #ax2.patch.set_alpha(0.5)
        
        camera.snap()
    
    anim = camera.animate(blit=True)
    filename = 'Vicsek_8_density'+str(density)+'_noise'+str(noise_value/2/np.pi)+'_step'+str(step_size)+'.mp4'
    print('done')
    anim.save(filename, fps=30, dpi=300)
    
    print('Elapsed time: '+str((time.time()-tic)/60)+' minutes')
    

def Alpha_Plots():
    number_particles = 2000
    density = 5
    system_size = number_particles/density
    boxes_amount = 50
    min_boxsize = np.sqrt(0.001*system_size) #Boxes have a minimum size of 0.1% of the system size
    max_boxsize = np.sqrt(0.005*system_size) #Boxes have a maximum size of 10% of the system size
    boxes_size = np.linspace(min_boxsize, max_boxsize, boxes_amount)
    noise_values = np.arange(9)/10
    alpha_values = np.zeros(9)
    
    for index_1, noise_value in enumerate(noise_values):
        run_positions = Load_Data(number_particles, density, 1, 0.03, noise_value, 2000, 10000)
        mean_array = np.zeros(boxes_amount)
        std_array = np.zeros(boxes_amount)
        for index_2, size in enumerate(boxes_size):
            x_in_box = run_positions[:,0,:] < size
            y_in_box = run_positions[:,1,:] < size
            particles_in_box = x_in_box * y_in_box
            particles_in_box = np.sum(particles_in_box,0)
            mean_array[index_2] = np.mean(particles_in_box)
            std_array[index_2] = np.std(particles_in_box)           
        #determine alpha
        mean_array_log = np.log(mean_array)
        std_array_log = np.log(std_array)
        coeff = np.polyfit(mean_array_log, std_array_log, 1)
        alpha_values[index_1] = coeff[0]

        
        
        #plot std vs. mean
        plt.figure()
        plt.scatter(mean_array, std_array, marker='d', color='b', label='Data')
        x_plot = np.linspace(mean_array[0], mean_array[-1])
        plt.plot(x_plot, np.exp(coeff[1])*x_plot**coeff[0], color='r', label=r'Fit with $\alpha =$'+str(round(coeff[0],2)))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r'$\langle N \rangle$')
        plt.ylabel(r'$\Delta N$')
        plt.legend(loc='upper left')
        plt.title(r'Noise value $\eta =$'+str(noise_value))
        plt.tight_layout()
        plt.savefig(os.getcwd() + '\\Figures\\Alpha_fit_noise_'+str(noise_value)+'.png', dpi=300)
    
    #plot alpha vs. noise
    plt.figure()
    plt.scatter(noise_values, alpha_values, marker='d', color='purple')
    plt.plot(np.array([-0.05, 0.85]), np.array([1,1]), color='darkgoldenrod', ls='--')
    plt.plot(np.array([-0.05, 0.85]), np.array([0.5, 0.5]), color='darkgoldenrod', ls='--')
    plt.xlabel(r'Noise value $\eta$')
    plt.ylabel(r'Critical exponent $\alpha$')
    plt.xlim(-0.05, 0.85)
    plt.ylim(0.45, 1.05)
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\Figures\\Alpha_vs_noise.png', dpi=300)


