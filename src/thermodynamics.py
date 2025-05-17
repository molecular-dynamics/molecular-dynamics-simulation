"""
Модуль для расчета термодинамических параметров системы.
Реализует функции для вычисления температуры, давления,
и других термодинамических величин.
"""

import numpy as np

def calculate_temperature(velocities, masses=None):
    """
    Рассчитывает температуру системы.
    T = 2/(2N-3)k * ∑(m_i*(v_i-v_cm)^2/2)
    """
    num_particles = len(velocities)
    
    if masses is None:
        masses = np.ones(num_particles)
    
    v_cm = np.sum(masses.reshape(-1, 1) * velocities, axis=0) / np.sum(masses)
    
    rel_velocities = velocities - v_cm
    
    rel_kinetic_energy = 0.5 * np.sum(masses.reshape(-1, 1) * rel_velocities**2)
    
    dof = max(1, 2 * num_particles - 3)
    temperature = 2 * rel_kinetic_energy / dof
    
    max_temp = 100.0
    return min(temperature, max_temp)

def calculate_bond_length_fluctuation(positions, bond_cutoff=1.5):
    """
    Рассчитывает среднеквадратичную флуктуацию длины связи.
    δ = √[2/N(N-1) * ∑_i<j(<r_ij^2> - <r_ij>^2)/<r_ij>]
    """
    num_particles = len(positions)
    pair_distances = []
    
    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            dr = positions[j] - positions[i]
            r = np.linalg.norm(dr)
            
            if r < bond_cutoff:
                pair_distances.append(r)
    
    return pair_distances

def calculate_bond_fluctuation_from_time_series(bond_distances):
    """
    Рассчитывает среднеквадратичную флуктуацию длины связи из временного ряда.
    """
    all_distances = []
    for step_distances in bond_distances:
        all_distances.extend(step_distances)
    
    if not all_distances:
        return 0.0
    
    r_mean = np.mean(all_distances)
    r2_mean = np.mean(np.array(all_distances)**2)
    
    delta = np.sqrt(r2_mean - r_mean**2) / r_mean
    
    return delta

def calculate_pair_correlation_function(positions, box_size=None, num_bins=50, rmax=None):
    """
    Рассчитывает парную корреляционную функцию g(r).
    """
    num_particles = len(positions)
    
    if num_particles < 2:
        r_dummy = np.linspace(0, 5.0, num_bins)
        return r_dummy, np.zeros_like(r_dummy)
    
    if rmax is None:
        pos_range = np.max(positions, axis=0) - np.min(positions, axis=0)
        rmax = np.max(pos_range) * 0.5
        rmax = max(rmax, 5.0)
    
    r_bins = np.linspace(0, rmax, num_bins + 1)
    r = 0.5 * (r_bins[1:] + r_bins[:-1])
    histogram = np.zeros(num_bins)
    
    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            dr = positions[j] - positions[i]
            
            if box_size is not None:
                dr = dr - box_size * np.round(dr / box_size)
                
            distance = np.linalg.norm(dr)
            
            if distance < rmax:
                bin_idx = int(distance / rmax * num_bins)
                if bin_idx < num_bins:
                    histogram[bin_idx] += 2
    
    bin_volume = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    
    if box_size is None:
        area = np.pi * (rmax/2)**2
    else:
        area = box_size**2
    
    density = num_particles / area
    
    bin_volume = np.maximum(bin_volume, 1e-10)
    
    g_r = histogram / bin_volume / (num_particles * density)
    
    g_r = np.minimum(g_r, 10.0)
    
    return r, g_r

def calculate_heat_capacity(energies, temperatures):
    """
    Рассчитывает теплоемкость системы как dE/dT.
    """
    if len(energies) < 5 or len(temperatures) < 5:
        return np.zeros_like(energies)
    
    sorted_indices = np.argsort(temperatures)
    sorted_temp = temperatures[sorted_indices]
    sorted_energy = energies[sorted_indices]
    
    window_size = min(5, len(energies) // 5)
    
    smoothed_energies = np.zeros_like(sorted_energy)
    smoothed_temperatures = np.zeros_like(sorted_temp)
    
    for i in range(len(sorted_energy)):
        start = max(0, i - window_size // 2)
        end = min(len(sorted_energy), i + window_size // 2 + 1)
        smoothed_energies[i] = np.mean(sorted_energy[start:end])
        smoothed_temperatures[i] = np.mean(sorted_temp[start:end])
    
    heat_capacity = np.zeros_like(smoothed_energies)
    
    for i in range(1, len(smoothed_energies) - 1):
        dE = smoothed_energies[i+1] - smoothed_energies[i-1]
        dT = smoothed_temperatures[i+1] - smoothed_temperatures[i-1]
        
        if abs(dT) > 1e-6:
            heat_capacity[i] = dE / dT
    
    valid_hc = heat_capacity[heat_capacity != 0]
    if len(valid_hc) > 0:
        max_hc = np.percentile(valid_hc, 95)
        min_hc = np.percentile(valid_hc, 5)
        heat_capacity = np.clip(heat_capacity, min_hc, max_hc)
    
    reordered_heat_capacity = np.zeros_like(energies)
    reordered_heat_capacity[sorted_indices] = heat_capacity
    
    return reordered_heat_capacity