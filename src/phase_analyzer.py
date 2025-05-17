"""
Модуль для анализа фазовых переходов в кластерах.
Реализует методы для определения плавления и затвердевания,
а также для идентификации оболочечного плавления.
"""

import numpy as np
from scipy.signal import savgol_filter
from thermodynamics import calculate_heat_capacity, calculate_bond_fluctuation_from_time_series

def detect_phase_transition(energies, temperatures):
    """
    Обнаруживает фазовый переход по зависимости температуры от энергии.
    
    Фазовый переход определяется по пику теплоемкости. Для вычисления
    теплоемкости используется производная dE/dT.
    """
    results = {}
    
    if len(energies) < 10 or len(temperatures) < 10:
        return results
    
    e_max = np.percentile(energies, 99)
    e_min = np.percentile(energies, 1)
    t_max = np.percentile(temperatures, 99)
    t_min = np.percentile(temperatures, 1)
    
    valid_indices = (energies >= e_min) & (energies <= e_max) & (temperatures >= t_min) & (temperatures <= t_max)
    
    if np.sum(valid_indices) < 10:
        return results
        
    filtered_energies = energies[valid_indices]
    filtered_temperatures = temperatures[valid_indices]
    
    try:
        window_size = min(51, len(filtered_energies) // 5)
        if window_size % 2 == 0:
            window_size += 1
        poly_order = min(3, window_size - 1)
        
        smoothed_energy = savgol_filter(filtered_energies, window_size, poly_order)
        smoothed_temperature = savgol_filter(filtered_temperatures, window_size, poly_order)
    except Exception as e:
        print(f"Предупреждение при сглаживании данных: {e}")
        smoothed_energy = filtered_energies.copy()
        smoothed_temperature = filtered_temperatures.copy()
    
    heat_capacity = calculate_heat_capacity(smoothed_energy, smoothed_temperature)
    
    heat_capacity = np.clip(heat_capacity, -10.0, 10.0)
    
    start_idx = window_size // 2 if window_size > 1 else 0
    end_idx = len(heat_capacity) - start_idx if window_size > 1 else len(heat_capacity)
    
    if start_idx >= end_idx or end_idx <= 0:
        return results
    
    valid_heat_capacity = heat_capacity[start_idx:end_idx]
    
    if len(valid_heat_capacity) == 0:
        return results
    
    try:
        valid_indices = np.isfinite(valid_heat_capacity)
        if not np.any(valid_indices):
            return results
        
        max_c_idx = start_idx + np.argmax(valid_heat_capacity[valid_indices])
        
        if max_c_idx < len(smoothed_temperature) and max_c_idx < len(smoothed_energy):
            results['transition_index'] = max_c_idx
            results['transition_temperature'] = smoothed_temperature[max_c_idx]
            results['transition_energy'] = smoothed_energy[max_c_idx]
            results['max_heat_capacity'] = heat_capacity[max_c_idx]
            results['heat_capacity'] = heat_capacity
    except Exception as e:
        print(f"Ошибка при поиске пика теплоемкости: {e}")
    
    return results

def analyze_bond_fluctuations(bond_distances_history, temperatures):
    """
    Анализирует флуктуации длины связи для определения фазового перехода.
    
    При фазовом переходе флуктуации длины связи резко возрастают.
    Критерием плавления считается превышение порогового значения
    флуктуаций (Линдеманна).
    """
    results = {}
    
    if not bond_distances_history or len(temperatures) == 0:
        return results
    
    fluctuations = np.zeros(len(bond_distances_history))
    
    for i, distances in enumerate(bond_distances_history):
        if distances:
            r_mean = np.mean(distances)
            r2_mean = np.mean(np.array(distances)**2)
            fluctuations[i] = np.sqrt(r2_mean - r_mean**2) / r_mean
    
    fluctuations = np.clip(fluctuations, 0.0, 1.0)
    
    results['fluctuations'] = fluctuations
    
    threshold = 0.1
    
    valid_indices = np.isfinite(fluctuations)
    if np.any(valid_indices):
        melting_indices = np.where((fluctuations > threshold) & valid_indices)[0]
        if len(melting_indices) > 0:
            first_melting_idx = melting_indices[0]
            if first_melting_idx < len(temperatures):
                results['melting_index'] = first_melting_idx
                results['melting_temperature'] = temperatures[first_melting_idx]
                results['is_molten'] = True
        else:
            results['is_molten'] = False
    else:
        results['is_molten'] = False
    
    return results

def analyze_hysteresis(heating_energy, heating_temp, cooling_energy, cooling_temp):
    """
    Анализирует гистерезис между нагревом и охлаждением.
    
    Гистерезис является одним из признаков фазового перехода первого рода.
    Температура плавления часто выше температуры затвердевания при 
    одинаковой энергии, что связано с метастабильными состояниями.
    """
    results = {}
    
    if (len(heating_energy) < 3 or len(heating_temp) < 3 or 
        len(cooling_energy) < 3 or len(cooling_temp) < 3):
        print("Недостаточно данных для анализа гистерезиса")
        return results, None, None, None, None
    
    h_e_max = np.percentile(heating_energy, 99)
    h_e_min = np.percentile(heating_energy, 1)
    h_t_max = np.percentile(heating_temp, 99)
    h_t_min = np.percentile(heating_temp, 1)
    
    c_e_max = np.percentile(cooling_energy, 99)
    c_e_min = np.percentile(cooling_energy, 1)
    c_t_max = np.percentile(cooling_temp, 99)
    c_t_min = np.percentile(cooling_temp, 1)
    
    h_valid = (heating_energy >= h_e_min) & (heating_energy <= h_e_max) & (heating_temp >= h_t_min) & (heating_temp <= h_t_max)
    c_valid = (cooling_energy >= c_e_min) & (cooling_energy <= c_e_max) & (cooling_temp >= c_t_min) & (cooling_temp <= c_t_max)
    
    if np.sum(h_valid) < 3 or np.sum(c_valid) < 3:
        print("Недостаточно валидных данных после фильтрации")
        return results, None, None, None, None
    
    h_energy_filtered = heating_energy[h_valid]
    h_temp_filtered = heating_temp[h_valid]
    c_energy_filtered = cooling_energy[c_valid]
    c_temp_filtered = cooling_temp[c_valid]
    
    try:
        min_energy = max(np.min(h_energy_filtered), np.min(c_energy_filtered))
        max_energy = min(np.max(h_energy_filtered), np.max(c_energy_filtered))
        
        if max_energy <= min_energy:
            print("Недостаточно перекрытия между данными нагрева и охлаждения")
            return results, None, None, None, None
        
        e_range = np.linspace(min_energy + 0.1*(max_energy-min_energy), 
                             max_energy - 0.1*(max_energy-min_energy), 50)
        
        h_sort_idx = np.argsort(h_energy_filtered)
        c_sort_idx = np.argsort(c_energy_filtered)
        
        h_energy_sorted = h_energy_filtered[h_sort_idx]
        h_temp_sorted = h_temp_filtered[h_sort_idx]
        
        c_energy_sorted = c_energy_filtered[c_sort_idx]
        c_temp_sorted = c_temp_filtered[c_sort_idx]
        
        heating_interp = np.interp(e_range, h_energy_sorted, h_temp_sorted)
        cooling_interp = np.interp(e_range, c_energy_sorted, c_temp_sorted)
        
        temperature_diff = heating_interp - cooling_interp
        
        max_diff = min(5.0, np.percentile(np.abs(temperature_diff), 95))
        temperature_diff = np.clip(temperature_diff, -max_diff, max_diff)
        
        max_diff_idx = np.argmax(np.abs(temperature_diff))
        
        if max_diff_idx < len(e_range):
            results['hysteresis_energy'] = e_range[max_diff_idx]
            results['hysteresis_magnitude'] = temperature_diff[max_diff_idx]
            
            if np.abs(heating_interp[max_diff_idx]) > 1e-6:
                results['hysteresis_relative'] = temperature_diff[max_diff_idx] / heating_interp[max_diff_idx]
            else:
                results['hysteresis_relative'] = 0.0
            
            results['average_hysteresis'] = np.mean(np.abs(temperature_diff))
        
        return results, e_range, heating_interp, cooling_interp, temperature_diff
    
    except Exception as e:
        print(f"Ошибка при анализе гистерезиса: {e}")
        return results, None, None, None, None

def detect_shell_melting(positions_history, velocities_history, num_shells):
    """
    Обнаруживает оболочечное плавление, анализируя кинетическую энергию
    частиц в разных слоях кластера.
    
    Оболочечное плавление характерно для нанокластеров, когда внешние
    слои плавятся при более низкой температуре, чем внутренние.
    """
    results = {}
    
    if (len(positions_history) < 2 or len(velocities_history) < 2 or 
        num_shells < 1 or positions_history.shape[1] < 3):
        results['is_shell_melting'] = False
        return results
    
    try:
        num_frames = len(positions_history)
        
        shells = identify_shells(positions_history[0], num_shells)
        
        shell_energies = np.zeros((num_frames, num_shells))
        
        for frame in range(num_frames):
            for shell_idx in range(num_shells):
                shell_particles = shells[shell_idx]
                
                if len(shell_particles) > 0:
                    velocities = velocities_history[frame, shell_particles]
                    shell_energies[frame, shell_idx] = 0.5 * np.mean(np.sum(velocities**2, axis=1))
        
        shell_energies = np.clip(shell_energies, 0.0, 10.0)
        
        results['shell_energies'] = shell_energies
        results['shells'] = shells
        
        shell_melting_frames = np.zeros(num_shells, dtype=int)
        
        for shell_idx in range(num_shells):
            if len(shells[shell_idx]) == 0:
                continue
                
            energy_changes = np.gradient(shell_energies[:, shell_idx])
            
            threshold = np.std(energy_changes) * 2
            significant_changes = np.where(energy_changes > threshold)[0]
            
            if len(significant_changes) > 0:
                shell_melting_frames[shell_idx] = significant_changes[0]
        
        results['shell_melting_frames'] = shell_melting_frames
        
        is_shell_melting = False
        
        valid_shells = []
        for i in range(num_shells):
            if shell_melting_frames[i] > 0:
                valid_shells.append(i)
                
        if len(valid_shells) >= 2:
            ordered_pairs = 0
            total_pairs = 0
            
            for i in range(len(valid_shells)-1):
                outer_shell = valid_shells[i+1]
                inner_shell = valid_shells[i]
                
                if outer_shell > inner_shell:
                    total_pairs += 1
                    if shell_melting_frames[outer_shell] <= shell_melting_frames[inner_shell]:
                        ordered_pairs += 1
            
            if total_pairs > 0 and ordered_pairs / total_pairs >= 0.5:
                is_shell_melting = True
        
        results['is_shell_melting'] = is_shell_melting
    
    except Exception as e:
        print(f"Ошибка при определении оболочечного плавления: {e}")
        results['is_shell_melting'] = False
    
    return results

def identify_shells(positions, num_shells):
    """
    Идентифицирует частицы, принадлежащие разным оболочкам кластера.
    
    Оболочки определяются по расстоянию от центра масс кластера.
    """
    try:
        center = np.mean(positions, axis=0)
        
        distances = np.linalg.norm(positions - center, axis=1)
        
        sorted_indices = np.argsort(distances)
        
        shells = []
        
        shell_sizes = [1]
        for i in range(1, num_shells):
            shell_sizes.append(6 * i)
        
        total_particles = sum(shell_sizes[:num_shells])
        if total_particles > len(positions):
            actual_num_shells = 1
            total = 1
            for i in range(1, num_shells):
                total += 6 * i
                if total <= len(positions):
                    actual_num_shells += 1
                else:
                    break
            
            shell_sizes = shell_sizes[:actual_num_shells]
            if len(shell_sizes) < num_shells:
                shell_sizes[-1] += len(positions) - sum(shell_sizes)
        
        start_idx = 0
        for shell_size in shell_sizes:
            end_idx = start_idx + shell_size
            if end_idx > len(positions):
                end_idx = len(positions)
            
            shells.append(sorted_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
            
            if end_idx >= len(positions):
                break
        
        while len(shells) < num_shells:
            shells.append([])
        
        return shells
    
    except Exception as e:
        print(f"Ошибка при определении оболочек: {e}")
        return [[] for _ in range(num_shells)]

def analyze_cluster_size_effect(cluster_sizes, melting_temps):
    """
    Анализирует влияние размера кластера на температуру плавления.
    
    Температура плавления нанокластеров зависит от их размера:
    T_melt = T_bulk - c/N^(1/3), где T_bulk - температура плавления
    объёмного материала, N - число частиц, c - константа.
    """
    results = {}
    
    if len(cluster_sizes) < 2 or len(melting_temps) < 2:
        results['model'] = "Недостаточно данных для анализа"
        results['T_bulk'] = 0.0
        results['c'] = 0.0
        return results
    
    try:
        N = np.array(cluster_sizes)
        T = np.array(melting_temps)
        
        valid_indices = np.isfinite(T) & (T > 0) & (T < 100.0)
        if np.sum(valid_indices) < 2:
            results['model'] = "Недостаточно валидных данных для анализа"
            results['T_bulk'] = 0.0
            results['c'] = 0.0
            return results
            
        N = N[valid_indices]
        T = T[valid_indices]
        
        N_power = N**(-1/3)
        
        A = np.vstack([N_power, np.ones(len(N_power))]).T
        m, c = np.linalg.lstsq(A, T, rcond=None)[0]
        
        T_bulk = c
        c_const = -m
        
        results['T_bulk'] = T_bulk
        results['c'] = c_const
        results['model'] = f"T_melt = {T_bulk:.4f} - {c_const:.4f}/N^(1/3)"
        
        T_predicted = T_bulk - c_const * N_power
        
        mse = np.mean((T - T_predicted)**2)
        results['mse'] = mse
    
    except Exception as e:
        print(f"Ошибка при анализе размерной зависимости: {e}")
        results['model'] = "Ошибка анализа"
        results['T_bulk'] = 0.0
        results['c'] = 0.0
    
    return results