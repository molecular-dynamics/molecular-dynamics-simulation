"""
Основной скрипт для запуска моделирования плавления и затвердевания кластеров.
Использует функции из всех модулей для проведения полного исследования.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from cluster_generator import create_hexagonal_cluster, initialize_velocities, get_magic_number
from md_engine import MDEngine
from thermodynamics import calculate_temperature, calculate_bond_length_fluctuation, calculate_pair_correlation_function, calculate_heat_capacity
from phase_analyzer import detect_phase_transition, analyze_bond_fluctuations, analyze_hysteresis, detect_shell_melting, analyze_cluster_size_effect
from visualization import plot_cluster, create_cluster_animation, plot_energy_temperature, plot_heating_cooling_comparison, plot_heat_capacity, plot_bond_fluctuations, plot_pair_correlation_function, plot_comprehensive_results, plot_size_dependence

def run_simulation_for_cluster(num_particles, equilibration_steps=100, heating_steps=500, cooling_steps=500, dt=0.0005, heating_factor=1.002, cooling_factor=0.998, save_animation=False):
    """
    Запускает полное моделирование для одного кластера.
    """
    print(f"\n{'='*60}")
    print(f"Моделирование кластера из {num_particles} частиц")
    print(f"{'='*60}")
    
    epsilon = 1.0
    b = 1.0
    cutoff_radius = 2.5 * b
    
    print("Создание начальной конфигурации...")
    positions = create_hexagonal_cluster(num_particles, b)
    
    initial_temperature = 0.01
    velocities = initialize_velocities(num_particles, temperature=initial_temperature)
    
    plot_cluster(positions, title=f"Начальная конфигурация кластера из {num_particles} частиц")
    
    print("Инициализация движка молекулярной динамики...")
    md = MDEngine(positions, velocities, epsilon=epsilon, b=b, cutoff_radius=cutoff_radius)
    
    print("Уравновешивание системы...")
    eq_results = md.run_simulation(equilibration_steps, dt, scale_factor=1.0)
    
    positions_after_equilibration = md.positions.copy()
    velocities_after_equilibration = md.velocities.copy()
    
    print("Нагрев системы...")
    heating_results = md.run_simulation(heating_steps, dt, scale_factor=heating_factor, scale_interval=5)
    
    print("Охлаждение системы...")
    cooling_results = md.run_simulation(cooling_steps, dt, scale_factor=cooling_factor, scale_interval=5)
    
    total_steps = heating_steps + cooling_steps
    positions_history = np.vstack([heating_results['positions'], cooling_results['positions']])
    velocities_history = np.vstack([heating_results['velocities'], cooling_results['velocities']])
    energy_history = np.concatenate([heating_results['energy'], cooling_results['energy']])
    kinetic_history = np.concatenate([heating_results['kinetic'], cooling_results['kinetic']])
    potential_history = np.concatenate([heating_results['potential'], cooling_results['potential']])
    
    energy_per_particle = energy_history
    
    print("Расчет температур и других характеристик...")
    temperatures = np.zeros(total_steps)
    bond_distances = []
    
    for i in range(total_steps):
        temperatures[i] = calculate_temperature(velocities_history[i], np.ones(num_particles))
        bond_distances.append(calculate_bond_length_fluctuation(positions_history[i], bond_cutoff=1.5*b))
    
    heating_energy_per_particle = energy_per_particle[:heating_steps]
    heating_temp = temperatures[:heating_steps]
    
    cooling_energy_per_particle = energy_per_particle[heating_steps:]
    cooling_temp = temperatures[heating_steps:]
    
    if len(heating_temp) > 10:
        heating_energy_filtered = heating_energy_per_particle
        heating_temp_filtered = heating_temp
        
        temp_max = np.percentile(heating_temp, 97)
        temp_min = np.percentile(heating_temp, 3)
        mask = (heating_temp <= temp_max) & (heating_temp >= temp_min)
        
        if np.sum(mask) > 0.5 * len(heating_temp):
            heating_energy_filtered = heating_energy_per_particle[mask]
            heating_temp_filtered = heating_temp[mask]
    else:
        heating_energy_filtered = heating_energy_per_particle
        heating_temp_filtered = heating_temp
    
    if len(cooling_temp) > 10:
        cooling_energy_filtered = cooling_energy_per_particle
        cooling_temp_filtered = cooling_temp
        
        temp_max = np.percentile(cooling_temp, 97)
        temp_min = np.percentile(cooling_temp, 3)
        mask = (cooling_temp <= temp_max) & (cooling_temp >= temp_min)
        
        if np.sum(mask) > 0.5 * len(cooling_temp):
            cooling_energy_filtered = cooling_energy_per_particle[mask]
            cooling_temp_filtered = cooling_temp[mask]
    else:
        cooling_energy_filtered = cooling_energy_per_particle
        cooling_temp_filtered = cooling_temp
    
    heating_heat_capacity = calculate_heat_capacity(heating_energy_filtered, heating_temp_filtered)
    cooling_heat_capacity = calculate_heat_capacity(cooling_energy_filtered, cooling_temp_filtered)
    
    bond_fluctuations = np.zeros(total_steps)
    for i in range(total_steps):
        if bond_distances[i]:
            r_mean = np.mean(bond_distances[i])
            r2_mean = np.mean(np.array(bond_distances[i])**2)
            bond_fluctuations[i] = np.sqrt(r2_mean - r_mean**2) / r_mean
    
    print("Анализ фазовых переходов...")
    heating_transition = detect_phase_transition(heating_energy_filtered, heating_temp_filtered)
    cooling_transition = detect_phase_transition(cooling_energy_filtered, cooling_temp_filtered)
    
    fluctuation_analysis = analyze_bond_fluctuations(bond_distances, temperatures)
    
    hysteresis_results, energy_range, heating_interp, cooling_interp, temp_diff = analyze_hysteresis(
        heating_energy_filtered, heating_temp_filtered, cooling_energy_filtered, cooling_temp_filtered)
    
    num_shells = 1
    while get_magic_number(num_shells) <= num_particles:
        num_shells += 1
    num_shells -= 1
    
    shell_melting = detect_shell_melting(positions_history, velocities_history, num_shells)
    
    print("Расчет корреляционных функций...")
    g_r_data = []
    temp_samples = []
    
    indices = np.linspace(0, heating_steps-1, 5, dtype=int)
    
    for idx in indices:
        if idx < len(positions_history):
            r, g_r = calculate_pair_correlation_function(positions_history[idx])
            g_r_data.append(g_r)
            temp_samples.append(temperatures[idx])
    
    results = {
        'num_particles': num_particles,
        'heating_energy': heating_energy_filtered,
        'heating_temp': heating_temp_filtered,
        'cooling_energy': cooling_energy_filtered,
        'cooling_temp': cooling_temp_filtered,
        'heating_heat_capacity': heating_heat_capacity,
        'cooling_heat_capacity': cooling_heat_capacity,
        'bond_fluctuations': bond_fluctuations,
        'heating_transition': heating_transition,
        'cooling_transition': cooling_transition,
        'hysteresis': hysteresis_results,
        'shell_melting': shell_melting,
        'g_r_data': g_r_data,
        'temp_samples': temp_samples,
        'positions_history': positions_history,
        'velocities_history': velocities_history,
        'energy_history': energy_history,
        'kinetic_history': kinetic_history,
        'potential_energy': potential_history
    }
    
    print("Визуализация результатов...")
    plot_heating_cooling_comparison(heating_energy_filtered, heating_temp_filtered, 
                                   cooling_energy_filtered, cooling_temp_filtered)
    
    plot_heat_capacity(heating_energy_filtered, heating_heat_capacity, 
                      heating_transition['transition_index'] if 'transition_index' in heating_transition else None)
    
    plot_bond_fluctuations(range(len(bond_fluctuations)), bond_fluctuations)
    
    if len(g_r_data) > 0:
        plot_pair_correlation_function(r, g_r_data, temp_samples)
    
    if save_animation:
        print("Создание анимации...")
        create_cluster_animation(positions_history[::10], bond_cutoff=1.5*b, 
                                interval=50, filename=f"cluster_{num_particles}_animation.gif")
    
    print("\nРезультаты анализа фазовых переходов:")
    if 'transition_temperature' in heating_transition:
        print(f"Температура плавления: {heating_transition['transition_temperature']:.4f} при E/N = {heating_transition['transition_energy']:.4f}")
    
    if 'transition_temperature' in cooling_transition:
        print(f"Температура затвердевания: {cooling_transition['transition_temperature']:.4f} при E/N = {cooling_transition['transition_energy']:.4f}")
    
    if 'melting_temperature' in fluctuation_analysis:
        print(f"Температура плавления по флуктуациям длины связи: {fluctuation_analysis['melting_temperature']:.4f}")
    
    if hysteresis_results:
        print(f"Величина гистерезиса: {hysteresis_results.get('hysteresis_magnitude', 0.0):.4f} при E/N = {hysteresis_results.get('hysteresis_energy', 0.0):.4f}")
    
    if shell_melting.get('is_shell_melting', False):
        print("Обнаружено оболочечное плавление.")
    
    return results

def main():
    """
    Основная функция запуска проекта.
    """
    print("Исследование плавления и затвердевания малых кластеров")
    print("======================================================")
    
    equilibration_steps = 100
    heating_steps = 500   
    cooling_steps = 500   
    dt = 0.0005
    heating_factor = 1.002
    cooling_factor = 0.998
    
    cluster_sizes = [7, 19, 37]
    
    all_results = {}
    melting_temperatures = []
    cluster_sizes_with_data = []
    
    for size in cluster_sizes:
        results = run_simulation_for_cluster(
            size, 
            equilibration_steps=equilibration_steps,
            heating_steps=heating_steps,
            cooling_steps=cooling_steps,
            dt=dt,
            heating_factor=heating_factor,
            cooling_factor=cooling_factor,
            save_animation=(size == 19)
        )
        
        all_results[size] = results
        
        if 'heating_transition' in results and 'transition_temperature' in results['heating_transition']:
            melting_temperatures.append(results['heating_transition']['transition_temperature'])
            cluster_sizes_with_data.append(size)
    
    if len(melting_temperatures) >= 2 and len(cluster_sizes_with_data) >= 2:
        print("\nАнализ зависимости температуры плавления от размера кластера:")
        size_analysis = analyze_cluster_size_effect(cluster_sizes_with_data, melting_temperatures)
        
        print(f"Модель зависимости: {size_analysis['model']}")
        print(f"Экстраполированная температура плавления для объемного материала: {size_analysis['T_bulk']:.4f}")
        
        plot_size_dependence(cluster_sizes_with_data, melting_temperatures)
    
    print("\nИсследование завершено!")

if __name__ == "__main__":
    main()