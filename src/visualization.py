"""
Модуль визуализации результатов молекулярной динамики.
Содержит функции для создания графиков и анимаций.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

def plot_cluster(positions, bond_cutoff=1.5, title="Кластер"):
    """
    Строит статическое изображение кластера.
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', alpha=0.7)
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            pi = positions[i]
            pj = positions[j]
            dr = pi - pj
            r = np.linalg.norm(dr)
            
            if r < bond_cutoff:
                plt.plot([pi[0], pj[0]], [pi[1], pj[1]], 'k-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_cluster_animation(positions_history, bond_cutoff=1.5, interval=50, filename=None):
    """
    Создает анимацию движения частиц в кластере.
    """
    num_frames, num_particles, _ = positions_history.shape
    
    min_x = np.min(positions_history[:, :, 0]) - 1
    max_x = np.max(positions_history[:, :, 0]) + 1
    min_y = np.min(positions_history[:, :, 1]) - 1
    max_y = np.max(positions_history[:, :, 1]) + 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title("Динамика кластера")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    particles = ax.scatter([], [], s=100, c='blue', alpha=0.7)
    
    lines = []
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            line, = ax.plot([], [], 'k-', alpha=0.3)
            lines.append((i, j, line))
    
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        particles.set_offsets(np.empty((0, 2)))
        
        for i, j, line in lines:
            line.set_data([], [])
        
        text.set_text('')
        
        return [particles, text] + [line for _, _, line in lines]
    
    def update(frame):
        particles.set_offsets(positions_history[frame])
        
        for i, j, line in lines:
            pi = positions_history[frame, i]
            pj = positions_history[frame, j]
            r = np.linalg.norm(pi - pj)
            
            if r < bond_cutoff:
                line.set_data([pi[0], pj[0]], [pi[1], pj[1]])
                line.set_alpha(max(0, 1 - r/bond_cutoff))
            else:
                line.set_data([], [])
                line.set_alpha(0)
        
        text.set_text(f'Кадр: {frame}')
        
        return [particles, text] + [line for _, _, line in lines]
    
    ani = FuncAnimation(fig, update, frames=num_frames,
                        init_func=init, blit=True, interval=interval)
    
    if filename:
        ani.save(filename, writer='pillow', fps=30)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def plot_energy_temperature(energy, temperature, title="Зависимость температуры от энергии"):
    """
    Строит график зависимости температуры от энергии.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energy, temperature, 'r.-')
    plt.xlabel("Энергия на частицу E/N")
    plt.ylabel("Температура T")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heating_cooling_comparison(heating_energy, heating_temp, cooling_energy, cooling_temp):
    """
    Сравнивает процессы нагрева и охлаждения.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(heating_energy, heating_temp, 'r.-', label='Нагрев')
    plt.plot(cooling_energy, cooling_temp, 'b.-', label='Охлаждение')
    plt.xlabel("Энергия на частицу E/N")
    plt.ylabel("Температура T")
    plt.title("Гистерезис при фазовом переходе")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heat_capacity(energy, heat_capacity, transition_idx=None):
    """
    Строит график теплоемкости.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energy, heat_capacity, 'g.-')
    
    if transition_idx is not None:
        plt.axvline(x=energy[transition_idx], color='r', linestyle='--',
                   label=f'Фазовый переход при E/N = {energy[transition_idx]:.4f}')
    
    plt.xlabel("Энергия на частицу E/N")
    plt.ylabel("Теплоемкость C")
    plt.title("Теплоемкость системы")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bond_fluctuations(steps, fluctuations, threshold=0.1):
    """
    Строит график флуктуаций длины связи.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, fluctuations, 'm.-')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Порог плавления δ = {threshold}')
    plt.xlabel("Шаг")
    plt.ylabel("Флуктуация длины связи δ")
    plt.title("Флуктуации длины связи")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pair_correlation_function(r, g_r, temperatures=None):
    """
    Строит парную корреляционную функцию.
    """
    plt.figure(figsize=(10, 6))
    
    if temperatures is not None and len(temperatures) == len(g_r):
        cmap = LinearSegmentedColormap.from_list('temp', ['blue', 'green', 'red'])
        
        for i, (g, temp) in enumerate(zip(g_r, temperatures)):
            plt.plot(r, g, label=f'T = {temp:.2f}', color=cmap(i / (len(temperatures) - 1)))
    else:
        plt.plot(r, g_r, 'b-')
    
    plt.xlabel('Расстояние r')
    plt.ylabel('g(r)')
    plt.title('Парная корреляционная функция')
    
    if temperatures is not None:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_comprehensive_results(results, num_particles):
    """
    Создает комплексный набор графиков с результатами моделирования.
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.plot(results['heating_energy'], results['heating_temp'], 'r.-', label='Нагрев')
    plt.plot(results['cooling_energy'], results['cooling_temp'], 'b.-', label='Охлаждение')
    plt.xlabel('Энергия на частицу E/N')
    plt.ylabel('Температура T')
    plt.title('Зависимость T(E/N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(222)
    plt.plot(results['heating_energy'], results['heating_heat_capacity'], 'r.-', label='Нагрев')
    plt.plot(results['cooling_energy'], results['cooling_heat_capacity'], 'b.-', label='Охлаждение')
    plt.xlabel('Энергия на частицу E/N')
    plt.ylabel('Теплоемкость C = dE/dT')
    plt.title('Теплоемкость')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(223)
    plt.plot(range(len(results['bond_fluctuations'])), results['bond_fluctuations'], 'm.-')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Порог плавления')
    plt.xlabel('Шаг')
    plt.ylabel('Флуктуация длины связи δ')
    plt.title('Флуктуации длины связи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(224)
    steps = np.arange(len(results['potential_energy']))
    plt.plot(steps, results['potential_energy'], 'g.-')
    plt.xlabel('Шаг')
    plt.ylabel('Потенциальная энергия')
    plt.title('Потенциальная энергия системы')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Результаты моделирования кластера из {num_particles} частиц', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_size_dependence(cluster_sizes, melting_temperatures):
    """
    Строит график зависимости температуры плавления от размера кластера.
    """
    plt.figure(figsize=(10, 6))
    
    N = np.array(cluster_sizes)
    T = np.array(melting_temperatures)
    
    plt.subplot(121)
    plt.plot(N, T, 'bo-', label='Данные моделирования')
    plt.xlabel('Число частиц N')
    plt.ylabel('Температура плавления T')
    plt.title('T(N)')
    plt.grid(True)
    
    plt.subplot(122)
    N_power = N**(-1/3)
    
    plt.plot(N_power, T, 'ro-', label='Данные моделирования')
    
    coeffs = np.polyfit(N_power, T, 1)
    x_fit = np.linspace(min(N_power), max(N_power), 100)
    y_fit = coeffs[1] + coeffs[0] * x_fit
    
    plt.plot(x_fit, y_fit, 'k--', 
             label=f'T = {coeffs[1]:.4f} - {-coeffs[0]:.4f}/N^(1/3)')
    
    plt.xlabel('N^(-1/3)')
    plt.ylabel('Температура плавления T')
    plt.title('T(N^(-1/3))')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()