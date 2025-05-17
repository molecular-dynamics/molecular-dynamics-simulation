"""
Модуль для генерации начальных конфигураций кластеров.
Отвечает за создание кластеров с "магическими" числами частиц
и гексагональной структурой.
"""

import numpy as np

def create_hexagonal_cluster(num_particles, b=1.0):
    """
    Создает двумерный кластер с гексагональной структурой.
    """
    positions = np.zeros((num_particles, 2))
    
    idx = 0
    positions[idx] = [0, 0]
    idx += 1
    
    if num_particles <= 1:
        return positions
    
    shell = 1
    while idx < num_particles:
        for i in range(6*shell):
            if idx >= num_particles:
                break
                
            angle = 2 * np.pi * i / (6 * shell)
            
            positions[idx] = [
                shell * b * np.cos(angle),
                shell * b * np.sin(angle)
            ]
            idx += 1
        
        shell += 1
    
    return positions

def initialize_velocities(num_particles, temperature=0.01, seed=None):
    """
    Инициализирует скорости частиц для заданной температуры.
    """
    if seed is not None:
        np.random.seed(seed)
    
    velocities = np.random.normal(0, np.sqrt(temperature), (num_particles, 2))
    
    v_cm = np.mean(velocities, axis=0)
    velocities -= v_cm
    
    return velocities

def get_magic_number(shell_count):
    """
    Возвращает "магическое" число частиц для заданного числа оболочек.
    """
    return 1 + 3 * shell_count * (shell_count + 1)

def list_magic_numbers(max_count=10):
    """
    Выводит список "магических" чисел для кластеров.
    """
    return [get_magic_number(i) for i in range(max_count)]

if __name__ == "__main__":
    print("Магические числа частиц для гексагональных кластеров:")
    print(list_magic_numbers())
    
    positions = create_hexagonal_cluster(19)
    
    print("\nКоординаты частиц кластера из 19 частиц:")
    for i, pos in enumerate(positions):
        print(f"Частица {i+1}: {pos}")