"""
Основной модуль молекулярной динамики.
Реализует интегрирование уравнений движения по алгоритму Верле
и потенциал Леннард-Джонса для взаимодействия частиц.
"""

import numpy as np
from tqdm import tqdm

class MDEngine:
    """
    Класс для моделирования молекулярной динамики с использованием
    потенциала Леннард-Джонса и алгоритма Верле.
    """
    
    def __init__(self, positions, velocities, masses=None, epsilon=1.0, b=1.0, cutoff_radius=2.5):
        """
        Инициализация системы молекулярной динамики.
        """
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.num_particles = len(positions)
        
        if masses is None:
            self.masses = np.ones(self.num_particles)
        else:
            self.masses = masses.copy()
        
        self.epsilon = epsilon
        self.b = b
        self.rc = cutoff_radius * b
        
        self.accelerations = np.zeros_like(positions)
        
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.total_energy = 0.0
        
        self.min_distance = 0.2 * b
        
        self.max_force = 50.0 * epsilon
        
        self.calculate_forces()
        
    def lennard_jones_potential(self, r):
        """
        Вычисляет потенциальную энергию Леннард-Джонса.
        U_LJ(r) = ε[(b/r)^12 - 2(b/r)^6]
        
        Потенциал обрезается на расстоянии rc и смещается для обеспечения
        непрерывности на границе обрезания.
        
        При r < min_distance используется смягченный потенциал для предотвращения
        численной нестабильности.
        """
        if r < self.min_distance:
            r_scaled = self.min_distance
            U = self.epsilon * ((self.b/r_scaled)**12 - 2 * (self.b/r_scaled)**6)
            penalty_factor = 5.0
            penalty = penalty_factor * self.epsilon * (1.0 - r/self.min_distance)
            return min(U + penalty, 100.0 * self.epsilon)
        
        if r > self.rc:
            return 0.0
        
        U = self.epsilon * ((self.b/r)**12 - 2 * (self.b/r)**6)
        
        U_rc = self.epsilon * ((self.b/self.rc)**12 - 2 * (self.b/self.rc)**6)
        
        return U - U_rc
    
    def lennard_jones_force(self, r, dr):
        """
        Вычисляет силу взаимодействия Леннард-Джонса.
        F = -dU/dr = 12ε[(b/r)^12 - (b/r)^6] / r^2 * r_vec
        
        При r < min_distance используется смягченная сила для предотвращения
        численной нестабильности.
        """
        if r < self.min_distance:
            r_scaled = self.min_distance
            force_direction = dr / (r + 1e-10)
            f_scalar = 5.0 * self.epsilon * (1.0 - r/self.min_distance) / r_scaled
            f_scalar = min(f_scalar, self.max_force)
            return f_scalar * force_direction
        
        if r > self.rc:
            return np.zeros(2)
        
        f_scalar = 12 * self.epsilon * ((self.b/r)**12 - (self.b/r)**6) / r**2
        
        f_scalar = np.clip(f_scalar, -self.max_force, self.max_force)
        
        return f_scalar * dr
    
    def calculate_forces(self):
        """
        Рассчитывает силы взаимодействия между всеми частицами
        и обновляет ускорения. Также вычисляет потенциальную энергию.
        
        Силы рассчитываются с учетом третьего закона Ньютона (принцип действия и противодействия).
        """
        self.accelerations = np.zeros((self.num_particles, 2))
        total_potential = 0.0
        
        for i in range(self.num_particles - 1):
            for j in range(i + 1, self.num_particles):
                dr = self.positions[j] - self.positions[i]
                
                r = np.linalg.norm(dr)
                
                r = max(r, self.min_distance)
                
                force = self.lennard_jones_force(r, dr)
                
                self.accelerations[i] += force / self.masses[i]
                self.accelerations[j] -= force / self.masses[j]
                
                pot_energy = self.lennard_jones_potential(r)
                pot_energy = np.clip(pot_energy, -100.0 * self.epsilon, 100.0 * self.epsilon)
                total_potential += pot_energy
        
        self.potential_energy = total_potential / self.num_particles
        
        max_acc = 20.0
        acc_norm = np.linalg.norm(self.accelerations, axis=1)
        for i in range(self.num_particles):
            if acc_norm[i] > max_acc:
                self.accelerations[i] = self.accelerations[i] * max_acc / acc_norm[i]
        
        self.calculate_kinetic_energy()
        
        self.total_energy = self.kinetic_energy + self.potential_energy
    
    def calculate_kinetic_energy(self):
        """
        Рассчитывает кинетическую энергию системы.
        K = Σ(1/2 * m_i * v_i^2)
        """
        total_kinetic = 0.5 * np.sum(self.masses.reshape(-1, 1) * self.velocities**2)
        
        self.kinetic_energy = total_kinetic / self.num_particles
    
    def verlet_step(self, dt):
        """
        Выполняет один шаг интегрирования по алгоритму Верле в скоростной форме.
        
        Алгоритм Верле имеет третий порядок точности и хорошо сохраняет энергию
        при постоянном шаге по времени.
        """
        half_velocities = self.velocities + 0.5 * self.accelerations * dt
        
        max_vel = 3.0
        vel_norm = np.linalg.norm(half_velocities, axis=1)
        for i in range(self.num_particles):
            if vel_norm[i] > max_vel:
                half_velocities[i] = half_velocities[i] * max_vel / vel_norm[i]
        
        self.positions += half_velocities * dt
        
        old_accelerations = self.accelerations.copy()
        self.calculate_forces()
        
        self.velocities = half_velocities + 0.5 * self.accelerations * dt
        
        vel_norm = np.linalg.norm(self.velocities, axis=1)
        for i in range(self.num_particles):
            if vel_norm[i] > max_vel:
                self.velocities[i] = self.velocities[i] * max_vel / vel_norm[i]
        
        self.calculate_kinetic_energy()
        self.total_energy = self.kinetic_energy + self.potential_energy
    
    def run_simulation(self, num_steps, dt, scale_factor=1.0, scale_interval=10):
        """
        Запускает симуляцию на заданное количество шагов.
        """
        positions_history = np.zeros((num_steps, self.num_particles, 2))
        velocities_history = np.zeros((num_steps, self.num_particles, 2))
        energy_history = np.zeros(num_steps)
        kinetic_history = np.zeros(num_steps)
        potential_history = np.zeros(num_steps)
        
        initial_energy = self.total_energy
        
        for step in tqdm(range(num_steps), desc="Симуляция"):
            positions_history[step] = self.positions.copy()
            velocities_history[step] = self.velocities.copy()
            
            energy_history[step] = np.clip(self.total_energy, -100.0, 100.0)
            kinetic_history[step] = np.clip(self.kinetic_energy, 0.0, 100.0)
            potential_history[step] = np.clip(self.potential_energy, -100.0, 100.0)
            
            self.verlet_step(dt)
            
            if step % scale_interval == 0 and scale_factor != 1.0:
                self.velocities *= scale_factor
                self.calculate_kinetic_energy()
                self.total_energy = self.kinetic_energy + self.potential_energy
            
            if scale_factor == 1.0 and step > 0:
                energy_deviation = abs(self.total_energy - initial_energy) / max(1e-10, abs(initial_energy))
                if energy_deviation > 0.05:
                    print(f"Предупреждение: Отклонение энергии {energy_deviation*100:.2f}% на шаге {step}")
        
        results = {
            'positions': positions_history,
            'velocities': velocities_history,
            'energy': energy_history,
            'kinetic': kinetic_history,
            'potential': potential_history
        }
        
        return results
        
    def scale_velocities(self, factor):
        """
        Масштабирует скорости частиц.
        """
        self.velocities *= factor
        self.calculate_kinetic_energy()
        self.total_energy = self.kinetic_energy + self.potential_energy
    
    def reset_velocities(self, target_temperature):
        """
        Сбрасывает скорости частиц для достижения заданной температуры.
        """
        self.velocities = np.random.normal(0, 1, (self.num_particles, 2))
        
        v_cm = np.sum(self.masses.reshape(-1, 1) * self.velocities, axis=0) / np.sum(self.masses)
        self.velocities -= v_cm
        
        self.calculate_kinetic_energy()
        dof = 2 * self.num_particles - 3
        current_temperature = 2 * self.kinetic_energy * self.num_particles / dof
        
        scaling_factor = np.sqrt(target_temperature / max(current_temperature, 1e-10))
        self.velocities *= scaling_factor
        
        self.calculate_kinetic_energy()
        self.total_energy = self.kinetic_energy + self.potential_energy
        
        max_vel = 3.0 * np.sqrt(target_temperature)
        vel_norm = np.linalg.norm(self.velocities, axis=1)
        for i in range(self.num_particles):
            if vel_norm[i] > max_vel:
                self.velocities[i] = self.velocities[i] * max_vel / vel_norm[i]