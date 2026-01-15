import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy import signal

class RoadProfileGenerator:
    """
    Generuje profile drogi r(t) i r_dot(t) zgodnie z wymaganiami M8.
    """
    def __init__(self, dt=0.001):
        self.dt = dt  # Krok symulacji [s]

    def sine_bump(self, height=0.04, length=0.5, speed=10.0, pre_time=0.5, post_time=2.0):
        """
        M8a. Przejazd przez garb (półfala sinusoidalna).
        """
        duration_bump = length / speed
        total_time = pre_time + duration_bump + post_time
        t = np.arange(0, total_time, self.dt)
        
        r = np.zeros_like(t)
        r_dot = np.zeros_like(t)
        
        # Indeksy
        start_idx = int(pre_time / self.dt)
        end_idx = start_idx + int(duration_bump / self.dt)
        
        # Właściwy garb
        if end_idx < len(t):
            t_bump = t[start_idx:end_idx] - pre_time
            omega = np.pi * speed / length
            
            r[start_idx:end_idx] = height * np.sin(omega * t_bump)
            r_dot[start_idx:end_idx] = height * omega * np.cos(omega * t_bump)
        
        return t, r, r_dot

    def road_noise(self, duration=5.0, bandwidth_hz=15.0, roughness=1e-4):
        """
        M8b. Szum drogi.
        """
        t = np.arange(0, duration, self.dt)
        white_noise = np.random.normal(0, 1, size=len(t))
        
        sos = signal.butter(4, bandwidth_hz, 'low', fs=1/self.dt, output='sos')
        r_filtered = signal.sosfilt(sos, white_noise)
        
        # Skalowanie
        r = r_filtered * np.sqrt(roughness)
        # Pochodna numeryczna
        r_dot = np.gradient(r, self.dt)
        
        return t, r, r_dot

    def curb_step(self, height=0.02, duration=3.0, start_time=0.5):
        """
        M8c. Skok krawężnika (schodek).
        Generuje impuls prędkości r_dot zamiast korzystać z warunku początkowego.
        """
        t = np.arange(0, duration, self.dt)
        r = np.zeros_like(t)
        r_dot = np.zeros_like(t)
        
        # Generacja profilu r (dla wykresów i odniesienia)
        start_idx = int(start_time / self.dt)
        if start_idx < len(t):
            r[start_idx:] = height
            
            # Generacja impulsu r_dot
            # Aproksymacja Diraca: prostokąt o szerokości dt i polu = height
            # Wysokość impulsu = height / dt
            r_dot[start_idx] = height / self.dt
            
        return t, r, r_dot

class ValidationRunner:
    """
    Przeprowadza symulacje i liczy metryki (M8d).
    Poprawka dla NumPy 2.0+: zamiana np.trapz na np.trapezoid
    """
    def __init__(self, model):
        self.model = model

    def calculate_metrics(self, t, y, u):
        # y = [zs_dot, zu-r, zs-zu]
        zs_dot = y[0]
        zu_r = y[1] 
        zs_zu = y[2]
        
        # Całka z kwadratu sterowania - obsługa zmian w NumPy 2.0
        # Używamy np.trapezoid jeśli dostępna, w przeciwnym razie stara np.trapz
        if u is not None:
            if hasattr(np, 'trapezoid'):
                int_u2 = np.trapezoid(u**2, t)
            else:
                int_u2 = np.trapz(u**2, t)
        else:
            int_u2 = 0
        
        metrics = {
            'RMS_Comfort (zs_dot)': np.sqrt(np.mean(zs_dot**2)),
            'RMS_Handling (zu-r)': np.sqrt(np.mean(zu_r**2)),
            'RMS_Suspension (zs-zu)': np.sqrt(np.mean(zs_zu**2)),
            'Max_Suspension_Deflection': np.max(np.abs(zs_zu)),
            'Control_Effort': int_u2
        }
        return metrics

    def run_passive(self, scenario_type='bump', **kwargs):
        """Uruchamia symulację dla układu biernego (u=0)."""
        gen = RoadProfileGenerator()
        
        # Generacja sygnałów w zależności od scenariusza
        if scenario_type == 'bump':
            t, r, r_dot = gen.sine_bump(**kwargs)
        elif scenario_type == 'noise':
            t, r, r_dot = gen.road_noise(**kwargs)
        elif scenario_type == 'curb':
            t, r, r_dot = gen.curb_step(**kwargs)
        else:
            raise ValueError(f"Nieznany scenariusz: {scenario_type}")

        # Przygotowanie modelu symulacyjnego
        # Wejście: r_dot (druga kolumna macierzy E)
        E_eff = self.model.E[:, [1]]
        
        # Układ otwarty (u=0), więc sterowanie B*u pomijamy (lub u=0)
        # Wyjścia: y = C*x (D=0)
        sys_passive = ct.ss(self.model.A, E_eff, self.model.C, np.zeros((3,1)))
        
        # Symulacja wymuszona sygnałem r_dot
        T, y = ct.forced_response(sys_passive, T=t, U=r_dot)
        
        # Obsługa wymiarów wyjścia (w zależności od wersji python-control)
        if y.ndim == 1:
            # Jeśli y jest płaskie, coś poszło nie tak z wymiarami, ale przy SISO powinno być ok.
            # Przy MIMO (3 wyjścia) y powinno być (3, N)
            pass 
        
        # Obliczenie metryk
        u_zeros = np.zeros_like(T)
        metrics = self.calculate_metrics(T, y, u_zeros)
        
        return T, y, r, metrics

class ValidationRunner:
    """
    Przeprowadza symulacje i liczy metryki (M8d).
    Wersja z poprawionymi wykresami (podpisy, profil drogi).
    """
    def __init__(self, model):
        self.model = model

    def calculate_metrics(self, t, y, u):
        # y = [zs_dot, zu-r, zs-zu]
        zs_dot = y[0]
        zu_r = y[1] 
        zs_zu = y[2]
        
        # Obsługa zmian w NumPy 2.0 (trapz -> trapezoid)
        if u is not None:
            if hasattr(np, 'trapezoid'):
                int_u2 = np.trapezoid(u**2, t)
            else:
                int_u2 = np.trapz(u**2, t)
        else:
            int_u2 = 0
        
        metrics = {
            'RMS_Comfort (zs_dot)': np.sqrt(np.mean(zs_dot**2)),
            'RMS_Handling (zu-r)': np.sqrt(np.mean(zu_r**2)),
            'RMS_Suspension (zs-zu)': np.sqrt(np.mean(zs_zu**2)),
            'Max_Suspension_Deflection': np.max(np.abs(zs_zu)),
            'Control_Effort': int_u2
        }
        return metrics

    def run_passive(self, scenario_type='bump', **kwargs):
        gen = RoadProfileGenerator()
        
        if scenario_type == 'bump':
            t, r, r_dot = gen.sine_bump(**kwargs)
        elif scenario_type == 'noise':
            t, r, r_dot = gen.road_noise(**kwargs)
        elif scenario_type == 'curb':
            t, r, r_dot = gen.curb_step(**kwargs)
        else:
            raise ValueError(f"Nieznany scenariusz: {scenario_type}")

        # Symulacja układu otwartego (u=0)
        # Wejście: r_dot (kolumna 1 macierzy E - indeksowanie od 0)
        E_eff = self.model.E[:, [1]]
        sys_passive = ct.ss(self.model.A, E_eff, self.model.C, np.zeros((3,1)))
        
        T, y = ct.forced_response(sys_passive, T=t, U=r_dot)
        
        if y.ndim == 1: pass 
        
        u_zeros = np.zeros_like(T)
        metrics = self.calculate_metrics(T, y, u_zeros)
        
        return T, y, r, metrics

    def plot_results(self, t, y, r, title):
        """
        Generuje czytelne wykresy z podpisami i profilem drogi.
        """
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        
        # 1. Komfort (Prędkość pionowa nadwozia)
        ax[0].plot(t, y[0], linewidth=1.5, label=r'$\dot{z}_s$ (Prędkość nadwozia)')
        ax[0].set_ylabel('Prędkość [m/s]')
        ax[0].set_title(f'{title} - Komfort', fontsize=11, fontweight='bold')
        ax[0].legend(loc='upper right')
        ax[0].grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 2. Prowadzenie (Ugięcie opony)
        # Dodajemy profil drogi r (przeskalowany) dla kontekstu
        ax[1].plot(t, r*1000, 'k--', alpha=0.2, label='Profil drogi $r$')
        ax[1].plot(t, y[1]*1000, 'g', linewidth=1.5, label=r'$z_u - r$ (Ugięcie opony)')
        ax[1].set_ylabel('Ugięcie [mm]')
        ax[1].set_title('Prowadzenie (Przyczepność)', fontsize=11, fontweight='bold')
        ax[1].legend(loc='upper right')
        ax[1].grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 3. Skok zawieszenia
        ax[2].plot(t, y[2]*1000, 'r', linewidth=1.5, label=r'$z_s - z_u$ (Skok)')
        # Opcjonalnie: Limit 80mm
        ax[2].axhline(80, color='k', linestyle=':', alpha=0.5, label='Limit +/- 80mm')
        ax[2].axhline(-80, color='k', linestyle=':', alpha=0.5)
        
        ax[2].set_ylabel('Skok [mm]')
        ax[2].set_xlabel('Czas [s]')
        ax[2].set_title('Skok zawieszenia (Przestrzeń robocza)', fontsize=11, fontweight='bold')
        ax[2].legend(loc='upper right')
        ax[2].grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# --- Uruchomienie Walidacji (Baseline) ---

