import numpy as np
import control as ct
import matplotlib.pyplot as plt

class ActiveSuspension:
    """
    Moduł M1 + M2: Pomiary = [Prędkość nadwozia, Ugięcie zawieszenia].
    Stan: x = [zs-zu, zs_dot, zu-r, zu_dot - r_dot]
    """
    def __init__(self, S_student):
        self.S = S_student if S_student >= 100 else S_student + 100
        
        # [cite_start]Parametry [cite: 22-27]
        self.ms = 280 + 0.4 * self.S
        self.mu = 35 + 0.05 * self.S
        self.ks = (18 + 0.02 * self.S) * 1e3
        self.kt = (150 + 0.05 * self.S) * 1e3
        self.cs = (1.4 + 0.002 * self.S) * 1e3
        self.ct = (0.4 + 0.001 * self.S) * 1e3
        
        self.u_max = 3000.0
        
        self._build_matrices()

    def _build_matrices(self):
        # --- A, B, E (Bez zmian) ---
        self.A = np.array([
            [0, 1, 0, -1],
            [-self.ks/self.ms, -self.cs/self.ms, 0, self.cs/self.ms],
            [0, 0, 0, 1],
            [self.ks/self.mu, self.cs/self.mu, -self.kt/self.mu, -(self.cs+self.ct)/self.mu]
        ])

        self.B = np.array([[0], [1/self.ms], [0], [-1/self.mu]])

        # Uproszczone E (dla w=[r, r_dot])
        self.E = np.array([
            [0, -1],
            [0, self.cs/self.ms],
            [0, 0],
            [0, -self.cs/self.mu]
        ])

        # --- C, D (Wyjścia metryk jakości) ---
        self.C = np.array([
            [0, 1, 0, 0], # y1: zs_dot
            [0, 0, 1, 0], # y2: zu - r
            [1, 0, 0, 0]  # y3: zs - zu
        ])
        self.D = np.zeros((3, 1))

        # --- C_meas, D_meas (POMIARY: Prędkość i Ugięcie) ---
        # y_meas = [zs_dot, zs - zu]^T
        # Odpowiada to stanom x2 oraz x1.
        
        self.C_meas = np.array([
            [0, 1, 0, 0],  # Wybiera x2 (zs_dot)
            [1, 0, 0, 0]   # Wybiera x1 (zs - zu)
        ])
        
        self.D_meas = np.zeros((2, 1)) # Brak wpływu u na pomiar (bo nie mierzymy przyspieszenia)
        
        # Systemy
        self.sys = ct.ss(self.A, self.B, self.C, self.D)
        self.sys_meas = ct.ss(self.A, self.B, self.C_meas, self.D_meas)

    def run_sanity_check(self):
        # Kod bez zmian
        print(f"\n--- M1b. Weryfikacja (S={self.S}) ---")
        force = 100.0
        t, y = ct.step_response(self.sys, T=np.linspace(0, 2, 500))
        y = np.squeeze(y) * force
        
        check_v = np.mean(y[0][:50]) > 0
        check_susp = y[2][-1] > 0
        check_tire = y[1][-1] < 0
        status = "PASSED" if (check_v and check_susp and check_tire) else "FAILED"
        print(f"Status: {status}")
        return t, y, status

    def plot_sanity_check(self, t, y):
        # Kod bez zmian
        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Sanity Check: u=100N")
        labels = [r'$\dot{z}_s$', r'$z_u - r$', r'$z_s - z_u$']
        for i in range(3):
            plt.subplot(3, 1, i+1); plt.plot(t, y[i], 'b'); plt.ylabel(labels[i]); plt.grid(True)
        plt.tight_layout(); plt.show()