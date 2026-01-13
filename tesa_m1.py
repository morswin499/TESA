import numpy as np
import control as ct
import matplotlib.pyplot as plt

class ActiveSuspension:
    """
    Moduł M1: Model uproszczony (zgodny z instrukcją).
    Stan: x = [zs-zu, zs_dot, zu-r, zu_dot - r_dot]
    Zakłócenie: w = [r, r_dot] (pomijamy r_ddot)
    """
    def __init__(self, S_student):
        self.S = S_student if S_student >= 100 else S_student + 100
        
        # Parametry [cite: 22-27]
        self.ms = 280 + 0.4 * self.S
        self.mu = 35 + 0.05 * self.S
        self.ks = (18 + 0.02 * self.S) * 1e3
        self.kt = (150 + 0.05 * self.S) * 1e3
        self.cs = (1.4 + 0.002 * self.S) * 1e3
        self.ct = (0.4 + 0.001 * self.S) * 1e3
        
        self.u_max = 3000.0
        
        self._build_matrices()

    def _build_matrices(self):
        """
        Budowa macierzy.
        x4 = zu_dot - r_dot (prędkość względna)
        Wejście zakłócające w ograniczone do [r, r_dot].
        """
        # Macierz A (bez zmian - wynika z definicji stanu)
        self.A = np.array([
            [0, 1, 0, -1],
            [-self.ks/self.ms, -self.cs/self.ms, 0, self.cs/self.ms],
            [0, 0, 0, 1],
            [self.ks/self.mu, self.cs/self.mu, -self.kt/self.mu, -(self.cs+self.ct)/self.mu]
        ])

        # Macierz B (Sterowanie u)
        self.B = np.array([[0], [1/self.ms], [0], [-1/self.mu]])

        # Macierz E (Zakłócenie w = [r, r_dot]^T)
        # Pomijamy kolumnę dla r_ddot (zakładamy r_ddot ~ 0)
        # Kolumna 1 (r): 0
        # Kolumna 2 (r_dot): -1 (dla x1), cs/ms (dla x2), -cs/mu (dla x4)
        self.E = np.array([
            [0, -1],
            [0, self.cs/self.ms],
            [0, 0],
            [0, -self.cs/self.mu]
        ])

        # Macierz C
        self.C = np.array([
            [0, 1, 0, 0], # y1: zs_dot
            [0, 0, 1, 0], # y2: zu - r
            [1, 0, 0, 0]  # y3: zs - zu
        ])

        self.D = np.zeros((3, 1))
        
        # System podstawowy (dla sterowania u, przy w=0)
        self.sys = ct.ss(self.A, self.B, self.C, self.D)

    def run_sanity_check(self):
        """Test fizyki na skoku siły u=100N."""
        print(f"\n--- M1b. Weryfikacja (Simplified w=[r, rd], S={self.S}) ---")
        
        force = 100.0
        t, y = ct.step_response(self.sys, T=np.linspace(0, 2, 500))
        y = np.squeeze(y) * force
        
        check_v = np.mean(y[0][:50]) > 0
        check_susp = y[2][-1] > 0
        check_tire = y[1][-1] < 0
        
        status = "PASSED" if (check_v and check_susp and check_tire) else "FAILED"
        print(f"Test 1 (V>0): {'OK' if check_v else 'FAIL'}")
        print(f"Test 2 (Skok>0): {'OK' if check_susp else 'FAIL'}")
        print(f"Test 3 (Opona<0): {'OK' if check_tire else 'FAIL'}")
        print(f"WYNIK: {status}")
        
        return t, y, status

    def plot_sanity_check(self, t, y):
        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Sanity Check: u=100N")
        labels = [r'$\dot{z}_s$', r'$z_u - r$', r'$z_s - z_u$']
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(t, y[i], 'b')
            plt.ylabel(labels[i])
            plt.grid(True)
        plt.tight_layout()
        plt.show()