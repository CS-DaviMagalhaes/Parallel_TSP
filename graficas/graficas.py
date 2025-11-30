import matplotlib.pyplot as plt
import numpy as np
import os


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})
colors = {'omp': '#d62728', 'mpi': '#1f77b4', 'lns': '#2ca02c', 'ideal': '#7f7f7f', 
          'eff': '#ff7f0e', 'lns_bad': '#8c564b', 'sec': '#333333', 'mpi_p2': '#17becf'}


p_vals = [1, 2, 4, 8, 16, 32]

# --- SECUENCIAL ---
sec_n_scale = [5, 8, 10, 12, 15, 18, 20, 22, 25]
sec_t_scale = [0.1, 2, 10, 36, 352, 13551, 85338, 25790, 51666]
t_sec_n15 = 352.0
t_sec_n20 = 85338.0

# --- OPENMP 
# N=20
t_omp_n20 = [114394, 35714, 23605, 15288, 11144, 24843]
nodes_omp_n20 = [41669947, 22166218, 26081829, 23684132, 15643535, 29390618]
gflops_omp_n20 = [0.0145707, 0.0248264, 0.0441971, 0.0619679, 0.0561505, 0.0473222]
speedup_omp_n20 = [t_sec_n20 / t for t in t_omp_n20]

# N=15
t_omp_n15 = [590, 267, 178, 140, 147, 217]
speedup_omp_n15 = [t_sec_n15 / t for t in t_omp_n15]

# OMP Escalabilidad (p=16)
omp_n_scale = np.arange(15, 31)
omp_t_scale = [126, 381, 1461, 3105, 11076, 17182, 18778, 4647, 1757, 1630, 2414, 3724, 12991, 25065, 55457, 143331]

# OMP Granularidad
omp_depths = [1, 3, 6, 10]
omp_t_depth = [115988, 12902, 11522, 4122]

# --- MPI 
# N=20
t_mpi_n20 = [87529.9, 87078.1, 95596, 200187, 257327, 295869]
nodes_mpi_n20 = [41669947, 80130158, 153290619, 304318111, 558255693, 700009197]
gflops_mpi_n20 = [0.0190426, 0.0368084, 0.064141, 0.0608069, 0.0867776, 0.0946376]
speedup_mpi_n20 = [t_sec_n20 / t for t in t_mpi_n20]

# N=15
t_mpi_n15 = [342.868, 278.459, 313.036, 413.196, 414.565, 420.428]
speedup_mpi_n15 = [t_sec_n15 / t for t in t_mpi_n15]

# MPI Escalabilidad
mpi_n_scale = np.arange(15, 28)
t_mpi_p2 = [272.732, 1143.29, 5324.19, 13979.6, 84398.7, 86551.6, 49157.7, 24527.4, 9575.5, 11431.5, 55536.5, 196297, 593503]
t_mpi_p4 = [314.764, 1282.03, 6105.36, 14077.6, 83056.8, 96643.8, 112667, 48975.9, 14096.1, 12314.7, 58502.4, 204135, 610543]

# --- LNS Heurístico ---
lns_p = [1, 2, 4, 8, 16, 32]
# N=20 
t_lns_n20_bad = [176.985, 325.882, 534.869, 1277.88, 2114.06, 4781.1]
gflops_lns_n20_bad = [0.0497218, 0.0540073, 0.0658105, 0.0550911, 0.0666017, 0.0588985]
# N=131 
t_lns_n131_bad = [765.629, 1162.29, 2064.3, 5677.23, 9505.4, 19521.9]
gflops_lns_n131 = [0.493114, 0.649654, 0.731563, 0.532009, 0.635499, 0.618861]


# LNS Escalabilidad
lns_n = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 131]
lns_t = [88.6591, 106.583, 128.718, 150.346, 189.677, 203.801, 248.237, 278.3, 313.626, 360.085, 401.309, 479.483]
lns_gflops_scale = [0.0721866, 0.135106, 0.198885, 0.266053, 0.303674, 0.38469, 0.412508, 0.465684, 0.510162, 0.53765, 0.574121, 0.57265]




def save_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Generada: {filename}")
    plt.close()

# --- SECUENCIAL ---
plt.figure(figsize=(8, 5))
plt.plot(sec_n_scale, sec_t_scale, 'o-', color=colors['sec'], label='Secuencial')
plt.title('Escalabilidad Secuencial (Tiempo vs N)')
plt.xlabel('Tamaño N')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.legend()
plt.grid(True)
save_plot('sec_1_scalability.png')


# --- OPENMP ---

# 1.1a OMP Tiempo N=20
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_omp_n20, 'o-', color=colors['omp'], label='N=20')
plt.title('OpenMP: Tiempo N=20')
plt.xlabel('Hilos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
plt.grid(True)
save_plot('omp_1a_time_n20.png')

# 1.1b OMP Tiempo N=15
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_omp_n15, 'o--', color=colors['omp'], alpha=0.8, label='N=15')
plt.title('OpenMP: Tiempo N=15')
plt.xlabel('Hilos')
plt.ylabel('Tiempo (ms)')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
plt.grid(True)
save_plot('omp_1b_time_n15.png')

# 1.1c OMP Comparativa
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_omp_n20, 'o-', color=colors['omp'], label='N=20')
plt.plot(p_vals, t_omp_n15, 'o--', color=colors['omp'], alpha=0.6, label='N=15')
plt.title('OpenMP: N=20 vs N=15')
plt.xlabel('Hilos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
plt.grid(True)
save_plot('omp_1c_time_compare.png')

# 1.2 Speedup
plt.figure(figsize=(8, 5))
plt.plot(p_vals, speedup_omp_n20, 'o-', color=colors['omp'], label='N=20')
plt.plot(p_vals, speedup_omp_n15, 'o--', color=colors['omp'], alpha=0.5, label='N=15')
plt.plot(p_vals, p_vals, '--', color=colors['ideal'], label='Ideal Lineal', alpha=0.5)
plt.title('OpenMP: Speedup Absoluto')
plt.xlabel('Hilos')
plt.ylabel('Speedup')
plt.legend()
plt.xticks(p_vals)
save_plot('omp_2_speedup.png')

# 1.3 Escalabilidad N
plt.figure(figsize=(10, 6))
plt.plot(omp_n_scale, omp_t_scale, 'o-', color=colors['omp'])
plt.title('OpenMP: Escalabilidad N (p=16)')
plt.xlabel('Tamaño N')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
save_plot('omp_4_scalability_N.png')

# 1.4 Granularidad
plt.figure(figsize=(8, 5))
plt.plot([str(d) for d in omp_depths], omp_t_depth, 'r-o')
plt.title('OpenMP: Granularidad (Depth)')
plt.xlabel('MAX_TASK_DEPTH')
plt.ylabel('Tiempo (ms)')
save_plot('omp_5_granularity.png')

# 1.5 OMP GFLOPs Especifico
plt.figure(figsize=(8, 5))
plt.plot(p_vals, gflops_omp_n20, 'D-', color=colors['omp'])
plt.title('OpenMP: GFLOPs vs Hilos (N=20)')
plt.xlabel('Hilos')
plt.ylabel('GFLOPs')
plt.xticks(p_vals)
save_plot('omp_6_gflops_n20.png')


# --- MPI ---

# 2.1a MPI Tiempo N=20
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_mpi_n20, 's-', color=colors['mpi'], label='N=20')
plt.title('MPI: Tiempo N=20')
plt.xlabel('Procesos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
save_plot('mpi_1a_time_n20.png')

# 2.1b MPI Tiempo N=15
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_mpi_n15, 's--', color=colors['mpi'], alpha=0.8, label='N=15')
plt.title('MPI: Tiempo N=15')
plt.xlabel('Procesos')
plt.ylabel('Tiempo (ms)')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
save_plot('mpi_1b_time_n15.png')

# 2.1c MPI Comparativa
plt.figure(figsize=(8, 5))
plt.plot(p_vals, t_mpi_n20, 's-', color=colors['mpi'], label='N=20')
plt.plot(p_vals, t_mpi_n15, 's--', color=colors['mpi'], alpha=0.6, label='N=15')
plt.title('MPI: N=20 vs N=15')
plt.xlabel('Procesos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
save_plot('mpi_1c_time_compare.png')

# 2.2 Speedup
plt.figure(figsize=(8, 5))
plt.plot(p_vals, speedup_mpi_n20, 's-', color=colors['mpi'], label='N=20')
plt.plot(p_vals, speedup_mpi_n15, 's--', color=colors['mpi'], alpha=0.5, label='N=15')
plt.plot(p_vals, p_vals, '--', color=colors['ideal'], label='Ideal', alpha=0.5)
plt.title('MPI: Speedup')
plt.xlabel('Procesos')
plt.ylabel('Speedup')
plt.xticks(p_vals)
plt.legend()
save_plot('mpi_2_speedup.png')

# 2.3 Escalabilidad N (Comparativa p=2 vs p=4)
plt.figure(figsize=(10, 6))
plt.plot(mpi_n_scale, t_mpi_p2, 's-', color=colors['mpi_p2'], label='MPI (p=2)')
plt.plot(mpi_n_scale, t_mpi_p4, 's--', color=colors['mpi'], label='MPI (p=4)')
plt.title('MPI: Escalabilidad N (p=2 vs p=4)')
plt.xlabel('Tamaño N')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.legend()
save_plot('mpi_3_scalability_N.png')

# 2.4 MPI GFLOPs Especifico (NUEVO)
plt.figure(figsize=(8, 5))
plt.plot(p_vals, gflops_mpi_n20, 'D-', color=colors['mpi'])
plt.title('MPI: GFLOPs vs Procesos (N=20)')
plt.xlabel('Procesos')
plt.ylabel('GFLOPs')
plt.xticks(p_vals)
save_plot('mpi_4_gflops_n20.png')


# --- LNS  ---

# 3.1 Overhead
plt.figure(figsize=(9, 5))
plt.plot(lns_p, t_lns_n20_bad, 'o-', color=colors['lns_bad'], label='N=20')
plt.plot(lns_p, t_lns_n131_bad, 's--', color=colors['lns'], label='N=131')
plt.title('LNS: Overhead')
plt.xlabel('Procesos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(lns_p)
plt.legend()
save_plot('lns_0_overhead.png')

# 3.2 Escalabilidad
plt.figure(figsize=(10, 6))
plt.plot(lns_n, lns_t, 'D-', color=colors['lns'])
plt.title('LNS: Escalabilidad (p=4)')
plt.xlabel('Tamaño N')
plt.ylabel('Tiempo (ms)')
save_plot('lns_1_scalability.png')

# 3.3 LNS GFLOPs vs P (N=131) 
plt.figure(figsize=(8, 5))
plt.plot(lns_p, gflops_lns_n131, 'D-', color=colors['lns'])
plt.title('LNS: GFLOPs vs Procesos (N=131)')
plt.xlabel('Procesos')
plt.ylabel('GFLOPs')
plt.xticks(lns_p)
save_plot('lns_3_gflops_n131_p.png')

# 3.4 GFLOPs Comparativa N=20 vs N=131
plt.figure(figsize=(8, 5))
plt.plot(lns_p, gflops_lns_n20_bad, 'o-', color=colors['lns_bad'], label='N=20 (Bad)')
plt.plot(lns_p, gflops_lns_n131, 'D--', color=colors['lns'], label='N=131')
plt.title('LNS: GFLOPs vs Procesos (Comparativa)')
plt.xlabel('Procesos')
plt.ylabel('GFLOPs')
plt.xticks(lns_p)
plt.legend()
save_plot('lns_4_gflops_compare.png')


# --- COMPARATIVAS ---

# 4.1 OMP vs MPI vs LNS vs Secuencial
plt.figure(figsize=(10, 6))
plt.axhline(t_sec_n20, color=colors['sec'], linestyle='--', label='Secuencial', linewidth=2)
plt.plot(p_vals, t_omp_n20, 'o-', color=colors['omp'], label='OpenMP')
plt.plot(p_vals, t_mpi_n20, 's-', color=colors['mpi'], label='MPI')
plt.plot(lns_p, t_lns_n20_bad, 'D-', color=colors['lns'], label='LNS')
plt.title('Comparativa Total: Todos los Métodos (N=20)')
plt.xlabel('Hilos / Procesos')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.xticks(p_vals)
plt.legend()
save_plot('comp_3_all_methods_n20.png')

# 4.2 Exacto vs Heurístico
plt.figure(figsize=(10, 6))
plt.plot(omp_n_scale, omp_t_scale, 'o-', color=colors['omp'], label='Exacto (OMP)')
plt.plot(lns_n, lns_t, 'D-', color=colors['lns'], label='Heurístico (LNS)')
plt.title('Paradigma: Exacto vs Heurístico')
plt.xlabel('Tamaño N')
plt.ylabel('Tiempo (ms) - Log')
plt.yscale('log')
plt.legend()
save_plot('comp_2_exact_vs_heuristic.png')


# --- (NODOS Y GFLOPS) ---

# 5.1 OMP Nodos Visitados
plt.figure(figsize=(8, 5))
plt.plot([str(p) for p in p_vals], nodes_omp_n20, 'o-', color=colors['omp'])
plt.title('OpenMP: Nodos Visitados')
plt.xlabel('Hilos')
plt.ylabel('Nodos (Log)')
plt.yscale('log')
save_plot('metrica_1_nodos_omp.png')

# 5.2 MPI Nodos Visitados
plt.figure(figsize=(8, 5))
plt.plot(p_vals, nodes_mpi_n20, 's-', color=colors['mpi'], label='MPI Nodos')
plt.title('MPI: Nodos Visitados')
plt.xlabel('Procesos')
plt.ylabel('Nodos (Log)')
plt.yscale('log')
plt.xticks(p_vals)
plt.grid(True)
save_plot('metrica_2_nodos_mpi.png')

# 5.3 Comparativa GFLOPs Totales
plt.figure(figsize=(9, 6))
plt.plot(p_vals, gflops_omp_n20, 'o-', color=colors['omp'], label='OpenMP')
plt.plot(p_vals, gflops_mpi_n20, 's-', color=colors['mpi'], label='MPI')
plt.title('Comparativa GFLOPs (N=20)')
plt.xlabel('Hilos / Procesos')
plt.ylabel('GFLOPs')
plt.legend()
plt.xticks(p_vals)
save_plot('metrica_4_comp_gflops.png')

# 5.4 GFLOPs LNS Bueno
plt.figure(figsize=(8, 5))
plt.plot(lns_n, lns_gflops_scale, 'D-', color=colors['lns'])
plt.title('LNS: GFLOPs (Escalabilidad)')
plt.xlabel('Tamaño N')
plt.ylabel('GFLOPs')
save_plot('lns_gflops_bueno.png')
