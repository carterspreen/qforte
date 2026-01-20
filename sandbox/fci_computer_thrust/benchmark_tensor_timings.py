import qforte as qf
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the systems to benchmark
systems = [
    ('H2', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0))]),
    ('H4', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0))]),
    ('H6', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0)),
            ('H', (0., 0., 5.0)), ('H', (0., 0., 6.0))]),
    ('H8', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0)),
            ('H', (0., 0., 5.0)), ('H', (0., 0., 6.0)), ('H', (0., 0., 7.0)), ('H', (0., 0., 8.0))]),
    ('H10', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0)),
             ('H', (0., 0., 5.0)), ('H', (0., 0., 6.0)), ('H', (0., 0., 7.0)), ('H', (0., 0., 8.0)),
             ('H', (0., 0., 9.0)), ('H', (0., 0., 10.0))]),
    ('H12', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0)),
             ('H', (0., 0., 5.0)), ('H', (0., 0., 6.0)), ('H', (0., 0., 7.0)), ('H', (0., 0., 8.0)),
             ('H', (0., 0., 9.0)), ('H', (0., 0., 10.0)), ('H', (0., 0., 11.0)), ('H', (0., 0., 12.0))]),
    ('H14', [('H', (0., 0., 1.0)), ('H', (0., 0., 2.0)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.0)),
             ('H', (0., 0., 5.0)), ('H', (0., 0., 6.0)), ('H', (0., 0., 7.0)), ('H', (0., 0., 8.0)),
             ('H', (0., 0., 9.0)), ('H', (0., 0., 10.0)), ('H', (0., 0., 11.0)), ('H', (0., 0., 12.0)),
             ('H', (0., 0., 13.0)), ('H', (0., 0., 14.0))])
]

# Storage for results
system_names = []
cpu_timings = []
gpu_timings = []
speedups = []
num_qubits_list = []

print("=" * 80)
print("GPU vs CPU Tensor Apply Benchmark")
print("=" * 80)

for system_name, geom in systems:
    print(f"\n{'=' * 80}")
    print(f"Processing {system_name}")
    print(f"{'=' * 80}")
    
    # Build the molecular system
    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        basis='sto-3g', 
        run_fci=0,
        build_qb_ham=False,
        store_mo_ints=True,
        store_mo_ints_np=True,
        build_df_ham=0,
        df_icut=1.0e-6
    )
    
    ref = mol.hf_reference
    nel = sum(ref)
    sz = 0
    norb = int(len(ref) / 2)
    num_qubits = norb * 2
    
    print(f"Number of qubits: {num_qubits}")
    print(f"Number of electrons: {nel}")
    print(f"Number of orbitals: {norb}")
    
    # Initialize FCI computers
    fci_comp_cpu = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb, on_gpu=False, data_type="complex")
    
    fci_comp_cpu.hartree_fock()
    fci_comp_gpu.hartree_fock_cpu()
    
    timer = qf.local_timer()
    
    # CPU timing
    print("Running CPU apply tensor...")
    timer.reset()
    fci_comp_cpu.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb
    )
    cpu_time = timer.get()
    timer.record('FCI apply tensor')
    
    # GPU timing - prepare GPU tensors
    print("Running GPU apply tensor...")
    mo_oeis_gpu = qf.TensorGPU(mol.mo_oeis.shape(), "mo_oeis_gpu", False)
    mo_teis_gpu = qf.TensorGPU(mol.mo_teis.shape(), "mo_teis_gpu", False)
    mo_teis_einsum_gpu = qf.TensorGPU(mol.mo_teis_einsum.shape(), "mo_teis_einsum_gpu", False)
    mo_oeis_gpu.fill_from_tensor_cpu(mol.mo_oeis, mol.mo_oeis.shape())
    mo_teis_gpu.fill_from_tensor_cpu(mol.mo_teis, mol.mo_teis.shape())
    mo_teis_einsum_gpu.fill_from_tensor_cpu(mol.mo_teis_einsum, mol.mo_teis_einsum.shape())
    
    mo_oeis_gpu.to_gpu()
    mo_teis_gpu.to_gpu()
    mo_teis_einsum_gpu.to_gpu()
    fci_comp_gpu.to_gpu()
    
    timer.reset()
    fci_comp_gpu.apply_tensor_spat_012bdy_gpu(
        mol.nuclear_repulsion_energy, 
        mo_oeis_gpu, 
        mo_teis_gpu, 
        mo_teis_einsum_gpu, 
        norb
    )
    gpu_time = timer.get()
    timer.record('FCI GPU apply tensor')
    
    # Verify results match
    fci_comp_gpu.to_cpu()
    Cfci = fci_comp_cpu.get_state_deep()
    Cfci_gpu = qf.Tensor(Cfci.shape(), "Cfci_gpu")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci_gpu)
    Cfci.subtract(Cfci_gpu)
    diff_norm = Cfci.norm()
    
    print(f"\nResults for {system_name}:")
    print(f"  CPU time:     {cpu_time:.6f} s")
    print(f"  GPU time:     {gpu_time:.6f} s")
    print(f"  Speedup:      {cpu_time/gpu_time:.2f}x")
    print(f"  |dC| (diff):  {diff_norm:.2e}")

    # Store results
    system_names.append(system_name)
    cpu_timings.append(cpu_time)
    gpu_timings.append(gpu_time)
    speedups.append(cpu_time / gpu_time)
    num_qubits_list.append(num_qubits)

print("\n" + "=" * 80)
print("Benchmark Complete - Generating Plots")
print("=" * 80)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GPU vs CPU Tensor Apply Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: CPU vs GPU timings
ax1 = axes[0, 0]
x_pos = np.arange(len(system_names))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, cpu_timings, width, label='CPU', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, gpu_timings, width, label='GPU', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('System', fontweight='bold')
ax1.set_ylabel('Time (s)', fontweight='bold')
ax1.set_title('CPU vs GPU Execution Time')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(system_names)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8)

# Plot 2: Speedup
ax2 = axes[0, 1]
bars = ax2.bar(system_names, speedups, color='#2ecc71', alpha=0.8)
ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='No speedup')
ax2.set_xlabel('System', fontweight='bold')
ax2.set_ylabel('Speedup (CPU time / GPU time)', fontweight='bold')
ax2.set_title('GPU Speedup Factor')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: Timings vs Number of Qubits
ax3 = axes[1, 0]
ax3.plot(num_qubits_list, cpu_timings, 'o-', label='CPU', linewidth=2, 
         markersize=8, color='#3498db')
ax3.plot(num_qubits_list, gpu_timings, 's-', label='GPU', linewidth=2, 
         markersize=8, color='#e74c3c')
ax3.set_xlabel('Number of Qubits', fontweight='bold')
ax3.set_ylabel('Time (s)', fontweight='bold')
ax3.set_title('Scaling with System Size')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Log scale timings vs qubits
ax4 = axes[1, 1]
ax4.semilogy(num_qubits_list, cpu_timings, 'o-', label='CPU', linewidth=2, 
             markersize=8, color='#3498db')
ax4.semilogy(num_qubits_list, gpu_timings, 's-', label='GPU', linewidth=2, 
             markersize=8, color='#e74c3c')
ax4.set_xlabel('Number of Qubits', fontweight='bold')
ax4.set_ylabel('Time (s) - Log Scale', fontweight='bold')
ax4.set_title('Scaling with System Size (Log Scale)')
ax4.legend()
ax4.grid(alpha=0.3, which='both')

plt.tight_layout()

# Save the plot
output_file = 'gpu_vs_cpu_tensor_benchmark.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also save data to file
data_file = 'gpu_vs_cpu_tensor_benchmark.txt'
with open(data_file, 'w') as f:
    f.write("GPU vs CPU Tensor Apply Benchmark Results\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'System':<10} {'Qubits':<10} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<10} {'Diff Norm':<15}\n")
    f.write("-" * 80 + "\n")
    for i, name in enumerate(system_names):
        f.write(f"{name:<10} {num_qubits_list[i]:<10} {cpu_timings[i]:<15.6f} "
                f"{gpu_timings[i]:<15.6f} {speedups[i]:<10.2f}\n")

print(f"Data saved to: {data_file}")

# Print summary table
print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'System':<10} {'Qubits':<10} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<10}")
print("-" * 80)
for i, name in enumerate(system_names):
    print(f"{name:<10} {num_qubits_list[i]:<10} {cpu_timings[i]:<15.6f} "
          f"{gpu_timings[i]:<15.6f} {speedups[i]:<10.2f}x")
print("=" * 80)

plt.show()
