# The lines below are necessary so that correct paths to the rok and reaktoro libraries are used
import sys
sys.path.remove('/home/skyas/polybox/rok')
sys.path.insert(1, '/home/skyas/work/allanleal-cpp-reactivetransportsolver-demo/build/lib/python3.7/site-packages')
#print(sys.path)

# Import necessary libraries
import reaktoro as rkt
import rok
import numpy as np
from progress.bar import IncrementalBar
import time

# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
month = 30 *day
year = 365 * day

# Thermodynamical parameters for the reactive transport simulation
T = 25.0 + 273.15  # the temperature (in units of K)
P_left = 1.01325 * 200 * 1e5  # the pressure (in units of Pa) on the left boundary
P_right = 9e-1 * P_left # the pressure on the right boundary
P = P_left

# Discretization parameters for the reactive transport simulation
# Parameters for the reactive transport simulation
lx = 100
ly = 30
nx = 150  # the number of mesh cells along the x-coordinate
ny = 120  # the number of mesh cells along the y-coordinate
nsteps = 1000  # the number of time steps
cfl = 0.3     # the CFL number to be used in the calculation of time step
nnodes = (nx + 1)*(ny + 1)

# PDE methods
method_flow = "sdhm"
method_transport = 'supg'

# Activity model for the aqueous species

#activity_model = "pitzer-full"
#activity_model = "pitzer-selected-species"
activity_model = "dk-full"
#activity_model = "dk-selected-species"

# -------------------------------------------------------------------------------------------------------------------- #
# Parameters and auxiliary stats lists for the ODML algorithm
# -------------------------------------------------------------------------------------------------------------------- #

smart_equlibrium_reltol = 0.01
#smart_equlibrium_reltol = 0.005
#smart_equlibrium_reltol = 0.001

amount_fraction_cutoff = 1e-14
mole_fraction_cutoff = 1e-14

tag_smart = "-" + activity_model + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nnodes-" + str(nnodes) + \
      "-nsteps-" + str(nsteps) + \
      "-reltol-" + "{:.{}e}".format(smart_equlibrium_reltol, 1) + \
      "-smart"
tag_conv = "-" + activity_model + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nnodes-" + str(nnodes) + \
      "-nsteps-" + str(nsteps) + \
      "-conv"
folder_results = 'results-fixed-pressure/demo-chemicaltransport-heterogeneous-odml-scavenging'

# The seconds spent on equilibrium and transport calculations per time step
time_steps = []
timings_equilibrium_smart = []  # using conventional equilibrium solver
timings_equilibrium_conv = []  # using smart equilibrium solver
timings_transport = []

# The counter of full chemical equilibrium training calculations each time step
learnings = []

def print_test_summary():

    print("Activity model  :", activity_model)
    print("ODML retol      :", smart_equlibrium_reltol)
    print("Number of DOFs  : {} = ({} + 1) x ({} + 1)".format(nnodes, nx, ny))
    print("Number of steps :", nsteps)

print_test_summary()

# -------------------------------------------------------------------------------------------------------------------- #
# Define distretization setting
# -------------------------------------------------------------------------------------------------------------------- #

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
x_coords = mesh.coordinates.dat.data[:, 0]
y_coords = mesh.coordinates.dat.data[:, 1]

# Initialize the function spaces
V = rok.FunctionSpace(mesh, "CG", 1)

# Number of degrees of freedom in the functional space V
ndofs = V.dof_count

# -------------------------------------------------------------------------------------------------------------------- #
# Model parameters
# -------------------------------------------------------------------------------------------------------------------- #

# Parameters for the flow simulation
D = rok.Constant(0)  # the diffusion coefficient (in units of m2/s)
rho = rok.Constant(997.05)  # water density (in units of kg/m3), https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.html?vA=25&units=C#
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.permeability(V)
f = rok.Constant(0.0)  # the source rate in the flow calculation

# -------------------------------------------------------------------------------------------------------------------- #
# Flow problem
# -------------------------------------------------------------------------------------------------------------------- #

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(P_left, "left")
problem.addPressureBC(P_right, "right")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "bottom")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "top")

flow = rok.DarcySolver(problem, method=method_flow)
flow.solve()

# -------------------------------------------------------------------------------------------------------------------- #
# Chemical system
# -------------------------------------------------------------------------------------------------------------------- #

# Initialise the chemical database
db = rkt.Database('supcrt07.xml')
dhModel = rkt.DebyeHuckelParams()
dhModel.setPHREEQC()

# Initialise the chemical editor
editor = rkt.ChemicalEditor(db)

elements_list = 'C Ca Cl Fe H K Mg Na O S'
selected_species = ['Ca(HCO3)+', 'CO3--', 'CaCO3(aq)', 'Ca++', 'CaSO4(aq)', 'CaOH+', 'Cl-',
                    'FeCl++', 'FeCl2(aq)', 'FeCl+', 'Fe++', 'FeOH+', 'Fe+++', 'FeOH++',
                    'H2S(aq)', 'H2(aq)', 'HS-', 'H2O(l)', 'H+', 'OH-', 'HCO3-', 'HSO4-',
                    'KSO4-',  'K+', 'Mg++', 'Mg(HCO3)+', 'MgCO3(aq)', 'MgSO4(aq)', 'MgOH+',
                    'Na+', 'NaSO4-', 'O2(aq)', 'S5--', 'S4--', 'S3--', 'S2--', 'SO4--']
# Initialise the chemical editor phases
if activity_model == "pitzer-full":
    editor.addAqueousPhaseWithElements(elements_list) \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "pitzer-selected-species":
    editor.addAqueousPhase(selected_species) \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "hkf-full":
    editor.addAqueousPhaseWithElements(elements_list)
elif activity_model == "hkf-selected-species":
    editor.addAqueousPhase(selected_species)
elif activity_model == "dk-full":
    editor.addAqueousPhaseWithElements(elements_list) \
        .setChemicalModelDebyeHuckel(dhModel)
elif activity_model == "dk-selected-species":
    editor.addAqueousPhase(selected_species) \
        .setChemicalModelDebyeHuckel(dhModel)
editor.addMineralPhase('Pyrrhotite')
editor.addMineralPhase('Siderite')

# Initialise the chemical system
system = rkt.ChemicalSystem(editor)

# Define equilibrium options
equilibrium_options = rkt.EquilibriumOptions()
equilibrium_options.hessian = rkt.GibbsHessian.Exact # ensure the use of an exact Hessian of the Gibbs energy function
equilibrium_options.optimum.tolerance = 1e-10 # ensure the use of a stricter residual tolerance for the Gibbs energy minimization
# Define smart equilibrium options
smart_equilibrium_options = rkt.SmartEquilibriumOptions()
smart_equilibrium_options.reltol = smart_equlibrium_reltol
smart_equilibrium_options.amount_fraction_cutoff = amount_fraction_cutoff
smart_equilibrium_options.mole_fraction_cutoff = mole_fraction_cutoff

# -------------------------------------------------------------------------------------------------------------------- #
# Output results
# -------------------------------------------------------------------------------------------------------------------- #

# Initialize the array for documenting the cells, where smart prediction happened
rt_results = []

# List of name of species and elements we track
species_out = ["Fe++", "H+", "HS-", "S2--", "CO3--", "HSO4-", "H2S(aq)", "Pyrrhotite", "Siderite"]
elements_out = ["C", "Ca", "Cl", "Fe", "H", "K", "Mg", "Na", "O", "S", "Z"]

# List of functions representing species and elements we track
n_out = [rok.Function(V, name=name) for name in species_out]
b_out = [rok.Function(V, name=name) for name in elements_out]

def make_results_folders(use_smart_equilibrium_solver):

    if use_smart_equilibrium_solver:
        folder_results_ = folder_results + tag_smart
    else:
        folder_results_ = folder_results + tag_conv

    file_species_amounts = rok.File(folder_results_ + "/species-amounts.pvd")
    file_element_amounts = rok.File(folder_results_ + "/element-amounts.pvd")
    file_porosity = rok.File(folder_results_ + "/porosity.pvd")
    file_volume = rok.File(folder_results_ + '/volume.pvd')
    file_ph = rok.File(folder_results_ + '/ph.pvd')

    rok.File(folder_results + tag_smart + '/flow.pvd').write(flow.u, flow.p, k)

    return file_species_amounts, file_element_amounts, file_porosity, file_volume, file_ph

def run_transport(use_smart_equilibrium):

    # ---------------------------------------------------------------------------------------------------------------- #
    # Initial and boundary chemical states
    # ---------------------------------------------------------------------------------------------------------------- #

    # Define the initial condition of the reactive transport modeling problem
    problem_ic = rkt.EquilibriumInverseProblem(system)
    problem_ic.setTemperature(T)
    problem_ic.setPressure(P)
    problem_ic.add("H2O", 58.0, "kg")
    problem_ic.add("Cl-", 1122.3e-3, "kg")
    problem_ic.add("Na+", 624.08e-3, "kg")
    problem_ic.add("SO4--", 157.18e-3, "kg")
    problem_ic.add("Mg++", 74.820e-3, "kg")
    problem_ic.add("Ca++", 23.838e-3, "kg")
    problem_ic.add("K+", 23.142e-3, "kg")
    problem_ic.add("HCO3-", 8.236e-3, "kg")
    problem_ic.add("O2(aq)", 58e-12, "kg")
    # Initial composition is with Siderite only
    problem_ic.add("Pyrrhotite", 0.0, "mol")
    problem_ic.add("Siderite", 0.5, "mol")
    problem_ic.pH(8.951)

    # Define the boundary condition of the reactive transport modeling problem
    problem_bc = rkt.EquilibriumInverseProblem(system)
    problem_bc.setTemperature(T)
    problem_bc.setPressure(P)
    problem_bc.add("H2O", 58.0, "kg")
    problem_bc.add("Cl-", 1122.3e-3, "kg")
    problem_bc.add("Na+", 624.08e-3, "kg")
    problem_bc.add("SO4--", 157.18e-3, "kg")
    problem_bc.add("Mg++", 74.820e-3, "kg")
    problem_bc.add("Ca++", 23.838e-3, "kg")
    problem_bc.add("K+", 23.142e-3, "kg")
    problem_bc.add("HCO3-", 8.236e-3, "kg")
    problem_bc.add("O2(aq)", 58e-12, "kg")
    # No minerals in the injected brine
    problem_bc.add("Pyrrhotite", 0.0, "mol")
    problem_bc.add("Siderite", 0.0, "mol")
    # Scavenger, H2S-brine
    problem_bc.add("HS-", 0.0196504, "mol")
    problem_bc.add("H2S(aq)", 0.167794, "mol")
    problem_bc.pH(5.726)

    # Calculate the equilibrium states for the initial and boundary conditions
    state_ic = rkt.equilibrate(problem_ic)
    state_bc = rkt.equilibrate(problem_bc)

    # Fetch teh value of the ph in the initial chemical state
    evaluate_pH = rkt.ChemicalProperty.pH(system)
    print("ph(IC) = ", evaluate_pH(state_ic.properties()).val)
    print("ph(BC) = ", evaluate_pH(state_bc.properties()).val)

    # Scale the volumes of the phases in the initial condition such that their sum is 1 m3
    state_ic.scalePhaseVolume("Aqueous", 0.1, "m3")
    state_ic.scaleVolume(1.0, "m3")

    # Scale the volume of the boundary equilibrium state to 1 m3
    state_bc.scaleVolume(1.0)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Field with chemical states
    # ---------------------------------------------------------------------------------------------------------------- #

    # Initialise the chemical field
    field = rok.ChemicalField(system, V)
    field.fill(state_ic)
    field.update()

    # Project values of the pressure field from the dofs from DG1 to CG1 finite element spaces
    pressures = np.zeros(ndofs)
    for i, x, y in zip(np.linspace(0, ndofs - 1, num=ndofs, dtype=int), x_coords, y_coords):
        if x == 0.0:
            pressures[i] = P_left
        elif x == lx:
            pressures[i] = P_right
        else:
            pressures[i] = flow.p.at([x, y])
        if pressures[i] < 0:
            print(f"{i}: p({x}, {y}) = {pressures[i]}")

    # Set the pressure field to the chemical field
    field.setPressures(pressures)
    #field.setPressures(flow.p.dat.data)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Transport problem
    # ---------------------------------------------------------------------------------------------------------------- #

    # Initialize the chemical transport options
    transport_options = rok.ChemicalTransportOptions()
    transport_options.use_smart_equilibrium = use_smart_equilibrium
    transport_options.equilibrium = equilibrium_options
    transport_options.smart_equilibrium = smart_equilibrium_options

    # Initialize the chemical transport solver
    transport = rok.ChemicalTransportSolver(field, method=method_transport)
    transport.setOptions(transport_options)
    transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
    transport.setVelocity([flow.u])
    transport.setDiffusion([D])

    # ---------------------------------------------------------------------------------------------------------------- #
    # Problem characteristics and discretization parameters
    # ---------------------------------------------------------------------------------------------------------------- #

    max_ux = np.max(flow.u.dat.data[:, 0])
    max_uy = np.max(flow.u.dat.data[:, 1])
    delta_x = lx / nx
    delta_y = ly / ny

    # Define time step according to the velocity
    dt = cfl / max(max_ux / delta_x, max_uy / delta_y)

    print("dx = ", delta_x)
    print("dy = ", delta_y)
    print("dofs = ", ndofs)

    # Outputting the results
    print('max(u) = ', np.max(flow.u.dat.data[:, 0]), flush=True)
    print('max(k) = ', np.max(k.dat.data), flush=True)
    print('div(u)*dx =', rok.assemble(rok.div(flow.u) * rok.dx), flush=True)
    print('dt = {} minute'.format(dt / minute), flush=True)

    # Create the output files in the folders corresponding to the selected method
    file_species_amounts, file_element_amounts, file_porosity, file_volume, file_ph \
        = make_results_folders(use_smart_equilibrium)

    # ------------------------------------------------------------------------------------------------------------ #
    # Run time-dependent loop
    # ------------------------------------------------------------------------------------------------------------ #

    t = 0.0
    step = 0

    if use_smart_equilibrium: bar = IncrementalBar('Reactive transport with the ODML algorithm:', max=nsteps)
    else: bar = IncrementalBar('Reactive transport with the conventional algorithm:', max=nsteps)

    selected_steps = [1, 100, 200, 400, 1000]

    while step < nsteps:

        # if step in selected_steps:
        #     print('Progress at step {}: {:.2f} hours / {:.2f} days'.format(step, t / hour, t / day), flush=True)

        if step % 10 == 0:
            # For each selected species, output its molar amounts
            for f in n_out:
                f.assign(field.speciesAmount(f.name()))

            # For each selected species, output its molar amounts
            for f in b_out:
                f.assign(field.elementAmountInPhase(f.name(), "Aqueous"))

            file_species_amounts.write(*n_out)
            file_element_amounts.write(*b_out)
            file_porosity.write(field.porosity())
            file_volume.write(field.volume())
            file_ph.write(field.pH())

        # Perform one transport step from `t` to `t + dt`
        transport.step(field, dt)

        # Save the reactive transport results on each time step
        rt_results.append(transport.result)

        # Collect intermediate results
        # ----------------------------------------------------------------------------------------------------------
        timings_transport.append(transport.result.seconds)

        # If the ODML is chosen save the results into the smart cells arrays with True/ False and to learning arrays
        if use_smart_equilibrium:
            learnings.append(ndofs - np.count_nonzero(transport.result.smart_equilibrium.smart_predictions.vector()[:]))
            timings_equilibrium_smart.append(transport.result.seconds_equilibrium)
            # print("timings_equilibrium_smart = ", timings_equilibrium_smart)
            # print("learnings = ", learnings)
        else:
            timings_equilibrium_conv.append(transport.result.seconds_equilibrium)
            # print("timings_equilibrium_conv = ", timings_equilibrium_conv)

        # Update the current time
        step += 1
        t += dt

        bar.next()
    bar.finish()
    if use_smart_equilibrium:
        transport.equilibrium.outputClusterInfo()

# -------------------------------------------------------------------------------------------------------------------- #
# Run reactive transport with the ODML algorithm
# -------------------------------------------------------------------------------------------------------------------- #

start_rt = time.time()

use_smart_equilibrium = True; run_transport(use_smart_equilibrium)

timing_rt_smart = time.time() - start_rt

# ---------------------------------------------------------- #
# Summarize and plot results
# ---------------------------------------------------------- #

print("Total learnings: {} out of {} ( {:<5.2f}% )".format(int(np.sum(learnings)), ndofs * len(learnings), int(np.sum(learnings)) / ndofs / len(learnings) * 100))
import os
plots_folder_results = folder_results + "-plots" + tag_smart
os.system('mkdir -p ' + plots_folder_results)

# Plotting of the number of the learning:
import plotting as plt
time_steps = np.linspace(0, len(learnings), len(learnings))
plt.plot_on_demand_learning_countings_mpl(time_steps, learnings, plots_folder_results)
np.savetxt(folder_results + tag_smart + '/learnings.txt', learnings)
np.savetxt(folder_results + tag_smart + '/time-smart.txt', timings_equilibrium_smart)
np.savetxt(folder_results + tag_smart + '/time-transport.txt', timings_transport)

# -------------------------------------------------------------------------------------------------------------------- #
# Run reactive transport with the conventional algorithm
# -------------------------------------------------------------------------------------------------------------------- #

start_rt = time.time()

use_smart_equilibrium = False; run_transport(use_smart_equilibrium)

timing_rt_conv = time.time() - start_rt

np.savetxt(folder_results + tag_smart + '/time-conv.txt', timings_equilibrium_conv)

# --------------------------------------------------------------------------
# Analyze and plot results
# --------------------------------------------------------------------------
print("Chem. calc. speedup : ", np.sum(timings_equilibrium_conv) / np.sum(timings_equilibrium_smart))
print("Total speedup       : ", timing_rt_conv/timing_rt_smart)
print("")
# Plot with the CPU time comparison and speedup:
step = 1
plt.plot_computing_costs_mpl(time_steps, (timings_equilibrium_conv, timings_equilibrium_smart, timings_transport), step, plots_folder_results)
plt.plot_speedups_mpl(time_steps, (timings_equilibrium_conv, timings_equilibrium_smart), step, plots_folder_results)

