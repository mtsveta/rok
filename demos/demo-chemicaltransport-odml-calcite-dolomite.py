import sys
sys.path.insert(2, '/home/skyas/polybox/allanleal-cpp-reactivetransportsolver-demo/build/lib/python3.7/site-packages')

import plotting as plt
import firedrake as fire
import reaktoro as rkt
import rok

import numpy as np
import time
from progress.bar import IncrementalBar
# --------------------------------------------------------------------------
# Parameters for the reactive transport
# --------------------------------------------------------------------------

# Auxiliary time related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Parameters for the reactive transport simulation
lx = 2
ly = 1
lz = 0
nx = 50  # the number of mesh cells along the x-coordinate
ny = 25  # the number of mesh cells along the y-coordinate
nz = 0  # the number of mesh cells along the y-coordinate

dt = 30 * minute  # the time step (in units of s)
nsteps = 5000  # the number of time steps

D = fire.Constant(1.0e-9)  # the diffusion coefficient (in units of m2/s)
v = fire.Constant([1.0 / week, 0.0])  # the fluid pore velocity (in units of m/s)
T = 60.0 + 273.15  # the temperature (in units of K)
P = 100 * 1e5  # the pressure (in units of Pa)

# Activity model for the aqueous species

activity_model = "hkf-full"
#activity_model = "hkf-selected-species"
#activity_model = "pitzer-full"
#activity_model = "pitzer-selected-species"
#activity_model = "dk-full"
#activity_model = "dk-selected-species"


# --------------------------------------------------------------------------
# Parameters for the ODML algorithm
# --------------------------------------------------------------------------

smart_equlibrium_reltol = 0.001
amount_fraction_cutoff = 1e-14
mole_fraction_cutoff = 1e-14

tag_smart = "-" + activity_model + \
      "-dt-" + "{:d}".format(dt) + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nz-" + str(nz) + \
      "-ncells-" + str((nx + 1)*(ny + 1)*(nz + 1)) + \
      "-nsteps-" + str(nsteps) + \
      "-reltol-" + "{:.{}e}".format(smart_equlibrium_reltol, 1) + \
      "-smart"
tag_conv = "-" + activity_model + \
      "-dt-" + "{:d}".format(dt) + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nz-" + str(nz) + \
      "-ncells-" + str((nx + 1)*(ny + 1)*(nz + 1)) + \
      "-nsteps-" + str(nsteps) + \
      "-conv"
folder_results = 'results/demo-chemicaltransport-odml'

# The seconds spent on equilibrium and transport calculations per time step
time_steps = np.linspace(0, nsteps, nsteps)

timings_equilibrium_smart = np.zeros(nsteps)  # using conventional equilibrium solver
timings_equilibrium_conv = np.zeros(nsteps)  # using smart equilibrium solver
timings_transport = np.zeros(nsteps)

# The counter of full chemical equilibrium training calculations each time step
learnings = np.zeros(nsteps)

def print_test_summary():

    print("Activity model  :", activity_model)
    print("ODML retol      :", smart_equlibrium_reltol)
    print("Number of DOFs  : {} = ({} + 1) x ({} + 1) x ({} + 1)".format((nx+1)*(ny+1)*(nz+1), nx, ny, nz))
    print("Number of steps :", nsteps)
    print("Time step       : {} s".format(dt))

print_test_summary()

# Initialise the mesh
# mesh = fire.UnitIntervalMesh(nx)
# mesh = fire.UnitCubeMesh(nx, ny, nz)
# mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=True)
mesh = fire.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
V = fire.FunctionSpace(mesh, "CG", 1)
ndofs = V.dof_count

# Initialise the chemical editor
editor = rkt.ChemicalEditor()
if activity_model == "pitzer-full":
    editor.addAqueousPhaseWithElements('H O Na Cl Ca Mg C') \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "pitzer-selected-species":
    #editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--") \
    editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3-- CaCl+ Ca(HCO3)+ MgCl+ Mg(HCO3)+") \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "hkf-full":
    editor.addAqueousPhaseWithElements('H O Na Cl Ca Mg C')
elif activity_model == "hkf-selected-species":
    editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--")
elif activity_model == "dk-full":
    editor.addAqueousPhaseWithElements('H O Na Cl Ca Mg C') \
        .setChemicalModelDebyeHuckel()
editor.addMineralPhase("Quartz")
editor.addMineralPhase("Calcite")
editor.addMineralPhase("Dolomite")

# Initialise the chemical system
system = rkt.ChemicalSystem(editor)

# Define equilibrium options
equilibrium_options = rkt.EquilibriumOptions()
# Define smart equilibrium options
smart_equilibrium_options = rkt.SmartEquilibriumOptions()
smart_equilibrium_options.reltol = smart_equlibrium_reltol
smart_equilibrium_options.amount_fraction_cutoff = amount_fraction_cutoff
smart_equilibrium_options.mole_fraction_cutoff = mole_fraction_cutoff

# Initialize the array for documenting the cells, where smart prediction happened
smart_cells = np.empty((nsteps, ndofs))
# Initialize the list for documenting the results of the reactive transport on each time step
rt_results = [rok.ChemicalTransportResult()] * (nsteps)

# List of name of species and elements we track
species_out = ["Ca++", "Mg++", "Calcite", "Dolomite", "CO2(aq)", "HCO3-", "Cl-", "H2O(l)"]
elements_out = ["H", "O", "C", "Ca", "Mg", "Na", "Cl"]

# List of functions representing species and elements we track
n_out = [fire.Function(V, name=name) for name in species_out]
b_out = [fire.Function(V, name=name) for name in elements_out]

def make_results_folders(use_smart_equilibrium_solver):

    if use_smart_equilibrium_solver:
        folder_results_ = folder_results + tag_smart
    else:
        folder_results_ = folder_results + tag_conv

    file_species_amounts = fire.File(folder_results_ + "/species-amounts.pvd")
    file_element_amounts = fire.File(folder_results_ + "/element-amounts.pvd")
    file_porosity = fire.File(folder_results_ + "/porosity.pvd")

    return file_species_amounts, file_element_amounts, file_porosity

def run_transport(use_smart_equilibrium):

    # Define the initial condition of the reactive transport modeling problem
    problem_ic = rkt.EquilibriumProblem(system)
    problem_ic.setTemperature(T)
    problem_ic.setPressure(P)
    problem_ic.add("H2O", 1.0, "kg")
    problem_ic.add("O2", 1.0, "umol")
    problem_ic.add("NaCl", 0.7, "mol")
    problem_ic.add("CaCO3", 10, "mol")
    problem_ic.add("SiO2", 10, "mol")
    problem_ic.add("MgCl2", 1e-10, "mol")

    # Define the boundary condition of the reactive transport modeling problem
    problem_bc = rkt.EquilibriumProblem(system)
    problem_bc.setTemperature(T)
    problem_bc.setPressure(P)
    problem_bc.add("H2O", 1.0, "kg")
    problem_bc.add("O2", 1.0, "umol")
    problem_bc.add("NaCl", 0.90, "mol")
    problem_bc.add("MgCl2", 0.05, "mol")
    problem_bc.add("CaCl2", 0.01, "mol")
    problem_bc.add("CO2", 0.75, "mol")

    # Calculate the equilibrium states for the initial and boundary conditions
    state_ic = rkt.equilibrate(problem_ic)
    state_bc = rkt.equilibrate(problem_bc)

    # Scale the volumes of the phases in the initial condition such that their sum is 1 m3
    state_ic.scalePhaseVolume("Aqueous", 0.1, "m3")
    state_ic.scalePhaseVolume("Quartz", 0.882, "m3")
    state_ic.scalePhaseVolume("Calcite", 0.018, "m3")

    # Scale the volume of the boundary equilibrium state to 1 m3
    state_bc.scaleVolume(1.0)

    # Initialise the chemical field
    field = rok.ChemicalField(system, V)
    field.fill(state_ic)

    # Create the output files in the folders corresponding to the selected method
    file_species_amounts, file_element_amounts, file_porosity = make_results_folders(use_smart_equilibrium)

    # Initialize the chemical transport options
    transport_options = rok.ChemicalTransportOptions()
    transport_options.use_smart_equilibrium = use_smart_equilibrium
    transport_options.equilibrium = equilibrium_options
    transport_options.smart_equilibrium = smart_equilibrium_options

    # Initialize the chemical transport solver
    transport = rok.ChemicalTransportSolver(field)
    transport.setOptions(transport_options)
    transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
    transport.setVelocity([v])
    transport.setDiffusion([D])
    transport.initialize(field)

    t = 0.0
    step = 0

    if use_smart_equilibrium: bar = IncrementalBar('Reactive transport with the ODML algorithm:', max=nsteps)
    else: bar = IncrementalBar('Reactive transport with the conventional algorithm:', max=nsteps)

    while step < nsteps:

        # For each function representing selected species, output its molar amounts from the `field` instance
        for f in n_out:
            f.assign(field.speciesAmount(f.name()))

        # For each function representing selected elements, output its molar amounts from the `field` instance
        for f in b_out:
            f.assign(field.elementAmountInPhase(f.name(), "Aqueous"))

        # Write selected species, elements, and porosity to the files
        file_species_amounts.write(*n_out)
        file_element_amounts.write(*b_out)
        file_porosity.write(field.porosity())

        # Perform one transport step from `t` to `t + dt`
        transport.step(field, dt)

        # Save the reactive transport results on each time step
        rt_results[step] = transport.result

        # Collect intermediate results
        # ----------------------------------------------------------------------------------------------------------
        timings_transport[step] = transport.result.seconds

        # If the ODML is chosen save the results into the smart cells arrays with True/ False and to learning arrays
        if use_smart_equilibrium:
            smart_cells[step] = transport.result.smart_equilibrium.smart_predictions.vector()[:]
            learnings[step] = ndofs - np.count_nonzero(smart_cells[step])
            timings_equilibrium_smart[step] = transport.result.seconds_equilibrium
            #print("timings_equilibrium_smart = ", timings_equilibrium_smart)
            #print("learnings = ", learnings)
        else:
            timings_equilibrium_conv[step] = transport.result.seconds_equilibrium
            #print("timings_equilibrium_conv = ", timings_equilibrium_conv)

        #print("timings_transport = ", timings_transport)
        # Update the current time
        step += 1
        t += dt

        bar.next()
    bar.finish()

    if use_smart_equilibrium:
        transport.equilibrium.outputClusterInfo()
# --------------------------------------------------------------------------
# Run reactive transport with the ODML algorithm
# --------------------------------------------------------------------------


start_rt = time.time()

use_smart_equilibrium = True
run_transport(use_smart_equilibrium)

timing_rt_smart = time.time() - start_rt
#print("Timings equilibrium smart: ", timings_equilibrium_smart)
#print("Timings transport: ", timings_transport)
#print("Learnings: ", learnings)
print("Total learnings: {} out of {} ( {:<5.2f}% )".format(int(np.sum(learnings)), ndofs * nsteps, int(np.sum(learnings)) / ndofs / nsteps * 100))

# --------------------------------------------------------------------------
# Run reactive transport with the conventional algorithm
# --------------------------------------------------------------------------

start_rt = time.time()

use_smart_equilibrium = False
run_transport(use_smart_equilibrium)

timing_rt_conv = time.time() - start_rt
#print("Timings equilibrium conv = ", timings_equilibrium_conv)
#print("Timings transport: ", timings_transport)

# --------------------------------------------------------------------------
# Analyze and plot results
# --------------------------------------------------------------------------

print("Total speedup       : ", timing_rt_conv/timing_rt_smart)
print("Chem. calc. speedup : ", np.sum(timings_equilibrium_conv) / np.sum(timings_equilibrium_smart))
print("")
plots_folder_results = folder_results + "-plots" + tag_smart
import os; os.system('mkdir -p ' + plots_folder_results)
# Plotting of the number of the learning:
plt.plot_on_demand_learning_countings_mpl(time_steps, learnings, plots_folder_results)
# Plot with the CPU time comparison and speedup:
step = 1
plt.plot_computing_costs_mpl(time_steps, (timings_equilibrium_conv, timings_equilibrium_smart, timings_transport), step, plots_folder_results)
plt.plot_speedups_mpl(time_steps, (timings_equilibrium_conv, timings_equilibrium_smart), step, plots_folder_results)
