import sys
sys.path.insert(2, '/home/skyas/polybox/allanleal-cpp-reactivetransportsolver-demo/build/lib/python3.7/site-packages')
import firedrake as fire
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
D = fire.Constant(0)  # the diffusion coefficient (in units of m2/s)
cL = rok.Constant(1.0) # Dirichlet BC for the transport
T = 25.0 + 273.15  # the temperature (in units of K)
P = 1.01325 * 1e5  # the pressure (in units of Pa)

# Discretization parameters for the reactive transport simulation
lx = 1.6
ly = 1.0
lz = 1
nx = 100  # the number of mesh cells along the x-coordinate
ny = 100  # the number of mesh cells along the y-coordinate
nz = 0  # the number of mesh cells along the y-coordinate
nsteps = 2000  # the number of time steps
cfl = 0.3     # the CFL number to be used in the calculation of time step

# PDE methods
method_flow = "sdhm"
method_transport = 'supg'

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
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nz-" + str(nz) + \
      "-ncells-" + str((nx + 1)*(ny + 1)*(nz + 1)) + \
      "-nsteps-" + str(nsteps) + \
      "-reltol-" + "{:.{}e}".format(smart_equlibrium_reltol, 1) + \
      "-smart"
tag_conv = "-" + activity_model + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nz-" + str(nz) + \
      "-ncells-" + str((nx + 1)*(ny + 1)*(nz + 1)) + \
      "-nsteps-" + str(nsteps) + \
      "-conv"
folder_results = 'results/demo-scavenging-heterogeneous-odml'

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
    print("Number of DOFs  : {} = ({} + 1) x ({} + 1) x ({} + 1)".format((nx+1)*(ny+1)*(nz+1), nx, ny, nz))
    print("Number of steps :", nsteps)

print_test_summary()

# Initialise the mesh
mesh = fire.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
V = fire.FunctionSpace(mesh, "CG", 1)
ndofs = V.dof_count
x, y = rok.SpatialCoordinate(mesh)

# Parameters for the flow simulation
rho = rok.Constant(1000.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
#k = rok.permeability(V, minval=1e-18, len_scale=20)
k = rok.permeability(V)
f = rok.Constant(0.0)  # the source rate in the flow calculation

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(P, "left")
problem.addPressureBC(0.9 * P, "right")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "bottom")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "top")

flow = rok.DarcySolver(problem, method=method_flow)
flow.solve()

# Initialise the chemical editor
db = rkt.Database('supcrt07.xml')
dhModel = rkt.DebyeHuckelParams()
dhModel.setPHREEQC()

editor = rkt.ChemicalEditor(db)

if activity_model == "pitzer-full":
    editor.addAqueousPhaseWithElements('C Ca Cl Fe H K Mg Na O S') \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "pitzer-selected-species":
    #editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--") \
    editor.addAqueousPhase(['Ca(HCO3)+', 'CO3--', 'CaCO3(aq)', 'Ca++', 'CaSO4(aq)', 'CaOH+', 'Cl-',
                            'FeCl++', 'FeCl2(aq)', 'FeCl+', 'Fe++', 'FeOH+', 'FeOH++', 'Fe+++',
                            'H2S(aq)', 'H2(aq)', 'HS-', 'H2O(l)', 'H+', 'OH-', 'HCO3-', 'HSO4-',
                            'KSO4-',  'K+',
                            'Mg++', 'Mg(HCO3)+', 'MgCO3(aq)', 'MgSO4(aq)', 'MgOH+',
                            'Na+', 'NaSO4-',
                            'O2(aq)',
                            'S5--', 'S4--', 'S3--', 'S2--', 'SO4--']) \
        .setChemicalModelPitzerHMW() \
        .setActivityModelDrummondCO2()
elif activity_model == "hkf-full":
    editor.addAqueousPhaseWithElements('C Ca Cl Fe H K Mg Na O S')
elif activity_model == "hkf-selected-species":
    editor.addAqueousPhase(['Ca(HCO3)+', 'CO3--', 'CaCO3(aq)', 'Ca++', 'CaSO4(aq)', 'CaOH+', 'Cl-',
                            'FeCl++', 'FeCl2(aq)', 'FeCl+', 'Fe++', 'FeOH+', 'FeOH++', 'Fe+++',
                            'H2S(aq)', 'H2(aq)', 'HS-', 'H2O(l)', 'H+', 'OH-', 'HCO3-', 'HSO4-',
                            'KSO4-',  'K+',
                            'Mg++', 'Mg(HCO3)+', 'MgCO3(aq)', 'MgSO4(aq)', 'MgOH+',
                            'Na+', 'NaSO4-',
                            'O2(aq)',
                            'S5--', 'S4--', 'S3--', 'S2--', 'SO4--'])
elif activity_model == "dk-full":
    editor.addAqueousPhaseWithElements('C Ca Cl Fe H K Mg Na O S') \
        .setChemicalModelDebyeHuckel()
elif activity_model == "dk-selected-species":
    editor.addAqueousPhase(['Ca(HCO3)+', 'CO3--', 'CaCO3(aq)', 'Ca++', 'CaSO4(aq)', 'CaOH+', 'Cl-',
                            'FeCl++', 'FeCl2(aq)', 'FeCl+', 'Fe++', 'FeOH+', 'FeOH++', 'Fe+++',
                            'H2S(aq)', 'H2(aq)', 'HS-', 'H2O(l)', 'H+', 'OH-', 'HCO3-', 'HSO4-',
                            'KSO4-',  'K+',
                            'Mg++', 'Mg(HCO3)+', 'MgCO3(aq)', 'MgSO4(aq)', 'MgOH+',
                            'Na+', 'NaSO4-',
                            'O2(aq)',
                            'S5--', 'S4--', 'S3--', 'S2--', 'SO4--']) \
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

# Initialize the array for documenting the cells, where smart prediction happened
#smart_cells = np.empty((nsteps, ndofs))
# Initialize the list for documenting the results of the reactive transport on each time step
rt_results = []

# List of name of species and elements we track
species_out = ["H+", "HS-", "S2--", "SO4--", "HSO4-", "H2S(aq)", "Pyrrhotite", "Siderite"]
elements_out = ["C", "Ca", "Cl", "Fe", "H", "K", "Mg", "Na", "O", "S", "Z"]

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
    file_volume = rok.File(folder_results_ + '/volume.pvd')
    file_ph = rok.File(folder_results_ + '/ph.pvd')

    rok.File(folder_results + tag_smart + '/flow.pvd').write(flow.u, flow.p, k)

    return file_species_amounts, file_element_amounts, file_porosity, file_volume, file_ph

def run_transport(use_smart_equilibrium):

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
    problem_ic.pE(8.676)

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
    problem_bc.pE(8.220)


    # Calculate the equilibrium states for the initial and boundary conditions
    state_ic = rkt.equilibrate(problem_ic)
    state_bc = rkt.equilibrate(problem_bc)

    # Scale the volumes of the phases in the initial condition such that their sum is 1 m3
    state_ic.scalePhaseVolume("Aqueous", 0.1, "m3")
    state_ic.scaleVolume(1.0, "m3")

    # Scale the volume of the boundary equilibrium state to 1 m3
    state_bc.scaleVolume(1.0)

    # Initialise the chemical field
    field = rok.ChemicalField(system, V)
    field.fill(state_ic)
    field.update()

    # Set the pressure field to the chemical field
    field.setPressures(flow.p.dat.data) # TODO: should not pressure be sync w.r.t. to the pressure found in the Darcy?

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

    # Initialize the BC of the transport problem
    #bc = rok.DirichletBC(V_transport, cL, 1, method="geometric")
    #transport.setBoundaryConditions([bc])

    max_ux = np.max(flow.u.dat.data[:, 0])
    max_uy = np.max(flow.u.dat.data[:, 1])
    delta_x = lx / nx
    delta_y = ly / nx

    # Define time step according to the velocity
    dt = cfl / max(max_ux / delta_x, max_uy / delta_y)

    # Outputting the results
    print('max(u) = ', np.max(flow.u.dat.data[:, 0]), flush=True)
    print('max(k) = ', np.max(k.dat.data), flush=True)
    print('div(u)*dx =', rok.assemble(rok.div(flow.u) * rok.dx), flush=True)
    print('dt = {} minute'.format(dt / minute), flush=True)

    # Create the output files in the folders corresponding to the selected method
    file_species_amounts, file_element_amounts, file_porosity, file_volume, file_ph \
        = make_results_folders(use_smart_equilibrium)

    t = 0.0
    step = 0

    if use_smart_equilibrium: bar = IncrementalBar('Reactive transport with the ODML algorithm:', max=nsteps)
    else: bar = IncrementalBar('Reactive transport with the conventional algorithm:', max=nsteps)

    while step < nsteps:

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

        #print(f'Elapsed time in step {step}: {time.time() - time_step_start} seconds', flush=True)
        bar.next()
    bar.finish()
    if use_smart_equilibrium:
        transport.equilibrium.outputClusterInfo()

    print("Final time: ", t)
# --------------------------------------------------------------------------
# Run reactive transport with the ODML algorithm
# --------------------------------------------------------------------------

start_rt = time.time()

use_smart_equilibrium = True
run_transport(use_smart_equilibrium)

timing_rt_smart = time.time() - start_rt
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
# --------------------------------------------------------------------------
# Run reactive transport with the conventional algorithm
# --------------------------------------------------------------------------

start_rt = time.time()

use_smart_equilibrium = False
run_transport(use_smart_equilibrium)

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
