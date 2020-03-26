import firedrake as fire
import reaktoro as rkt
import rok
import numpy as np

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
nx = 25  # the number of mesh cells along the x-coordinate
ny = 25  # the number of mesh cells along the y-coordinate
nz = 0  # the number of mesh cells along the y-coordinate

nsteps = 10  # the number of time steps

D = fire.Constant(1.0e-9)  # the diffusion coefficient (in units of m2/s)
v = fire.Constant([1.0 / week, 0.0])  # the fluid pore velocity (in units of m/s)
dt = 30 * minute  # the time step (in units of s)
T = 60.0 + 273.15  # the temperature (in units of K)
P = 100 * 1e5  # the pressure (in units of Pa)

# Activity model for the aqueous species
# +
activity_model = "hkf-full"
#activity_model = "hkf-selected-species"
#activity_model = "pitzer-full"
#activity_model = "pitzer-selected-species"
#activity_model = "dk-full"
#activity_model = "dk-selected-species"
# -

# --------------------------------------------------------------------------
# Parameters for the ODML algorithm
# --------------------------------------------------------------------------

# +
smart_equlibrium_reltol = 0.001
amount_fraction_cutoff = 1e-14
mole_fraction_cutoff = 1e-14
use_smart_equilibrium_solver = True
# -

tag = "-dt-" + "{:d}".format(dt) + \
      "-nx-" + str(nx) + \
      "-ny-" + str(ny) + \
      "-nz-" + str(nz) + \
      "-nsteps-" + str(nsteps) + \
      "-reltol-" + "{:.{}e}".format(smart_equlibrium_reltol, 1) + \
      "-" + activity_model
folder_results = 'results-odml' + tag

# The seconds spent on equilibrium and transport calculations per time step
time_steps = np.linspace(0, nsteps, nsteps)
timings_equilibrium_smart = np.zeros(nsteps)  # using conventional equilibrium solver
timings_equilibrium_conv = np.zeros(nsteps)  # using smart equilibrium solver
timings_transport = np.zeros(nsteps)

# The counter of full chemical equilibrium training calculations each time step
learnings = np.zeros(nsteps)

# Initialise the mesh
# mesh = fire.UnitIntervalMesh(nx)
# mesh = fire.UnitCubeMesh(nx, ny, nz)
mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=True)
V = fire.FunctionSpace(mesh, "CG", 1)
ndofs = V.dof_count

# Initialise the database
database = rkt.Database("supcrt98.xml")

# Initialise the chemical editor
editor = rkt.ChemicalEditor(database)
editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--")
editor.addMineralPhase("Quartz")
editor.addMineralPhase("Calcite")
editor.addMineralPhase("Dolomite")

# Initialise the chemical system
system = rkt.ChemicalSystem(editor)

# Define the initial condition of the reactive transport modeling problem
problem_ic = rkt.EquilibriumProblem(system)
problem_ic.setTemperature(T)
problem_ic.setPressure(P)
problem_ic.add("H2O", 1.0, "kg")
problem_ic.add("NaCl", 0.7, "mol")
problem_ic.add("CaCO3", 10, "mol")
problem_ic.add("SiO2", 10, "mol")

# Define the boundary condition of the reactive transport modeling problem
problem_bc = rkt.EquilibriumProblem(system)
problem_bc.setTemperature(T)
problem_bc.setPressure(P)
problem_bc.add("H2O", 1.0, "kg")
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


equilibrium_options = rkt.EquilibriumOptions()
smart_equilibrium_options = rkt.SmartEquilibriumOptions()
smart_equilibrium_options.reltol = smart_equlibrium_reltol
smart_equilibrium_options.amount_fraction_cutoff = amount_fraction_cutoff
smart_equilibrium_options.mole_fraction_cutoff = mole_fraction_cutoff

smart_cells_per_step = np.zeros(ndofs)
smart_cells = np.zeros((nsteps, ndofs))

species_out = ["Ca++", "Mg++", "Calcite", "Dolomite", "CO2(aq)", "HCO3-", "Cl-", "H2O(l)"]
elements_out = ["H", "O", "C", "Ca", "Mg", "Na", "Cl"]

n_out = [fire.Function(V, name=name) for name in species_out]
b_out = [fire.Function(V, name=name) for name in elements_out]


# Create the output file
file_species_amounts = fire.File("results-test/demo-chemicaltransport/species-amounts.pvd")
file_element_amounts = fire.File("results-test/demo-chemicaltransport/element-amounts.pvd")
file_porosity = fire.File("results-test/demo-chemicaltransport/porosity.pvd")

def run_transport(use_smart_equilibrium):

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

    rt_results = [rok.ChemicalTransportResult()] * nsteps

    while step <= nsteps:
        print("Time: {:<5.2f} day ({}/{})".format(t / day, step, nsteps))

        # For each selected species, output its molar amounts
        for f in n_out:
            f.assign(field.speciesAmount(f.name()))

        # For each selected elements, output its molar amounts
        for f in b_out:
            f.assign(field.elementAmountInPhase(f.name(), "Aqueous"))

        file_species_amounts.write(*n_out)
        file_element_amounts.write(*b_out)
        file_porosity.write(field.porosity())

        # Perform one transport step from `t` to `t + dt`
        transport.step(field, dt)

        rt_results[step] = transport.result

        # Update the current time
        step += 1
        t += dt
