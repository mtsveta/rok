import sys
# The lines above are necessary to make sure that the `rok` library is fetch of
sys.path.remove('/home/skyas/polybox/rok')
sys.path.insert(1, '/home/skyas/work/rok-min-working-example')
print(sys.path)

import numpy as np
import reaktoro as rkt
import rok
import time

# Auxiliary time-related constants
second = 1
minute = 60
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# Thermodynamical parameters for the reactive transport simulation
D = rok.Constant(1.0e-9)  # the diffusion coefficient (in units of m2/s)
T = 60.0 + 273.15  # the temperature (in units of K)
P_left = 100 * 1e5  # the pressure (in units of Pa) on the left boundary
P_right = 1e-1 * P_left # the pressure on the right boundary
#P_right = 1e-4 * P_left # the pressure on the right boundary #TODO: for {1e-1, 1e-2, 1e-3} * P the pressure remains positive
P = P_left

# Discretization parameters for the reactive transport simulation
lx = 1.6
ly = 1.0
nx = 100  # the number of mesh cells along the x-coordinate
ny = 100  # the number of mesh cells along the y-coordinate
nsteps = 1000  # the number of time steps
tend = 1 * day  # the final time (in units of s)
cfl = 0.3      # the CFL number to be used in the calculation of time step

# PDE methods for the flow
method_flow = "sdhm"

# PDE method for the transport
method_transport = "supg"

# The path to where the result files are output
resultsdir = f"results/demo-reactiveflow-mwe/mesh-{nx}x{ny}-cfl-{cfl}-flow-{method_flow}-transport-{method_transport}/"

# Initialise the mesh
mesh = rok.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
x_coords = mesh.coordinates.dat.data[:, 0]
y_coords = mesh.coordinates.dat.data[:, 1]
print(f"x of size {len(x_coords)} =", x_coords)
print(f"y of size {len(y_coords)} =", y_coords)

# Initialize the function spaces
V = rok.FunctionSpace(mesh, "CG", 1)
# Number of degrees of freedom in the functional space V
ndofs = V.dof_count

# Parameters for the flow and transport simulation
rho = rok.Constant(997.0)  # water density (in units of kg/m3)
mu = rok.Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = rok.permeability(V)
f = rok.Constant(0.0)  # the source rate in the flow calculation
D = rok.Constant(1.0e-9)  # the diffusion coefficient (in units of m2/s)

# -------------------------------------------------------------------------------------------------------------------- #
# Flow problem
# -------------------------------------------------------------------------------------------------------------------- #

# Initialize the Darcy flow solver
problem = rok.DarcyProblem(mesh)
problem.setFluidDensity(rho)
problem.setFluidViscosity(mu)
problem.setRockPermeability(k)
problem.setSourceRate(f)
problem.addPressureBC(P, "left")
problem.addPressureBC(P_right, "right")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "bottom")
problem.addVelocityComponentBC(rok.Constant(0.0), "y", "top")

flow = rok.DarcySolver(problem, method=method_flow)
flow.solve()
rok.File(resultsdir + "flow.pvd").write(flow.u, flow.p, k)

print("len(flow.p.dat.data) =", len(flow.p.dat.data))
print('max(P[0:ndofs]) =', np.max(flow.p.dat.data[0:ndofs]), flush=True)
print('min(P[0:ndofs]) =', np.min(flow.p.dat.data[0:ndofs]), flush=True)
print('max(P) =', np.max(flow.p.dat.data), flush=True)
print('min(P) =', np.min(flow.p.dat.data), flush=True)
print('P_left  =', P_left)
print('P_right =', P_right)
print('max(P) < P_left  :', np.max(flow.p.dat.data) < P_left)
print('min(P) > P_right :', np.min(flow.p.dat.data) > P_right)

# -------------------------------------------------------------------------------------------------------------------- #
# Chemical problem
# -------------------------------------------------------------------------------------------------------------------- #

# Initialise the database
database = rok.Database("supcrt98.xml")

# Initialise the chemical editor
editor = rok.ChemicalEditor(database)
editor.addAqueousPhase("H2O(l) H+ OH- Na+ Cl- Ca++ Mg++ HCO3- CO2(aq) CO3--")
editor.addMineralPhase("Quartz")
editor.addMineralPhase("Calcite")
editor.addMineralPhase("Dolomite")

# Initialise the chemical system
system = rok.ChemicalSystem(editor)

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
problem_ic.add('MgCO3', 1.0, 'umol')

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
field.update()

# Set the pressure field to the chemical field
pressures = np.zeros(ndofs)
for i, x, y in zip(np.linspace(0, ndofs-1, num=ndofs, dtype=int), x_coords, y_coords):
    pressures[i] = flow.p.at([x, y])
    if pressures[i] < 0:
        print(f"{i}: p({x}, {y}) = {pressures[i]}")

print("len(pressures) =", len(pressures))
print("max(pressures) =", max(pressures))
print("min(pressures) =", min(pressures))
print('P_left  =', P_left)
print('P_right =', P_right)
print('max(pressures) < P_left  :', np.max(pressures) < P_left)
print('min(pressures) > P_right :', np.min(pressures) > P_right)

# # Auxiliary function space
# V0 = fire.FunctionSpace(mesh, "DG", 1)
# # Auxiliary peace-wise constant function
# p0 = fire.Function(V0)
# print(p0.project(flow.p))
# print("p0 =", p0.dat.data)
# print("len(p0) =", len(p0.dat.data))
# print("max(p0) =", max(p0.dat.data))
# print("min(p0) =", min(p0.dat.data))
# input()

field.setPressures(flow.p.dat.data) # TODO: should not pressure be sync w.r.t. to the pressure found in the Darcy?

# Initialize the chemical transport solver
transport = rok.ChemicalTransportSolver(field, method=method_transport)
transport.addBoundaryCondition(state_bc, 1)  # 1 means left side in a rectangular mesh
transport.setVelocity([flow.u])
transport.setDiffusion([D])

out_species = ["Ca++", "Mg++", "Calcite", "Dolomite", "CO2(aq)", "HCO3-", "Cl-", "H2O(l)"]
out_elements = ["H", "O", "C", "Ca", "Mg", "Na", "Cl"]

nout = [rok.Function(V, name=name) for name in out_species]
bout = [rok.Function(V, name=name) for name in out_elements]

# Create the output file
file_species_amounts = rok.File(resultsdir + "species-amounts.pvd")
file_element_amounts = rok.File(resultsdir + "element-amounts.pvd")
file_porosity = rok.File(resultsdir + "porosity.pvd")
file_volume = rok.File(resultsdir + "volume.pvd")
file_ph = rok.File(resultsdir + "ph.pvd")

t = 0.0
step = 0

max_ux = np.max(flow.u.dat.data[:, 0])
max_uy = np.max(flow.u.dat.data[:, 1])
delta_x = lx / nx
delta_y = ly / nx

print("dx = ", delta_x)
print("dy = ", delta_y)
print("dofs = ", ndofs)

dt = cfl / max(max_ux / delta_x, max_uy / delta_y)

print("max(u) = ", np.max(flow.u.dat.data[:, 0]))
print("max(k) = ", np.max(k.dat.data))
print("div(u)*dx =", rok.assemble(rok.div(flow.u) * rok.dx))
print("dt = {} minute".format(dt / minute))

input()

start_time = time.time()

while t < tend and step < nsteps:
    elapsed_time = (time.time() - start_time) / hour
    final_time = elapsed_time * (tend / t - 1) if t != 0.0 else 0.0

    print(
        "Progress at step {}: {:.2f} hour ({:.2f}% of {:.2f} days), elapsed time is {:.2f} hour (estimated to end in {:.2f} hours)".format(
            step, t / hour, t / tend * 100, tend / day, elapsed_time, final_time
        )
    )

    if step % 10 == 0:
        # For each selected species, output its molar amounts
        for f in nout:
            f.assign(field.speciesAmount(f.name()))

        # For each selected species, output its molar amounts
        for f in bout:
            f.assign(field.elementAmountInPhase(f.name(), "Aqueous"))

        file_species_amounts.write(*nout)
        file_element_amounts.write(*bout)
        file_porosity.write(field.porosity())
        file_volume.write(field.volume())
        file_ph.write(field.pH())

    # Perform one transport step from `t` to `t + dt`
    transport.step(field, dt)

    # rho.assign(field.densities()[0])

    # Update the current time
    t += dt
    step += 1
