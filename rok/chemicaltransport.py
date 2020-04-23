from .transport import TransportSolver

import firedrake as fire
import reaktoro as rkt
import numpy as np
import time as timer


class ChemicalDirichletBC:
    def __init__(self, function_space, state, boundary):
        self.state = state.clone()
        self.values = fire.Function(function_space)
        self.dofs = function_space.boundary_nodes(boundary, "topological")
        self.dirichlet = fire.DirichletBC(
            function_space, self.values, boundary, method="topological"
        )

    def elementDirichletBC(self, ielement, ispecies, porosity):
        bval = self.state.elementAmountInSpecies(ielement, ispecies)
        phi = porosity.dat.data[self.dofs]
        self.values.dat.data[self.dofs] = phi * bval
        return self.dirichlet

    def speciesDirichletBC(self, ispecies, porosity):
        nval = self.state.speciesAmount(ispecies)
        phi = porosity.dat.data[self.dofs]
        self.values.dat.data[self.dofs] = phi * nval
        return self.dirichlet


class ChemicalTransportResult(object):

    class EquilibriumResult(object):
        def __init__(self):
            # The number of iterations for equilibrium calculations performed at every degree of freedom
            self.iterations = []

            # The elapsed seconds for equilibrium calculations performed at every degree of freedom
            self.seconds = []

    class SmartEquilibriumResult(object):
        def __init__(self):
            # The number of iterations for equilibrium calculations performed at every degree of freedom
            self.smart_preditions = []

            # The number of iterations for equilibrium calculations performed at every degree of freedom
            self.iterations = []

            # The elapsed seconds for equilibrium calculations performed at every degree of freedom
            self.seconds = []

    class KineticsResult(object):
        def __init__(self):
            # The number of time-steps for kinetic calculations performed at every degree of freedom
            self.timesteps = []

            # The elapsed seconds for kinetic calculations performed at every degree of freedom
            self.seconds = []

    def __init__(self):
        # The flag that indicates if the chemical transport calculations succeeded
        self.succeeded = False

        # The total time in seconds of the chemical transport calculation
        self.seconds = 0

        # The total time in seconds spent on equilibrium calculations
        self.seconds_equilibrium = 0

        # The total time in seconds spent on finite element assembly operations
        self.seconds_assemble = 0

        # The total time in seconds spent on linear systems
        self.seconds_linear_systems = 0

        # The result of the equilibrium calculations
        self.smart_equilibrium = ChemicalTransportResult.SmartEquilibriumResult()

        # The result of the equilibrium calculations
        self.equilibrium = ChemicalTransportResult.EquilibriumResult()

        # The result of the kinetic calculations
        self.kinetics = ChemicalTransportResult.KineticsResult()


class ChemicalTransportOptions(object):
    def __init__(self):
        self.use_smart_equilibrium = False
        self.equilibrium = rkt.EquilibriumOptions()
        self.smart_equilibrium = rkt.SmartEquilibriumOptions()

    def setSmartEquilibriumOptions(self, options):
        self.smart_equilibrium.reltol = options.reltol
        self.smart_equilibrium.mole_fraction_cutoff = options.mole_fraction_cutoff
        self.smart_equilibrium.amount_fraction_cutoff = options.amount_fraction_cutoff

    def setEquilibriumOptions(self, options):
        self.equilibrium.hessian = options.hessian
        self.equilibrium.optimum.tolerance = options.optimum.tolerance


class ChemicalTransportSolver(object):
    def __init__(self, field, method="supg"):
        self.method = method
        self.system = field.system()
        self.partition = field.partition()
        self.num_species = self.system.numSpecies()
        self.num_elements = self.system.numElements()
        self.num_fluid_phases = len(self.partition.indicesFluidPhases())
        self.num_solid_phases = len(self.partition.indicesSolidPhases())
        self.velocity = [fire.Constant(0.0) for i in range(self.num_fluid_phases)]
        self.diffusion = [fire.Constant(0.0) for i in range(self.num_fluid_phases)]
        self.source = [fire.Constant(0.0) for i in range(self.num_fluid_phases)]
        self.boundary_conditions = []
        self.initialized = False
        self.solver_is_initialized = False
        self.options = ChemicalTransportOptions()

        # Initialize the function space used in the ChemicalField instance
        self.function_space = field.functionSpace()

        # Initialize the number of degrees-of-freedom in the current process
        self.num_dofs = self.function_space.dof_count

        # Initialize the ChemicalTransportResult instance
        self.result = ChemicalTransportResult()

        # TODO: why all these fields are firedrake functions
        self.result.equilibrium.iterations = fire.Function(self.function_space)
        self.result.equilibrium.seconds = fire.Function(self.function_space)

        self.result.smart_equilibrium.iterations = fire.Function(self.function_space)
        self.result.smart_equilibrium.seconds = fire.Function(self.function_space)
        self.result.smart_equilibrium.smart_predictions = fire.Function(self.function_space)

        self.result.kinetics.timesteps = fire.Function(self.function_space)
        self.result.kinetics.seconds = fire.Function(self.function_space)

        # TODO: why is this repetition?
        self.result.equilibrium._iterations = np.empty(self.num_dofs)
        self.result.equilibrium._seconds = np.empty(self.num_dofs)
        self.result.smart_equilibrium._iterations = np.empty(self.num_dofs)
        self.result.smart_equilibrium._seconds = np.empty(self.num_dofs)
        self.result.smart_equilibrium._smart_preditions = np.empty(self.num_dofs)
        self.result.kinetics._timesteps = np.empty(self.num_dofs)
        self.result.kinetics._seconds = np.empty(self.num_dofs)

        self.result.equilibrium.iterations.rename(
            "EquilibriumIterationsPerDOF", "EquilibriumIterationsPerDOF"
        )
        self.result.equilibrium.seconds.rename(
            "EquilibriumSecondsPerDOF", "EquilibriumSecondsPerDOF"
        )
        self.result.kinetics.timesteps.rename("KineticsTimeStepsPerDOF", "KineticsTimeStepsPerDOF")
        self.result.kinetics.seconds.rename("KineticsSecondsPerDOF", "KineticsSecondsPerDOF")

    def setOptions(self, transport_options):
        self.options.use_smart_equilibrium = transport_options.use_smart_equilibrium
        self.options.setEquilibriumOptions(transport_options.equilibrium)
        self.options.setSmartEquilibriumOptions(transport_options.smart_equilibrium)

    def setVelocity(self, velocity):
        assert type(velocity) is list, "Expecting a list of velocity fields for each fluid phase."
        assert len(velocity) == self.num_fluid_phases, (
            "There are %d fluid phases, but only %d velocity fields were given."
            % (self.num_fluid_phases, len(velocity))
        )
        self.velocity = velocity
        self.initialized = False

    def setDiffusion(self, diffusion):
        assert (
            type(diffusion) is list
        ), "Expecting a list of diffusion coefficients for each fluid phase."
        assert len(diffusion) == self.num_fluid_phases, (
            "There are %d fluid phases, but only %d diffusion coefficients were given."
            % (self.num_fluid_phases, len(diffusion))
        )
        self.diffusion = diffusion
        self.initialized = False

    def setEquilibriumSolver(self, solver):
        # Initialize the chemical equilibrium solver
        if self.options.use_smart_equilibrium:
            self.equilibrium = solver
        else:
            self.equilibrium = solver
        self.initialized = False
        self.solver_is_initialized = True

    def setSource(self, source):
        self.source = source
        self.initialized = False

    def setPartition(self, partition):
        self.partition = partition

    def addBoundaryCondition(self, state, boundary):
        self.boundary_conditions.append((state, boundary))

    def setTemperatures(self, temperatures):
        if type(temperatures) in [int, float]:
            temperatures = [temperatures] * self.num_dofs
        assert hasattr(temperatures, "__len__"), "Expecting a list of temperatures."
        assert len(temperatures) == self.num_dofs, (
            "Expecting a list of \
            temperatures of length %d, but given list has length %d."
            % (self.num_dofs, len(temperatures))
        )
        for i in range(self.num_dofs):
            self.states[i].setTemperature(temperatures[i])

    def setPressures(self, pressures):
        if type(pressures) in [int, float]:
            pressures = [pressures] * self.num_dofs
        assert hasattr(pressures, "__len__"), "Expecting a list of pressures."
        assert len(pressures) == self.num_dofs, (
            "Expecting a list of \
            pressures of length %d, but given list has length %d."
            % (self.num_dofs, len(pressures))
        )
        for i in range(self.num_dofs):
            self.states[i].setPressure(pressures[i])

    def initialize(self, field):

        self.initialized = True

        # Initialize zero step
        self.steps = 0

        # Test parameter
        self.skipped_equilibrium_cacl = 0

        # Check if the user provided any boundary conditions
        if self.boundary_conditions is []:
            RuntimeError(
                "Failed to initialize ChemicalTransportSolver. \
                No boundary conditions have been provided."
            )

        # Initialize the ChemicalDirichletBC instances
        self.bcs = [
            ChemicalDirichletBC(self.function_space, state, boundary)
            for (state, boundary) in self.boundary_conditions
        ]

        # The auxiliary Function instance used for transport steps
        self.u = fire.Function(self.function_space)

        # The auxiliary Function instance used for outputting
        self.output = fire.Function(self.function_space)

        if self.options.use_smart_equilibrium:
            # Initialize the chemical equilibrium solver
            if not self.solver_is_initialized:
                self.equilibrium = rkt.SmartEquilibriumSolver(self.partition)
            self.equilibrium.setOptions(self.options.smart_equilibrium)
        else:
            # Initialize the chemical equilibrium solver
            if not self.solver_is_initialized:
                self.equilibrium = rkt.EquilibriumSolver(self.partition)
            self.equilibrium.setOptions(self.options.equilibrium)

        # Initialize the indices of the equilibrium and kinetic species
        self.ispecies_e = self.partition.indicesEquilibriumSpecies()
        self.ispecies_k = self.partition.indicesKineticSpecies()

        # Initialize the indices of the fluid and solid species
        self.ispecies_f = self.partition.indicesFluidSpecies()
        self.ispecies_s = self.partition.indicesSolidSpecies()

        # Initialize the indices of the equilibrium-fluid and equilibrium-solid species
        # self.ispecies_ef = [sorted(set(self.ispecies_e) & set(indices)) for indices in self.ispecies_f]
        self.ispecies_ef = [sorted(set(self.ispecies_e) & set(self.ispecies_f))]
        self.ispecies_es = sorted(set(self.ispecies_e) & set(self.ispecies_s))

        # Initialize the indices of the kinetic-fluid and kinetic-solid species
        # self.ispecies_kf = [sorted(set(self.ispecies_k) & set(indices)) for indices in self.ispecies_f]
        self.ispecies_kf = [sorted(set(self.ispecies_k) & set(self.ispecies_f))]
        self.ispecies_ks = sorted(set(self.ispecies_k) & set(self.ispecies_s))

        # Initialize the indices of fluid and solid phases
        self.iphases_f = self.partition.indicesFluidPhases()
        self.iphases_s = self.partition.indicesSolidPhases()

        # Initialize the number of fluid and solid phases
        self.num_fluid_phases = len(self.iphases_f)
        self.num_solid_phases = len(self.iphases_s)

        # Initialize the arrays of element amounts for each fluid phase in
        # the equilibrium-fluid partition for each degree-of-freedom in the function space
        self.bef = [
            np.zeros((self.num_elements, self.num_dofs)) for i in range(self.num_fluid_phases)
        ]

        # Initialize the array of element amounts in the equilibrium
        # partition for each degree-of-freedom in the function space
        self.be = np.zeros((self.num_elements, self.num_dofs))

        # Get the dolfin Function's for the saturation fields of fluid phases
        self.saturations = field.saturations()

        # Get the dolfin Function for the porosity field
        self.porosity = field.porosity()

        # Define the dolfin forms for the pore velocity of each fluid phase
        self.pore_velocity = [
            self.velocity[i] / (self.porosity * self.saturations[i])
            for i in range(self.num_fluid_phases)
        ]

        # Create transport solvers for each fluid phase
        self.transport = [TransportSolver(method=self.method) for i in range(self.num_fluid_phases)]

        # Set the pore velocities and diffusion coefficients for each transport solver
        for i in range(self.num_fluid_phases):
            self.transport[i].setVelocity(self.pore_velocity[i])
            self.transport[i].setDiffusion(self.diffusion[i])

        # Initialize the porosity and saturations fields
        field.update()

        # Initialize the element amounts in the fluid phases of the equilibrium-fluid partition
        for i in range(self.num_fluid_phases):
            field.elementAmountsInSpecies(self.ispecies_ef[i], self.bef[i])

    def elementDirichletBC(self, ielement, iphase):
        ispecies = self.ispecies_ef[iphase]  # the indices of equilibrium species in phase `iphase`
        return [bc.elementDirichletBC(ielement, ispecies, self.porosity) for bc in self.bcs]

    def speciesDirichletBC(self, ispecies):
        return [bc.speciesDirichletBC(ispecies, self.porosity) for bc in self.bcs]

    def transportEquilibriumElementsInFluidPhases(self, field, dt):
        # Iterate over all elements in all fluid phases in the equilibrium-fluid
        # partition and tranport them
        for iphase in range(self.num_fluid_phases):

            # Iterate over all elements in the current fluid phase and tranport therm
            for ielement in range(self.num_elements):
                # The Dirichlet boundary conditions element `ielement` in fluid phase `iphase`
                bcs = self.elementDirichletBC(ielement, iphase)

                # Set the boundary conditions in the tranport solver w.r.t. fluid phase `iphase`
                self.transport[iphase].setBoundaryConditions(bcs)

                # Set initial condition for the transport equation of the current element
                self.u.vector()[:] = self.bef[iphase][ielement]

                # Transport the current element of the equilibrium-fluid partition
                self.transport[iphase].step(self.u, dt)

                # Extract the result to the array of element amounts `bef`
                self.bef[iphase][ielement][:] = self.u.vector().get_local()

    def transportKineticSpeciesInFluidPhases(self, field, dt):
        pass

    def equilibrate(self, field):

        # Compute the positive and negative part of the molar amounts of the elements in the equilibrium-fluid partition
        bef_positive = [np.maximum(x, 0) for x in self.bef]
        bef_negative = [np.minimum(x, 0) for x in self.bef]

        # Start timing the equilibrate step
        tbegin = timer.time()

        # Define auxiliary bindings
        states = field.states()
        iterations = self.result.equilibrium._iterations
        seconds = self.result.equilibrium._seconds
        smart_predictions = self.result.smart_equilibrium._smart_preditions

        # Calculate the element amounts in the equilibrium partition.
        # Compute the contribution from the equilibrium-solid partition
        field.elementAmountsInSpecies(self.ispecies_es, self.be)

        # Compute the contribution from the equilibrium-fluid partition
        for bef in bef_positive:
            self.be += bef

        # options = rkt.EquilibriumOptions()
        #         options.epsilon = 1e-30
        #         options.hessian = EquilibriumHessian.Exact
        #         options.optimum.tolerance = 1e-8
        #         options.optimum.output.active = True

        # self.equilibrium.setOptions(options)

        # Compute the equilibrium states
        for k in range(self.num_dofs):

            # Get temperature and pressure at current dof
            T = states[k].temperature()
            P = states[k].pressure()

            # Perform the equilibrium calculation at current degree of freedom
            result = self.equilibrium.solve(states[k], T, P, self.be[:, k])

            if self.options.use_smart_equilibrium:
                # Check if the smart equilibrium calculation was successful
                if not (result.learning.gibbs_energy_minimization.optimum.succeeded or result.estimate.accepted):
                    '''
                    b = [
                        (elem.name(), amount)
                        for elem, amount in zip(self.system.elements(), self.be[:, k])
                    ]
                    raise RuntimeError(
                        "Failed to calculate equilibrium state at dof (%d), under "
                        "temperature %f K, pressure %f Pa, and element molar amounts %s."
                        % (k, states[k].temperature(), states[k].pressure(), str(b))
                    )
                    '''
                    self.skipped_equilibrium_cacl += 1
                    print(f"Failed to calculate equilibrium state at dof {k}, step {self.steps}. "
                          f"Totally {self.skipped_equilibrium_cacl} equilibrium calculations skipped.")
                    if k > 0: states[k].setSpeciesAmounts(states[k - 1].speciesAmounts())


            else:
                # Check if the equilibrium calculation was successful
                if not result.optimum.succeeded:
                    self.skipped_equilibrium_cacl += 1
                    print(f"Failed to calculate equilibrium state at dof {k}, step {self.steps}. "
                          f"Totally {self.skipped_equilibrium_cacl} equilibrium calculations skipped.")
                    if k > 0: states[k].setSpeciesAmounts(states[k-1].speciesAmounts())
                '''
                # Check if the equilibrium calculation was successful
                if not result.optimum.succeeded:
                    # Restart calculations
                    n_tmp = np.array(states[k].speciesAmounts())
                    n_tmp[self.ispecies_e] = 0.0
                    states[k].setSpeciesAmounts(n_tmp)
                    # Make an initial guess
                    #states[k].setSpeciesAmounts(0.0)
                    #result = self.equilibrium.approximate(states[k], T, P, self.be[:, k])
                    #print("result.optimum.succeeded = ", result.optimum.succeeded)
                    result = self.equilibrium.solve(states[k], T, P, self.be[:, k])
                    if not result.optimum.succeeded:
                        b = [(elem.name(), amount) for elem, amount in zip(self.system.elements(), self.be[:, k])]
                        raise RuntimeError(
                            "Failed to calculate equilibrium state at dof (%d), under "
                            "temperature %f K, pressure %f Pa, and element molar amounts %s."
                            % (k, states[k].temperature(), states[k].pressure(), str(b))
                        )
                '''
            # Store the statistics of the equilibrium calculation
            if self.options.use_smart_equilibrium:
                smart_predictions[k] = result.estimate.accepted
                iterations[k] = result.learning.gibbs_energy_minimization.optimum.iterations
            else:
                iterations[k] = result.optimum.iterations
            seconds[k] = result.timing.solve

        # Update the element amounts in the fluid phases of the equilibrium-fluid partition
        for i in range(self.num_fluid_phases):
            field.elementAmountsInSpecies(self.ispecies_ef[i], self.bef[i])
            self.bef[i] += bef_negative[i]

        # Extract the calculation statistics from the auxiliary ndarray members
        if self.options.use_smart_equilibrium:
            self.result.smart_equilibrium.iterations.vector()[:] = iterations
            self.result.smart_equilibrium.seconds.vector()[:] = seconds
            self.result.smart_equilibrium.smart_predictions.vector()[:] = smart_predictions
        else:
            self.result.equilibrium.iterations.vector()[:] = iterations
            self.result.equilibrium.seconds.vector()[:] = seconds

        # Total time spent on performing equilibrium calculations
        self.result.seconds_equilibrium = timer.time() - tbegin

    def react(self, field, dt):
        pass

    def step(self, field, dt):
        # Check if the chemical transport solver has been initialized
        if not self.initialized:
            self.initialize(field)

        # Start timing the chemical transport step calculation
        tbegin = timer.time()

        # Transport the elements in the equilibrium-fluid partition
        self.transportEquilibriumElementsInFluidPhases(field, dt)

        # Transport the species in the kinetic-fluid partition
        self.transportKineticSpeciesInFluidPhases(field, dt)

        # Stop timing the transport time of the reactive transport
        self.result.seconds = timer.time() - tbegin

        # Equilibrate the equilibrium-solid and equilibrium-fluid partitions
        self.equilibrate(field)

        # React the equilibrium and kinetic partitions over a time of `dt`
        self.react(field, dt)

        # Update the porosity and fluid phases saturation fields
        field.update()

        # Compute the time elapsed for the chemical transport step
        self.result.time = timer.time() - tbegin

        # Increase step
        self.steps += 1

    def elementAmountInPhase(self, element, phase):
        ielement = self.system.indexElement(element)
        iphase = self.system.indexPhase(phase)
        out = fire.Function(self.function_space)
        out.vector()[:] = self.bef[iphase][ielement]
        out.rename(element, element)
        return out
