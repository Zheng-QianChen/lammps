# -*- CMake -*- file for testing LAMMPS commands

add_executable(test_simple_commands test_simple_commands.cpp)
add_library(noplugin MODULE noplugin.cpp)
set_target_properties(noplugin PROPERTIES PREFIX "" SUFFIX ".so")

# tests for the plugin command require the PLUGIN package.
# With a STATIC LAMMPS library the plugins must be built without embedding a
# full copy of LAMMPS (loading such a plugin corrupts memory and aborts at
# program exit, see the comment in examples/plugins/CMakeLists.txt), which
# requires the COMPILE_ONLY generator expression from CMake 3.27; the test
# executable must then export its symbols so the loaded plugins can resolve
# them.  Windows is not affected and keeps linking the full library.
if(PKG_PLUGIN)
    if(BUILD_SHARED_LIBS OR (CMAKE_SYSTEM_NAME STREQUAL "Windows"))
    add_subdirectory(${LAMMPS_DIR}/examples/plugins ${CMAKE_BINARY_DIR}/build-plugins)
    add_dependencies(test_simple_commands plugins noplugin)
    target_compile_definitions(test_simple_commands PRIVATE -DLMP_PLUGIN)
    if((NOT BUILD_SHARED_LIBS) AND (NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows")))
      set_target_properties(test_simple_commands PROPERTIES ENABLE_EXPORTS TRUE)
    endif()
  else()
    message(WARNING "Skipping tests for the plugin command: they require setting BUILD_SHARED_LIBS=on")
  endif()
endif()

target_link_libraries(test_simple_commands PRIVATE lammps GTest::GMock)
add_test(NAME SimpleCommands COMMAND test_simple_commands)

# the geturl test downloads from the network, which takes long and varies with
# the connection, so it is a separate executable with the "slow" label
add_executable(test_geturl_command test_geturl_command.cpp)
target_link_libraries(test_geturl_command PRIVATE lammps GTest::GMock)
add_test(NAME GeturlCommand COMMAND test_geturl_command)
set_tests_properties(GeturlCommand PROPERTIES LABELS "slow")
cmake_path(SET LAMMPS_PLUGIN_BIN_DIR NORMALIZE ${CMAKE_BINARY_DIR})
if(CMAKE_CONFIG_TYPE)
  cmake_path(APPEND LAMMPS_PLUGIN_BIN_DIR ${CMAKE_CONFIG_TYPE})
endif()
set_tests_properties(SimpleCommands PROPERTIES
                     ENVIRONMENT "LAMMPS_PLUGIN_DIR=${LAMMPS_PLUGIN_BIN_DIR}")
add_kokkos_test(NAME SimpleCommands COMMAND test_simple_commands
                ENVIRONMENT "LAMMPS_PLUGIN_DIR=${LAMMPS_PLUGIN_BIN_DIR}")

add_executable(test_lattice_region test_lattice_region.cpp)
target_link_libraries(test_lattice_region PRIVATE lammps GTest::GMock)
add_test(NAME LatticeRegion COMMAND test_lattice_region)
add_kokkos_test(NAME LatticeRegion COMMAND test_lattice_region)

add_executable(test_fix_spring_chunk test_fix_spring_chunk.cpp)
target_link_libraries(test_fix_spring_chunk PRIVATE lammps GTest::GMock)
add_test(NAME FixSpringChunk COMMAND test_fix_spring_chunk)
add_kokkos_test(NAME FixSpringChunk COMMAND test_fix_spring_chunk)

# tests for the inertial spin dynamics styles of the SPIN package
if(PKG_SPIN)
  add_executable(test_tspin test_tspin.cpp)
  target_link_libraries(test_tspin PRIVATE lammps GTest::GMock)
  add_test(NAME TSpin COMMAND test_tspin)
  set_tests_properties(TSpin PROPERTIES LABELS "spin")
  add_mpi_test(NAME TSpinTemperatureMPI NUM_PROCS 4
               COMMAND $<TARGET_FILE:test_tspin>
               --gtest_filter=TSpinTest.TemperatureUsesActiveSpinsAndGroup:TSpinTest.TemperatureCountsZeroModulusAndRecountsMasses)
  if(TEST TSpinTemperatureMPI)
    set_property(TEST TSpinTemperatureMPI APPEND PROPERTY LABELS "spin")
  endif()
  add_mpi_test(NAME TSpinIntegratorCompatibilityMPI NUM_PROCS 4
               COMMAND $<TARGET_FILE:test_tspin>
               --gtest_filter=TSpinTest.MixedSpinIntegratorsAreRejected:TSpinTest.SingleSpinIntegratorKeepsDipoleLatticeForces)
  if(TEST TSpinIntegratorCompatibilityMPI)
    set_property(TEST TSpinIntegratorCompatibilityMPI APPEND PROPERTY LABELS "spin")
  endif()
endif()

add_executable(test_kspace_auto_slab test_kspace_auto_slab.cpp)
target_compile_definitions(test_kspace_auto_slab PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_kspace_auto_slab PRIVATE lammps GTest::GMock)
add_test(NAME KSpaceAutoSlab COMMAND test_kspace_auto_slab)
add_kokkos_test(NAME KSpaceAutoSlab COMMAND test_kspace_auto_slab)

add_executable(test_kspace_group_group_slab test_kspace_group_group_slab.cpp)
target_compile_definitions(test_kspace_group_group_slab PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_kspace_group_group_slab PRIVATE lammps GTest::GMock)
add_test(NAME KSpaceGroupGroupSlab COMMAND test_kspace_group_group_slab)
add_kokkos_test(NAME KSpaceGroupGroupSlab COMMAND test_kspace_group_group_slab)

if(PKG_QMMM-XTB)
  add_executable(test_kspace_group_potential test_kspace_group_potential.cpp)
  target_include_directories(test_kspace_group_potential PRIVATE ${LAMMPS_SOURCE_DIR}/KSPACE
                                                                  ${LAMMPS_SOURCE_DIR}/QMMM-XTB)
  target_link_libraries(test_kspace_group_potential PRIVATE lammps GTest::GMock)
  if(FFT_SINGLE)
    target_compile_definitions(test_kspace_group_potential PRIVATE -DFFT_SINGLE)
  endif()
  add_test(NAME KSpaceGroupPotential COMMAND test_kspace_group_potential)
endif()

add_executable(test_groups test_groups.cpp)
target_link_libraries(test_groups PRIVATE lammps GTest::GMock)
add_test(NAME Groups COMMAND test_groups)
add_kokkos_test(NAME Groups COMMAND test_groups)

add_executable(test_regions test_regions.cpp)
target_link_libraries(test_regions PRIVATE lammps GTest::GMock)
add_test(NAME Regions COMMAND test_regions)
add_kokkos_test(NAME Regions COMMAND test_regions)

add_executable(test_info test_info.cpp)
target_link_libraries(test_info PRIVATE lammps GTest::GMock)
add_test(NAME InfoCommand COMMAND test_info)
add_kokkos_test(NAME InfoCommand COMMAND test_info)

add_executable(test_thermo test_thermo.cpp)
target_link_libraries(test_thermo PRIVATE lammps GTest::GMock)
add_test(NAME ThermoCommands COMMAND test_thermo)
add_kokkos_test(NAME ThermoCommands COMMAND test_thermo)

add_executable(test_delete_atoms test_delete_atoms.cpp)
target_link_libraries(test_delete_atoms PRIVATE lammps GTest::GMock)
add_test(NAME DeleteAtoms COMMAND test_delete_atoms)
add_kokkos_test(NAME DeleteAtoms COMMAND test_delete_atoms)

add_executable(test_set_property test_set_property.cpp)
target_link_libraries(test_set_property PRIVATE lammps GTest::GMock)
add_test(NAME SetProperty COMMAND test_set_property)
add_kokkos_test(NAME SetProperty COMMAND test_set_property)

if(PKG_COLLOID)
  add_executable(test_pair_lubricate_poly test_pair_lubricate_poly.cpp)
  target_link_libraries(test_pair_lubricate_poly PRIVATE lammps GTest::GMock)
  add_test(NAME PairLubricatePoly COMMAND test_pair_lubricate_poly)
  add_mpi_test(NAME PairLubricatePolyMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_pair_lubricate_poly>)

  add_executable(test_pair_brownian_poly test_pair_brownian_poly.cpp)
  target_link_libraries(test_pair_brownian_poly PRIVATE lammps GTest::GMock)
  add_test(NAME PairBrownianPoly COMMAND test_pair_brownian_poly)
  add_mpi_test(NAME PairBrownianPolyMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_pair_brownian_poly>)

  if(PKG_OPENMP)
    add_test(NAME PairLubricatePolyOMP COMMAND test_pair_lubricate_poly)
    add_test(NAME PairBrownianPolyOMP COMMAND test_pair_brownian_poly)
    set_tests_properties(PairLubricatePolyOMP PairBrownianPolyOMP PROPERTIES
                         ENVIRONMENT "LAMMPS_ACCELERATOR_ARGS=-sf omp -pk omp 2")
  endif()
endif()

add_executable(test_pair_ldd test_pair_ldd.cpp)
target_link_libraries(test_pair_ldd PRIVATE lammps GTest::GMock)
add_test(NAME PairLDD COMMAND test_pair_ldd)
add_kokkos_test(NAME PairLDD COMMAND test_pair_ldd)

add_executable(test_pair_body_granular test_pair_body_granular.cpp)
target_link_libraries(test_pair_body_granular PRIVATE lammps GTest::GMock)
add_test(NAME PairBodyGranular COMMAND test_pair_body_granular)

if(PKG_AMOEBA)
  add_executable(test_pair_amoeba test_pair_amoeba.cpp)
  target_compile_definitions(test_pair_amoeba PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_pair_amoeba PRIVATE lammps GTest::GMock)
  if(FFT_SINGLE)
    target_compile_definitions(test_pair_amoeba PRIVATE -DFFT_SINGLE)
  endif()
  add_test(NAME PairAmoeba COMMAND test_pair_amoeba)
endif()

add_executable(test_labelmap test_labelmap.cpp)
target_compile_definitions(test_labelmap PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_labelmap PRIVATE lammps GTest::GMock)
add_test(NAME Labelmap COMMAND test_labelmap)
add_kokkos_test(NAME Labelmap COMMAND test_labelmap)

if(PKG_AMOEBA)
  add_executable(test_pair_hippo test_pair_hippo.cpp)
  target_compile_definitions(test_pair_hippo PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_pair_hippo PRIVATE lammps GTest::GMock)
  add_test(NAME PairHippo COMMAND test_pair_hippo)
endif()

add_executable(test_compute_frenkel test_compute_frenkel.cpp)
target_link_libraries(test_compute_frenkel PRIVATE lammps GTest::GMock)
add_test(NAME ComputeFrenkel COMMAND test_compute_frenkel)
add_kokkos_test(NAME ComputeFrenkel COMMAND test_compute_frenkel)

add_executable(test_compute_peratom_group test_compute_peratom_group.cpp)
target_link_libraries(test_compute_peratom_group PRIVATE lammps GTest::GMock)
target_compile_definitions(test_compute_peratom_group PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR}/../force-styles/tests)
add_test(NAME ComputePeratomGroup COMMAND test_compute_peratom_group)
add_kokkos_test(NAME ComputePeratomGroup COMMAND test_compute_peratom_group)
add_mpi_test(NAME ComputePeratomGroupMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_compute_peratom_group>)
if(PKG_OPENMP)
  add_test(NAME ComputePeratomGroupOMP COMMAND test_compute_peratom_group)
  set_tests_properties(ComputePeratomGroupOMP PROPERTIES
                       ENVIRONMENT "LAMMPS_ACCELERATOR_ARGS=-sf omp -pk omp 2")
endif()

add_executable(test_variables test_variables.cpp)
target_link_libraries(test_variables PRIVATE lammps GTest::GMock)
add_test(NAME Variables COMMAND test_variables)
add_kokkos_test(NAME Variables COMMAND test_variables)

add_executable(test_kim_commands test_kim_commands.cpp)
if(KIM_EXTRA_UNITTESTS)
  if(CURL_FOUND)
    target_compile_definitions(test_kim_commands PRIVATE -DKIM_EXTRA_UNITTESTS)
  else()
    message(FATAL_ERROR "CURL not found. Enabling KIM extra unit tests requires to have libcurl installed.")
  endif()
endif()
target_link_libraries(test_kim_commands PRIVATE lammps GTest::GMock)
add_test(NAME KimCommands COMMAND test_kim_commands)

add_executable(test_reset_atoms test_reset_atoms.cpp)
target_compile_definitions(test_reset_atoms PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_reset_atoms PRIVATE lammps GTest::GMock)
add_test(NAME ResetAtoms COMMAND test_reset_atoms)
add_kokkos_test(NAME ResetAtoms COMMAND test_reset_atoms)

if(PKG_MOLECULE)
  add_executable(test_compute_global test_compute_global.cpp)
  target_compile_definitions(test_compute_global PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_compute_global PRIVATE lammps GTest::GMock)
  add_test(NAME ComputeGlobal COMMAND test_compute_global)
  add_kokkos_test(NAME ComputeGlobal COMMAND test_compute_global)

  add_executable(test_compute_chunk test_compute_chunk.cpp)
  target_compile_definitions(test_compute_chunk PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_compute_chunk PRIVATE lammps GTest::GMock)
  add_test(NAME ComputeChunk COMMAND test_compute_chunk)
  add_kokkos_test(NAME ComputeChunk COMMAND test_compute_chunk)

  add_executable(test_fix_halt test_fix_halt.cpp)
  target_compile_definitions(test_fix_halt PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_fix_halt PRIVATE lammps GTest::GMock)
  add_test(NAME FixHalt COMMAND test_fix_halt)
  add_kokkos_test(NAME FixHalt COMMAND test_fix_halt)
endif()

if(PKG_MC AND PKG_MOLECULE)
  add_executable(test_fix_bond_create test_fix_bond_create.cpp)
  target_link_libraries(test_fix_bond_create PRIVATE lammps GTest::GMock)
  add_test(NAME FixBondCreate COMMAND test_fix_bond_create)
endif()

if(PKG_PHONON)
  add_executable(test_phonon_commands test_phonon_commands.cpp)
  target_link_libraries(test_phonon_commands PRIVATE lammps GTest::GMock)
  add_test(NAME PhononCommands COMMAND test_phonon_commands)
  add_kokkos_test(NAME PhononCommands COMMAND test_phonon_commands)
endif()

if(PKG_VORONOI)
  add_executable(test_compute_voronoi test_compute_voronoi.cpp)
  target_link_libraries(test_compute_voronoi PRIVATE lammps GTest::GMock)
  add_test(NAME ComputeVoronoi COMMAND test_compute_voronoi)
  add_mpi_test(NAME ComputeVoronoiMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_compute_voronoi>)
endif()

if(PKG_ML-PACE)
  add_executable(test_compute_pace test_compute_pace.cpp)
  target_compile_definitions(test_compute_pace PRIVATE -DTEST_INPUT_FOLDER=${CMAKE_CURRENT_SOURCE_DIR}/../force-styles/tests)
  target_link_libraries(test_compute_pace PRIVATE lammps GTest::GMock)
  add_test(NAME ComputePACE COMMAND test_compute_pace)
  add_mpi_test(NAME ComputePACEMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_compute_pace>)
endif()

if(PKG_REAXFF)
  add_executable(test_reaxff_output test_reaxff_output.cpp)
  target_link_libraries(test_reaxff_output PRIVATE lammps GTest::GMock)
  add_test(NAME ReaxFFOutput COMMAND test_reaxff_output)
  set_tests_properties(ReaxFFOutput PROPERTIES
                       ENVIRONMENT "LAMMPS_POTENTIALS=${LAMMPS_POTENTIALS_DIR}")
  add_kokkos_test(NAME ReaxFFOutput COMMAND test_reaxff_output
                  ENVIRONMENT "LAMMPS_POTENTIALS=${LAMMPS_POTENTIALS_DIR}")
endif()

if(PKG_REAXFF OR PKG_QEQ)
  add_executable(test_qeq_subgroup test_qeq_subgroup.cpp)
  target_link_libraries(test_qeq_subgroup PRIVATE lammps GTest::GMock)
  add_test(NAME QEqSubgroup COMMAND test_qeq_subgroup)
  add_kokkos_test(NAME QEqSubgroup COMMAND test_qeq_subgroup)
endif()

add_executable(test_mpi_load_balancing test_mpi_load_balancing.cpp)
target_link_libraries(test_mpi_load_balancing PRIVATE lammps GTest::GMock)
add_mpi_test(NAME MPILoadBalancing NUM_PROCS 4 COMMAND $<TARGET_FILE:test_mpi_load_balancing>)

if(PKG_SPIN)
  add_executable(test_fix_nve_spin_mpi test_fix_nve_spin_mpi.cpp)
  target_link_libraries(test_fix_nve_spin_mpi PRIVATE lammps GTest::GMock)
  add_mpi_test(NAME FixNVESpinMPI NUM_PROCS 4 COMMAND $<TARGET_FILE:test_fix_nve_spin_mpi>)
  if(TEST FixNVESpinMPI)
    set_tests_properties(FixNVESpinMPI PROPERTIES LABELS "mpi;spin")
  endif()
endif()

add_executable(test_restart_settings test_restart_settings.cpp)
target_link_libraries(test_restart_settings PRIVATE lammps GTest::GMock)
add_test(NAME RestartSettings COMMAND test_restart_settings)
add_kokkos_test(NAME RestartSettings COMMAND test_restart_settings)

add_executable(test_input_validation test_input_validation.cpp)
target_link_libraries(test_input_validation PRIVATE lammps GTest::GMock)
add_test(NAME InputValidation COMMAND test_input_validation)
add_kokkos_test(NAME InputValidation COMMAND test_input_validation)

# tests for fix atom/swap of the MC package, also with multiple MPI ranks
if(PKG_MC)
  add_executable(test_fix_atom_swap test_fix_atom_swap.cpp)
  target_compile_definitions(test_fix_atom_swap PRIVATE -DTEST_POTENTIALS_FOLDER=${LAMMPS_POTENTIALS_DIR})
  target_link_libraries(test_fix_atom_swap PRIVATE lammps GTest::GMock)
  add_test(NAME FixAtomSwap COMMAND test_fix_atom_swap)
  # The MPI fixture constructs LAMMPS directly and does not consume
  # LAMMPS_ACCELERATOR_ARGS.  Exercise the accelerated path through lmp,
  # including an assertion on the local/full energy comparison.
  # This version always builds lmp; honor BUILD_EXE if a parent defines it.
  if(NOT DEFINED BUILD_EXE)
    set(BUILD_EXE ON)
  endif()
  if(PKG_KOKKOS AND BUILD_EXE)
    # Request one GPU for GPU backends and one thread for host backends.
    if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL OR Kokkos_ENABLE_OPENMPTARGET)
      set(swap_kokkos_cli_args -k on g 1 t 1 -sf kk)
    else()
      set(swap_kokkos_cli_args -k on t 1 -sf kk)
    endif()
    # Reduced precision pair tables do not satisfy the double precision tolerance.
    string(TOLOWER "${KOKKOS_PREC}" swap_kokkos_precision)
    if(swap_kokkos_precision STREQUAL "double")
      set(swap_kokkos_verify_quit [[if "$(f_mc[1]) <= 0 || $(f_mc[4]) >= 1.0e-10" then "quit 1"]])
    else()
      set(swap_kokkos_verify_quit [[if "$(f_mc[1]) <= 0" then "quit 1"]])
    endif()

    # Omitted extension keywords retain full energies and a two-element vector.
    set(swap_kokkos_legacy_input "${CMAKE_CURRENT_BINARY_DIR}/in.fix_atom_swap_legacy_full")
    file(GENERATE OUTPUT "${swap_kokkos_legacy_input}" CONTENT
[=[
units lj
atom_style atomic
lattice sc 0.8
region box block 0 8 0 8 0 8
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 4321
mass * 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.8 1.0
pair_coeff 1 2 0.3 1.0
neighbor 0.3 bin
velocity all create 1.5 7777 loop geom
fix integrate all nve
fix mc all atom/swap 1 5 12345 1.5 types 1 2 ke yes
variable size vector f_mc*0.0+1.0
thermo_style custom step pe f_mc[1] f_mc[2]
run 2
if "$(sum(v_size)) != 2 || $(f_mc[1]) != 10" then "quit 1"
if "$(f_mc[2]) <= 0 || $(f_mc[2]) >= $(f_mc[1])" then "quit 1"
]=])
    add_kokkos_test(NAME FixAtomSwapLegacyFull COMMAND ${LAMMPS_TEST_LAUNCHER} $<TARGET_FILE:lmp>
                    ${swap_kokkos_cli_args}
                    -pk kokkos newton on neigh half gpu/aware off
                    -in "${swap_kokkos_legacy_input}" -log none)

    # lj/cut/kk: valid (newton on, neigh half) and (newton off, neigh full)
    set(swap_kokkos_input "${CMAKE_CURRENT_BINARY_DIR}/in.fix_atom_swap_kokkos")
    set(swap_kokkos_lj_content
[=[
units lj
atom_style atomic
lattice fcc 0.8
region box block 0 8 0 8 0 8
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.35 12345
mass 1 1.0
mass 2 1.8
pair_style lj/cut 3.0
pair_coeff 1 1 1.0 1.0 2.0
pair_coeff 2 2 1.2 1.0 3.0
pair_coeff 1 2 0.7 1.0 2.5
pair_modify shift yes
neighbor 0.5 bin
comm_modify cutoff 3.6
velocity all create 1.5 7777
fix integrate all nve
fix mc all atom/swap 1 2 20261126 1.5 types 1 2 ke yes rmax 1.8 proposal hetero energy verify parallel checkerboard
thermo_style custom step pe f_mc[1] f_mc[2] f_mc[3] f_mc[4]
run 2
write_restart atom_swap_kokkos.restart
run 2
]=])
    set(swap_kokkos_lj_content "${swap_kokkos_lj_content}${swap_kokkos_verify_quit}\n")
    file(GENERATE OUTPUT "${swap_kokkos_input}" CONTENT "${swap_kokkos_lj_content}")
    add_kokkos_test(NAME FixAtomSwap COMMAND ${LAMMPS_TEST_LAUNCHER} $<TARGET_FILE:lmp>
                    ${swap_kokkos_cli_args}
                    -pk kokkos newton on neigh half gpu/aware off
                    -in "${swap_kokkos_input}" -log none)
    add_kokkos_test(NAME FixAtomSwapFull COMMAND ${LAMMPS_TEST_LAUNCHER} $<TARGET_FILE:lmp>
                    ${swap_kokkos_cli_args}
                    -pk kokkos newton off neigh full gpu/aware off
                    -in "${swap_kokkos_input}" -log none)

    # Each dimension must fit at least two checkerboard cells.
    if(PKG_MANYBODY AND EXISTS "${LAMMPS_POTENTIALS_DIR}/CuNi.eam.alloy")
      set(swap_kokkos_eam_input "${CMAKE_CURRENT_BINARY_DIR}/in.fix_atom_swap_kokkos_eam")
      set(swap_kokkos_eam_content
"units metal
atom_style atomic
lattice fcc 3.56
region box block 0 8 0 8 0 8
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.35 12345
mass 1 58.6934
mass 2 63.546
pair_style eam/alloy
pair_coeff * * \"${LAMMPS_POTENTIALS_DIR}/CuNi.eam.alloy\" Ni Cu
neighbor 0.5 bin
comm_modify cutoff 7.0
fix integrate all nve
fix mc all atom/swap 1 1 20261126 1200.0 types 1 2 rmax 3.0 proposal hetero energy verify parallel checkerboard
thermo_style custom step pe f_mc[1] f_mc[2] f_mc[3] f_mc[4]
run 1
${swap_kokkos_verify_quit}
")
      file(GENERATE OUTPUT "${swap_kokkos_eam_input}" CONTENT "${swap_kokkos_eam_content}")
      add_kokkos_test(NAME FixAtomSwapEAM COMMAND ${LAMMPS_TEST_LAUNCHER} $<TARGET_FILE:lmp>
                      ${swap_kokkos_cli_args}
                      -pk kokkos newton on neigh half gpu/aware off
                      -in "${swap_kokkos_eam_input}" -log none
                      ENVIRONMENT "LAMMPS_POTENTIALS=${LAMMPS_POTENTIALS_DIR}")
    endif()

    # MEAM requires newton on and a half neighbor list.
    if(PKG_MEAM AND EXISTS "${LAMMPS_POTENTIALS_DIR}/library.meam" AND EXISTS "${LAMMPS_POTENTIALS_DIR}/SiC.meam")
      set(swap_kokkos_meam_input "${CMAKE_CURRENT_BINARY_DIR}/in.fix_atom_swap_kokkos_meam")
      set(swap_kokkos_meam_content
"units metal
atom_style atomic
lattice diamond 4.36
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.35 12345
mass 1 28.0855
mass 2 12.011
pair_style meam
pair_coeff * * \"${LAMMPS_POTENTIALS_DIR}/library.meam\" Si C \"${LAMMPS_POTENTIALS_DIR}/SiC.meam\" Si C
neighbor 2.0 bin
comm_modify cutoff 6.0
fix integrate all nve
fix mc all atom/swap 1 1 20261126 5000.0 types 1 2 rmax 2.5 proposal hetero energy verify parallel checkerboard
thermo_style custom step pe f_mc[1] f_mc[2] f_mc[3] f_mc[4]
run 1
${swap_kokkos_verify_quit}
")
      file(GENERATE OUTPUT "${swap_kokkos_meam_input}" CONTENT "${swap_kokkos_meam_content}")
      add_kokkos_test(NAME FixAtomSwapMEAM COMMAND ${LAMMPS_TEST_LAUNCHER} $<TARGET_FILE:lmp>
                      ${swap_kokkos_cli_args}
                      -pk kokkos newton on neigh half gpu/aware off
                      -in "${swap_kokkos_meam_input}" -log none
                      ENVIRONMENT "LAMMPS_POTENTIALS=${LAMMPS_POTENTIALS_DIR}")
    endif()

    # Exercise the same checkerboard energy check with two MPI ranks.
    if(BUILD_MPI)
      set(swap_kokkos_mpi_input "${CMAKE_CURRENT_BINARY_DIR}/in.fix_atom_swap_kokkos_mpi2")
      string(REPLACE "run 2\nwrite_restart atom_swap_kokkos.restart\nrun 2\n" "run 1\n"
             swap_kokkos_mpi_content "${swap_kokkos_lj_content}")
      file(GENERATE OUTPUT "${swap_kokkos_mpi_input}" CONTENT "${swap_kokkos_mpi_content}")
      add_mpi_test(NAME FixAtomSwapKokkosMPI2 NUM_PROCS 2
                   COMMAND $<TARGET_FILE:lmp> ${swap_kokkos_cli_args}
                   -pk kokkos newton on neigh half gpu/aware off
                   -in "${swap_kokkos_mpi_input}" -log none)
      if(TEST FixAtomSwapKokkosMPI2)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/FixAtomSwapKokkosMPI2")
        set_tests_properties(FixAtomSwapKokkosMPI2 PROPERTIES
                             WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/FixAtomSwapKokkosMPI2")
        set_property(TEST FixAtomSwapKokkosMPI2 APPEND PROPERTY LABELS "kokkos")
      endif()
    endif()
  endif()
  add_mpi_test(NAME FixAtomSwapMPI2 NUM_PROCS 2 COMMAND $<TARGET_FILE:test_fix_atom_swap>)
  add_mpi_test(NAME FixAtomSwapMPI4 NUM_PROCS 4 COMMAND $<TARGET_FILE:test_fix_atom_swap>)
endif()
