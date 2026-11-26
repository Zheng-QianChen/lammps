.. index:: fix atom/swap
.. index:: fix atom/swap/kk

fix atom/swap command
=====================

Accelerator Variants: *atom/swap/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID atom/swap N X seed T keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* atom/swap = style name of this fix command
* N = invoke this fix every N steps
* X = number of swap attempts per invocation in serial mode, or rounds per
  color in checkerboard mode
* seed = random # seed (positive integer)
* T = scaling temperature of the MC swaps (temperature units)
* one or more keyword/value pairs may be appended to args
* keyword = *types* or *mu* or *ke* or *semi-grand* or *region* or *energy*
  or *rmax* or *proposal* or *parallel*

  .. parsed-literal::

       *types* values = two or more atom types (1-Ntypes or type label)
       *mu* values = chemical potential of swap types (energy units)
       *ke* value = *no* or *yes*
         *no* = no conservation of kinetic energy after atom swaps
         *yes* = kinetic energy is conserved after atom swaps
       *semi-grand* value = *no* or *yes*
         *no* = particle type counts and fractions conserved
         *yes* = semi-grand canonical ensemble, particle fractions not conserved
       *region* value = region-ID
         region-ID = ID of region to use as an exchange/move volume
       *energy* value = *auto* or *local* or *full* or *verify*
         *auto* = use local energy changes if possible, else full energies
         *local* = use local energy changes, stop with an error if not possible
         *full* = compute the full potential energy for every attempt
         *verify* = use local energy changes and compare them to full energies
       *rmax* value = distance
         distance = max distance of two swapped atoms (distance units),
           0.0 = no limit
       *proposal* values = *hetero* or *geometric* keyword args
         *hetero* = select pairs of atoms of different type with
           equal probability
         *geometric* = select pairs of atoms with equal probability
         zero or one keyword/args pair may be appended to *geometric*
         *weights* args = itype jtype w ...
           itype,jtype = two different swap types (1-Ntypes or type label)
           w = relative probability to attempt a selected pair of these types
       *parallel* value = *serial* or *checkerboard*
         *serial* = attempt swaps one after the other
         *checkerboard* = attempt swaps in many cells of a checkerboard
           grid at the same time

Examples
""""""""

.. code-block:: LAMMPS

   fix 2 all atom/swap 1 1 29494 300.0 ke no types 1 2
   fix myFix all atom/swap 100 1 12345 298.0 region my_swap_region types 5 6
   fix SGMC all atom/swap 1 100 345 1.0 semi-grand yes types 1 2 3 mu 0.0 4.3 -5.0
   fix mc all atom/swap 10 20 12345 800.0 types 1 2 rmax 2.5 &
     energy local parallel checkerboard
   fix mc all atom/swap 10 500 12345 800.0 types 1 2 3 &
     proposal geometric weights 1 2 1.0 2 3 0.5

Description
"""""""""""

This fix performs Monte Carlo swaps of atoms of one given atom type with
atoms of the other given atom types.  The specified scaling temperature
*T* is used in the Metropolis criterion dictating swap probabilities.

In serial mode, make *X* swap attempts between atoms of different
types according to a Monte Carlo probability.  Swap candidates must be in
the fix group, must be in the region (if specified), and must be of
one of the listed types. Swaps are attempted between candidates that
are chosen randomly with equal probability among the candidate
atoms. Swaps are not attempted between atoms of the same type since
nothing would happen.

All atoms in the simulation domain can also be moved using regular
time integration displacements (e.g., via :doc:`fix nvt <fix_nh>`),
resulting in a hybrid MC+MD simulation.  In serial mode, *X* MC swap
attempts are made once every *N* MD steps.  A smaller-than-usual timestep
size may be needed when running such a hybrid simulation, especially if the
swapped atoms are not well equilibrated.

.. note::

   To run an MC-only simulation (no MD), you should define no
   time-integration fix, set the :doc:`thermo <thermo>` command to 1,
   set *N* to 1, and set *X* small enough to see the MC evolution of
   the system.  But if *X* is too small, the overhead at the start and
   stop of MC moves each timestep will slow down the simulation.

The *types* keyword is required. At least two atom types must be
specified. If not using *semi-grand*, exactly two atom types are
required, unless the *proposal* or *rmax* keyword is used.

The *ke* keyword can be set to *no* to turn off kinetic energy
conservation for swaps. The default is *yes*, which means that swapped
atoms have their velocities scaled by the ratio of the masses of the
swapped atom types. This ensures that the kinetic energy of each atom is
the same after the swap as it was before the swap, even though the atom
masses have changed.

The *semi-grand* keyword can be set to *yes* to switch to the semi-grand
canonical ensemble as discussed in :ref:`(Sadigh) <Sadigh>`. This means
that the total number of each particle type does not need to be
conserved. The default is *no*, which means that the only kind of swap
allowed exchanges an atom of one type with an atom of a different given
type. In other words, the relative mole fractions of the swapped atoms
remains constant. Whereas in the semi-grand canonical ensemble, the
composition of the system can change. Note that when using *semi-grand*,
atoms in the fix group whose type is not listed in the *types* keyword
are ineligible for attempted conversion. An attempt is made to switch
the selected atom (if eligible) to one of the other listed types with
equal probability.  Acceptance of each attempt depends upon the
Metropolis criterion.

The *mu* keyword allows users to specify chemical potentials. This is
required and allowed only when using *semi-grand*\ .  All chemical
potentials are absolute, so there is one for each swap type listed
following the *types* keyword.  In semi-grand canonical ensemble
simulations the chemical composition of the system is controlled by the
difference in these values. So shifting all values by a constant amount
will have no effect on the simulation.

This command may optionally use the *region* keyword to define swap
volume.  The specified region must have been previously defined with a
:doc:`region <region>` command.  It must be defined with side = *in*\ .
Swap attempts occur only between atoms that are both within the
specified region. Swaps are not otherwise attempted.

You should ensure you do not swap atoms belonging to a molecule, or
LAMMPS will eventually generate an error when it tries to find those
atoms.  LAMMPS will warn you if any of the atoms eligible for swapping
have a non-zero molecule ID, but does not check for this at the time of
swapping.

If not using *semi-grand* this fix checks to ensure all atoms of the
given types have the same atomic charge. LAMMPS does not enforce this in
general, but it is needed for this fix to simplify the swapping
procedure. Successful swaps will swap the atom type and charge of the
swapped atoms. Conversely, when using *semi-grand*, it is assumed that
all the atom types involved in switches have the same charge. Otherwise,
charge would not be conserved. As a consequence, no checks on atomic
charges are performed, and successful switches update the atom type but
not the atom charge. While it is possible to use *semi-grand* with
groups of atoms that have different charges, these charges will not be
changed when the atom types change.  The same applies for systems
with per-atom masses: non *semi-grand* will swap atom masses, but
the masses have to be the same each for the atom types.  When using
*semi-grand* no per-atom masses are changed.

Since this fix computes total potential energies before and after
proposed swaps, even complicated potential energy calculations are
acceptable, including the following:

* long-range electrostatics (:math:`k`-space)
* many body pair styles
* hybrid pair styles (with restrictions)
* EAM pair styles
* triclinic systems

Some fixes have an associated potential energy. Examples of such fixes
include: :doc:`efield <fix_efield>`, :doc:`gravity <fix_gravity>`,
:doc:`addforce <fix_addforce>`, :doc:`langevin <fix_langevin>`,
:doc:`restrain <fix_restrain>`, :doc:`temp/berendsen
<fix_temp_berendsen>`, :doc:`temp/rescale <fix_temp_rescale>`, and
:doc:`wall fixes <fix_wall>`.  For that energy to be included in the
total potential energy of the system (the quantity used when
performing GCMC moves), you **must** enable the :doc:`fix_modify
<fix_modify>` *energy* option for that fix.  The doc pages for
individual :doc:`fix <fix>` commands specify if this should be done.

----------

Local energy changes and parallel sweeps
""""""""""""""""""""""""""""""""""""""""

.. versionadded:: TBD

The keywords *energy*, *rmax*, *proposal*, and *parallel* enable
extensions of this fix.  Without any of these keywords, the fix behaves
exactly as described above: full potential energies are computed for
every attempt and the global output vector has length 2.

The *parallel* keyword defaults to *serial*.  Checkerboard sweeps are
optional and must be selected with *parallel checkerboard*.

The *energy* keyword selects how the energy change of a swap attempt is
computed.  With *full*, the total potential energy of the system is
computed after each attempted swap as described above, which takes a
time proportional to the number of atoms for every attempt.  With
*local*, the energy change is computed from the finite-range
neighborhood of atoms whose energies are affected by the swapped
atoms.  This is exact (up to floating-point round-off).  The evaluation
cost scales with that affected neighborhood rather than with the total
number of atoms, while neighbor-list construction, proposal selection,
and communication of energy contributions still add overhead.  Each MPI
process computes the change of the energies of the atoms it owns, and
these changes are summed over all processes.  With *auto* (the default
when one of the extension keywords is used), local energy changes are
used when possible, and otherwise the fix prints a warning with the
reason and uses full energies.  With *local*, the fix stops with an
error instead.  With *verify*, local energy changes are used, but also
compared to full energy evaluations.  The largest absolute difference
between the local and full energy changes is available as the fourth
element of the output vector.  This is meant for testing and is slower
than *full*.  If the pair style cannot compute a local energy change
during a run, the fix stops with an error for *local* and *verify*,
while with *auto* it prints a warning and uses full energies for the
rest of that invocation.

Local energy changes are currently available for the
pair styles :doc:`lj/cut <pair_lj>`, :doc:`eam, eam/alloy, eam/fs, and
eam/he <pair_eam>`, and :doc:`meam <pair_meam>` (but not MS-MEAM).  They
are also available with *atom/swap/kk* for *lj/cut/kk*, *eam/kk*,
*eam/alloy/kk*, *eam/fs/kk*, *eam/he/kk*, and standard *meam/kk*.
The KOKKOS package accelerates local energy changes and supports the same
serial and multi-rank checkerboard sweeps.  GPU atomic reductions may
introduce floating-point rounding differences in the last bits across runs,
unlike the deterministic serial reduction on the host.  Other accelerated
or derived styles, including *meam/ms/kk*, do not support local energy changes.
These local energy calculations require that the potential energy comes
from the pair style only: no kspace style, no bonded interactions, and no
fixes that contribute to the potential energy (via :doc:`fix_modify energy
yes <fix_modify>`).  On molecular systems, special bond factors other than
1.0 are also not allowed; atomic systems are unaffected by that restriction.
They further require no neighbor list exclusions or include groups, no
*pair_modify tail yes*, no per-atom masses, :doc:`comm_style brick
<comm_style>` with a single communication cutoff, neighbor style *bin* or
*nsq*, and no :doc:`run_style respa <run_style>`.
The ghost atom cutoff must be at
least the larger of *rmax* and the dependency radius, which is the
distance up to which the energy of an atom depends on the type of
another atom.  This is the largest pair cutoff for lj/cut and eam
styles.  For pair style meam, it is :math:`\sqrt{C_{bound}}\, r_c` with
the largest bounding ellipse parameter :math:`C_{bound}` of the
screening function of all pairs of elements.  All atoms that can screen
a pair must then be within the neighbor list cutoff (pair cutoff plus
neighbor skin), otherwise local energy changes are not used.

The *rmax* keyword limits swaps to pairs of eligible atoms within the
distance *rmax* of each other.  It cannot be used with *semi-grand yes*.
The value of *rmax* must not exceed the communication cutoff (see
:doc:`comm_modify cutoff <comm_modify>`), which is also checked during
a run.  The *rmax* keyword cannot be used together with neighbor list
exclusions (see :doc:`neigh_modify exclude <neigh_modify>`), since
pairs of excluded atoms would never be swapped.

The *proposal* keyword selects how pairs of atoms are chosen for swap
attempts with *semi-grand no*.  The default without *proposal* and
*rmax* chooses a random atom of the first and of the second listed type
and requires exactly two types.  With *hetero*, pairs of eligible atoms
of different type (within *rmax* if set) are chosen with equal
probability, and more than two types may be listed.  When *rmax* is
used, the number of such pairs changes with the composition around the
swapped atoms, which is accounted for by a Hastings correction of the
acceptance probability.  With *geometric*, all pairs of eligible atoms
(within *rmax* if set) are chosen with equal probability.  Pairs of
atoms of the same type then result in a null move, which is counted in
the third element of the output vector.  The optional *weights* keyword
of *geometric* selection sets the probability :math:`w/w_{\max}` with
which a selected pair of the two given types is attempted, where
:math:`w_{\max}` is the largest of all weights.  The default weight is
1.0 for all pairs of different types.  If *rmax* is used without the
*proposal* keyword, *hetero* selection is used.

The *parallel* keyword selects how swaps are attempted.  With *serial*
(the default), the *X* swap attempts are made one after the other.  With
*checkerboard*, the periodic simulation box is divided into a grid of
:math:`n_x \times n_y \times n_z` cells with an even number
:math:`n \ge 2` of cells in each dimension.  Each cell is wider than the
larger of *rmax* and the conflict radius, the distance within which two
swaps can affect each other's energy change (twice the dependency radius
for eam and meam styles, the largest pair cutoff for lj/cut).
The grid is shifted by a random offset at each invocation.  The cells
are colored in 8 colors like a 3d checkerboard, so cells of the same
color do not
touch.  Each invocation visits the eight colors once, in forward or
reverse order with equal probability.  For each color, *X* rounds are
made.  In each round, every cell of that color selects at most one pair
of atoms that are both inside the cell with the *hetero* or *geometric*
selection rule, and all these swaps are evaluated and accepted or
rejected at the same time.  These swaps do not affect each other's
energy change.  A cell in which only a single pair can be selected makes
a null move with probability 1/2, which prevents the swap of such a pair
from alternating with a period of two attempts.  Empty cells, same-type
draws, weight-rejected pairs, and those single-pair 1/2 self-loops are
null moves and still count as attempts.  Checkerboard draws are keyed
by the random seed, the timestep, the sweep, the color, the cell, and
the attempt, so all MPI processes make the same decisions for pairs of
atoms owned by different processes.  Each cell counts as one attempt in
each round, so each invocation makes *X* times the total number of cells
attempts.
Checkerboard sweeps require local energy changes, *semi-grand no*,
*rmax* > 0.0, and a 3d box that is periodic in all dimensions and large
enough for at least 2 cells in each dimension.  Otherwise the fix prints
a warning and makes *X* serial swap attempts in that invocation, which
are far fewer attempts than a checkerboard sweep with the same *X*.
Since the box can change during a run, these conditions are checked at
every invocation.

----------

Dump image info
"""""""""""""""

.. versionadded:: 11Feb2026

Fix *atom/swap* supports the *fix* keyword of :doc:`dump image
<dump_image>`.  The fix will pass geometry information about atoms
involved in a swap to *dump image* so that these atoms can be
highlighted in the visualization as additional spheres.  For how
long those additional spheres will be shown depends on the value of the
*vizsteps* setting (default is 1000) which can be changed by using the
:doc:`fix_modify command <fix_modify>`.  If an atom is involved in
multiple swaps, the check on showing the additional graphics depends
on the timestep of its last swap.

The color of the additional spheres is by default that of the atom type
*before* the swap when using color styles "type" or "element".  With
color style "const" the default value of "white" can be changed using
:doc:`dump_modify fcolor <dump_image>`.  The transparency is by default
fully opaque and can be changed with *dump\_modify ftrans*\ .

The *fflag1* setting of *dump image fix* has no effect.

The *fflag2* setting allows you to set the radius of the added
spheres, since the radius is set to zero internally.

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the fix to :doc:`binary restart files
<restart>`.  This includes information about the random number generator
seed, the next timestep for MC exchanges, the number of exchange
attempts and successes, etc.  See the :doc:`read_restart <read_restart>`
command for info on how to re-specify a fix in an input script that
reads a restart file, so that the operation of the fix continues in an
uninterrupted fashion.  The number of null moves and the largest
difference between local and full energy changes (third and fourth
elements of the extended output vector) are not stored in restart files
and start again from 0 after a restart.

.. note::

   For this to work correctly, the timestep must **not** be changed
   after reading the restart with :doc:`reset_timestep
   <reset_timestep>`.  The fix will try to detect it and stop with an
   error.

None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix.

This fix computes a global vector of length 2, which can be accessed
by various :doc:`output commands <Howto_output>`.  The vector values are
the following global cumulative quantities:

  #. swap attempts
  #. swap accepts

.. versionchanged:: TBD

If any of the keywords *energy*, *rmax*, *proposal*, or *parallel* is
used, including *energy full* or *parallel serial*, the vector has
length 4 with these values:

  #. swap attempts (cumulative)
  #. swap accepts (cumulative)
  #. null moves (cumulative count of attempts that do not evaluate a
     different-type swap: same-type draws, weight-rejected pairs,
     empty checkerboard cells, and single-pair cells that take a 1/2
     self-loop)
  #. largest absolute difference between local and full energy changes
     (energy units); 0.0 when *energy verify* is not used; not a
     cumulative sum

The vector values calculated by this fix are "intensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.  The *atom/swap/kk* variant requires the KOKKOS
package, which supports only neighbor style *bin*.  Local energy changes
with *atom/swap/kk* work with the neighbor list settings supported by
the KOKKOS package (*neigh half* with *newton on* or *off*, *neigh full*
with *newton off*); with *meam/kk* they require *newton on* and *neigh
half*.

When this fix is used with a :doc:`hybrid pair style <pair_hybrid>`
system, only swaps between atom types of the same sub-style (or
combination of sub-styles) are permitted.

This fix can be used with systems that have per-atom masses
(e.g. atom style sphere) provided all atoms of the types handled
by this fix have the same mass per type. The fix will check for that.
In case both, per-type and per-atom masses are present, a warning is printed.

Related commands
""""""""""""""""

:doc:`fix nvt <fix_nh>`, :doc:`neighbor <neighbor>`,
:doc:`fix deposit <fix_deposit>`, :doc:`fix evaporate <fix_evaporate>`,
:doc:`delete_atoms <delete_atoms>`, :doc:`fix gcmc <fix_gcmc>`,
:doc:`fix mol/swap <fix_mol_swap>`, :doc:`fix sgcmc <fix_sgcmc>`

Default
"""""""

The option defaults are *ke* = yes, *semi-grand* = no, *mu* = 0.0 for
all atom types, *rmax* = 0.0, and *parallel* = serial.  The default for
*energy* is *full* if none of the keywords *energy*, *rmax*, *proposal*,
or *parallel* is used, and *auto* otherwise.  For canonical swaps without
the *proposal* keyword, a random atom of each of the first two listed
types is selected if *rmax* = 0.0, and *hetero* selection is used otherwise.

----------

.. _Sadigh:

**(Sadigh)** B Sadigh, P Erhart, A Stukowski, A Caro, E Martinez, and
L Zepeda-Ruiz, Phys. Rev. B, 85, 184203 (2012).
