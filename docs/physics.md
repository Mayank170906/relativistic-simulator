# Physics and numerical method

## Model

The package models a massive particle in one spatial dimension in flat Minkowski
spacetime. Units are SI: mass in kg, force in N, position in m, velocity in m/s,
time in s, momentum in kg m/s, and energy in J. The speed of light is the exact
constant `C = 299792458.0 m/s`.

The dynamical law is

\[
\frac{dp}{dt}=F, \qquad p=\gamma m v,
\]

not `F = m a`.

## Exact solution

For initial momentum \(p_0=\gamma_0mv_0\),

\[
p(t)=p_0+Ft,
\quad q=\frac{p}{mc},
\quad \gamma=\sqrt{1+q^2},
\quad v=c\frac{q}{\gamma}.
\]

The total and kinetic energies are

\[
E=\sqrt{m^2c^4+p^2c^2},\qquad
K=\frac{p^2c^2}{E+mc^2}.
\]

For nonzero force, position is evaluated in the cancellation-free form

\[
x(t)=x_0+ct\frac{q+q_0}{\gamma+\gamma_0}.
\]

This is algebraically equivalent to \(x_0+(K-K_0)/F\), but avoids subtracting two
large nearly equal kinetic energies in high-beta, small-impulse cases. For zero force,
\(x=x_0+v_0t\).

Proper time is accumulated from \(d\tau=dt/\gamma\). The implementation uses a
`log1p` rapidity-increment form, avoiding cancellation in
`asinh(q) - asinh(q0)` when the impulse is tiny compared with the initial momentum.

## Numerical safety

- Inputs with `|v| >= c`, `|beta| >= 1`, nonpositive mass, nonfinite values, or invalid
  integration steps raise typed exceptions.
- Values are never silently clipped to the light cone.
- `hypot` is used for square roots of sums of squares.
- `gamma - 1` and `E - mc²` are evaluated through rationalized identities.
- The exact solver is independent of the RK4 integrator used for cross-checking.

## Validity limits

This is not a general relativistic engine. It excludes gravity, curved spacetime,
electromagnetic-field models, radiation reaction, variable mass, quantum effects,
and spacelike/FTL trajectories. Float64 precision also limits how close a represented
state can be to the light cone; such states are rejected if the computed velocity can
no longer be represented as strictly subluminal.