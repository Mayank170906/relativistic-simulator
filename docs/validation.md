# Validation methodology

`validate_trajectory` and `validate_dataset` return measured errors rather than a
single opaque boolean. Checks include:

- `p = gamma*m*v`;
- `E = gamma*m*c²`;
- `E² - p²c² = m²c⁴`;
- central finite differences for `dp/dt = F` and `dx/dt = v`;
- trapezoidal integration of `dt/gamma` against proper time;
- initial conditions and `tau <= t`;
- independent RK4 versus analytical trajectory comparison in tests.

The finite-difference checks intentionally use interior central differences. One-sided
edge stencils can amplify floating-point roundoff even for bitwise-constant arrays.