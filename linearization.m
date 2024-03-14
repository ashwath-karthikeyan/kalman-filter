syms a psi p_x p_y theta v phi L dt
syms g_ref_x g_ref_y

% Define state and control vectors
x = [p_x; p_y; theta; v; phi];
u = [a; psi];

% System dynamics
px_dot = v*cos(theta);
py_dot = v*sin(theta);
theta_dot = v*tan(phi)/L;
v_dot = a;
phi_dot = psi;
f = [px_dot; py_dot; theta_dot; v_dot; phi_dot];

% Observation model
g = [cos(theta), -sin(theta); sin(theta), cos(theta)] * [g_ref_x; g_ref_y] + [p_x; p_y];

% Additional observation model
h = [v; theta_dot; g];

% Jacobians
F_t = jacobian(f, x);
G_t = jacobian(f, u); % Corrected to use jacobian of f with respect to u
H_t = jacobian(h, x);

% Display the results
F_t, G_t, H_t