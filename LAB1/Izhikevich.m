
% The Izhikevich model simulates a simple model of spiking neurons.
% Inputs:
%   u_0     : 
%   w_0     : 
%   a       : 
%   b       : 
%   c       : 
%   d       : 
%   T_max   : 
%   delta_t : Time dicretization which 
%   I       : Array of the current I vs time
%   p_1, p_2: Additional parameters
% Outputs:
%   U : Array of the membrane potential (u) vs time
%   W : Array of the recovery variable (w) vs time
function [U, W] = Izhikevich(u_0, w_0, a, b, c, d, T_max, delta_t, I, p_1, p_2)
    u = u_0;
    w = w_0;
    U = [];
    W = [];
    iter = 1;
    for t = 0:delta_t:T_max
        %[u, w] = euler(u, w, @dudt, @dwdt, delta_t, a, b, I(iter), p_1, p_2);
        [u, w] = leap_frog(u, w, @dudt, @dwdt, delta_t, a, b, I(iter), p_1, p_2);
        if u > 30
            U(iter) = 30;
            u = c;
            w = w + d;
        else
            U(iter) = u;
        end
        W(iter) = w;
        iter = iter + 1;
    end
end

% du/dt
function [r] = dudt(u, w, I, p_1, p_2)
    r = 0.04 * u^2 + p_1 * u + p_2 - w + I;
end
% dw/dt
function [r] = dwdt(u, w, a, b)
    r = a * (b * u - w);
end

% Given the current point (x, y) = (X_n, Y_n) and the two derivatives:
% (1) dx/dt = f
% (2) dy/dt = g
% find the next point in which the system evolves into by considering
% a time step of h.
function [X_n_plus_1, Y_n_plus_1] = euler(X_n, Y_n, f, g, h, a, b, I, p_1, p_2)
    X_n_plus_1 = X_n + h * f(X_n, Y_n, I, p_1, p_2);
    Y_n_plus_1 = Y_n + h * g(X_n, Y_n, a, b);
end

function [X_n_plus_1, Y_n_plus_1] = leap_frog(X_n, Y_n, f, g, h, a, b, I, p_1, p_2)
    X_n_plus_1 = X_n + h * f(X_n, Y_n, I, p_1, p_2);
    Y_n_plus_1 = Y_n + h * g(X_n_plus_1, Y_n, a, b);
end