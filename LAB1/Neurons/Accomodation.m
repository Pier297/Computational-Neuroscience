% Parameters
a = 0.02;
b = 1;
c = -55;
d = 4;

p_1 = 5;
p_2 = 140;

% Initial configuration
u = -65;
w = -16;

% Define the time of the simulation and the time discretization
T_max = 400;
delta_t = 0.5;
time = 0:delta_t:T_max;

% Create the current for each time step of the simulation
I = create_I(T_max, delta_t);

% Run the simulation
[U, W] = Custom_Izhikevich(u, w, a, b, c, d, T_max, delta_t, I, p_1, p_2);

% Plot the results
plot_membrane_potential(U, I, delta_t, T_max);
plot_phase_space(U, W);

% ---------------------------------------------------- %
function plot_membrane_potential(U, I, delta_t, T_max)
    figure
    % Plot the membrane potential vs time
    subplot(2,1,1)
    plot(0:delta_t:T_max, U)
    xlabel('Time (t)')
    ylabel('Membrane potential (u)')
    title('Membrane potential dynamics')
    subplot(2,1,2)

    % Plot the current vs time
    plot(0:delta_t:T_max, I)
    title('current I')
    xlabel('Time (t)')
    ylabel('current (I)')
    saveas(gcf, strcat('Neurons/Results/', mfilename, '_Membrane_Potential.png'))
end

function plot_phase_space(U, W)
    figure
    % plot phase space
    plot(U, W)
    xlabel('Membrane potential (u)')
    ylabel('Recovery variable (w)')
    title('Phase portrait')
    saveas(gcf, strcat('Neurons/Results/', mfilename, '_Phase_portrait.png'))
end

% creates the array of current applied to the neuron at a certain
% time during the simulation (Post synaptic potential)
function I = create_I(T_max, delta_t)
    I = [];
    for t = 0:delta_t:T_max
        if t < 200
            I(end + 1) = t / 25;
        elseif t < 300
            I(end + 1) = 0;
        elseif t < 312.5
            I(end + 1) = (t - 300)/12.5 * 4;
        else
            I(end + 1) = 0;
        end
    end
end

% Slighly modified version of the Izhikevich model needed to simulate this
% neuronal spike behaviour.
function [U, W] = Custom_Izhikevich(u_0, w_0, a, b, c, d, T_max, delta_t, I, p_1, p_2)
    u = u_0;
    w = w_0;
    U = [];
    W = [];
    iter = 1;
    for t = 0:delta_t:T_max
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
    r = a * (b * (u + 65));
end

function [X_n_plus_1, Y_n_plus_1] = leap_frog(X_n, Y_n, f, g, h, a, b, I, p_1, p_2)
    X_n_plus_1 = X_n + h * f(X_n, Y_n, I, p_1, p_2);
    Y_n_plus_1 = Y_n + h * g(X_n_plus_1, Y_n, a, b);
end