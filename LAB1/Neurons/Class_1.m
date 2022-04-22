% Parameters
a = 0.02;
b = -0.1;
c = -55;
d = 6;

p_1 = 4.1;
p_2 = 108;

% Initial configuration
u = -60;
w = b * u;

% Define the time of the simulation and the time discretization
T_max = 300;
delta_t = 0.25;
time = 0:delta_t:T_max;
T1 = 30;

% Create the current for each time step of the simulation
I = create_I(T_max, delta_t, T1);

% Run the simulation
[U, W] = Izhikevich(u, w, a, b, c, d, T_max, delta_t, I, p_1, p_2);

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
function I = create_I(T_max, delta_t, T1)
    I = [];
    for t = 0:delta_t:T_max
        if t > T1
            I(end + 1) = 0.075 * (t - T1);
        else
            I(end + 1) = 0;
        end
    end
end