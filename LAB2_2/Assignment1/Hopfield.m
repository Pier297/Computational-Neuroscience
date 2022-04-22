clear;
rng(1);

% ---- Load the input data ----
input_patterns = load('lab2_2_data.mat');

p0 = input_patterns.p0';
p1 = input_patterns.p1';
p2 = input_patterns.p2';

data = [p0; p1; p2];

% ---- Create and Train the Hopfield Network ----
[W, b] = learn(data);

% ---- For each digit, feed the distorted versions to the network ----
noise_levels = {0.05, 0.1, 0.25};
for d = 1:3
    for k = 1:length(noise_levels)
        input = distort_image(data(d, :), noise_levels{k});
        retrieve(data, W, b, input, num2str(d-1), num2str(noise_levels{k}));
    end
end

% Given an input, feed it to the hopfield network and plot the
% reconstructed memory along with a plot of the energy and overlaps
% with all the memories in the network.
function retrieve(data, W, b, input, input_digit, input_error)
    [s, m_t, E_t] = retrieval(W, b, input, data);
    % Get the most probable memory with overlap M with memory (I-1)
    overlaps = m_t(:, end);
    [M, I] = max(overlaps);
    
    fprintf('Input digit %s with noise %s, retrieved digit %d with overlap %f\n', input_digit, input_error, I-1, M);

    % Compute the error w.r.t the optimal memory
    error = immse(data(I, :), s);
    plot_input_and_retrieval(input, s, input_digit, input_error, error, I-1)
    saveas(gcf, ['Assignment1/Results/distorted_', input_digit, '_', input_error, '_reconstructed.png'])

    % Plot E(t)
    figure
    plot(0:size(E_t, 2)-1, E_t)
    xlabel('Time (t)')
    ylabel('E(t)')
    title('Energy function')
    saveas(gcf, ['Assignment1/Results/distorted_', input_digit, '_', input_error, '_energy.png'])

    % Plot overlap function
    figure
    % plot overlap with p0
    plot(0:size(m_t, 2)-1, m_t(1,:))
    hold on
    % plot overlap with p1
    plot(0:size(m_t, 2)-1, m_t(2,:))
    hold on
    % plot overlap with p2
    plot(0:size(m_t, 2)-1, m_t(3,:))
    xlabel('Time (t)')
    ylabel('Overlap')
    legend({'Overlap with 0', 'Overlap with 1', 'Overlap with 2'}, 'Location', 'southeast')
    saveas(gcf, ['Assignment1/Results/distorted_', input_digit, '_', input_error, '_overlap.png'])
end

function [W, b] = learn(data)
    N = size(data, 2);
    W = 1/N * (data'*data);
    % Set to 0 the diagonal
    for i = 1:size(W,1)
        W(i,i) = 0;
    end
    b = ones(N,1)*0.5;
end

function [x, m_t, E_t] = retrieval(W, b, input, data)
    % Number of neurons N
    N = size(W, 1);
    % Initial state
    x = input;
    prev_x = Inf;
    prev_energy = Inf;
    eps = 1;
    curr_energy = energy(W, b, x);
    m_t = [overlap(data(1, :), x); overlap(data(2, :), x); overlap(data(3, :), x)];
    E_t = [energy(W, b, x)];
    % Until we reach a fixed point
    while abs(curr_energy - prev_energy) >= eps %not(isequal(prev_x, x))
        % Choose a random permutation of the neurons
        prev_x = x;
        prev_energy = curr_energy;
        for j = randperm(N)
            % Update neuron j
            v = W(j, :) * x' + b(j);
            if v >= 0
                x(j) = 1;
            else
                x(j) = -1;
            end
            curr_energy = energy(W, b, x);
            E_t(end + 1) = energy(W, b, x);
            m_t = cat(2, m_t, [overlap(data(1, :), x); overlap(data(2, :), x); overlap(data(3, :), x)]);
        end
    end
end

function m = overlap(memory, state)
    N = size(state, 2);
    m = 1/N * (memory * state');
end

function E = energy(W, b, x)
    E = -1/2 * x * (W * x') - x * b;
end

function show_image(p, name)
    img = reshape(p, 32, 32);
    imagesc(img);
    title(name);
end

function plot_input_and_retrieval(input, retrieval, digit, noise, error, retrieved_memory)
    figure
    subplot(1,2,1)
    show_image(input, ['Digit ', digit, ' with ', noise, ' noise'])
    subplot(1,2,2)
    show_image(retrieval, ['Retrieved a ', num2str(retrieved_memory), '. MSE = ', num2str(error)])
end