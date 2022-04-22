rng(1) % fix seed for reproducible random numbers

data = load('lab2_1_data.csv'); % data \in R^(2x100)

% Our dataset has 100 samples in R^2, our objective is to train a linear
% firing rate neuron; so we have 2 inputs which are connected to the ouput
% neuron via 2 weights w.
% Formally:
% input   u \in R^2x1
% weights w \in R^2x1
% output  v \in R^1x1

% Define random initial weights with values from a uniform distribution
% over [-1; 1]
w = 2*rand(size(data, 1), 1) - 1;

MAX_EPOCHS = 20;
eta = 1e-3;

% history of the evolution of the weights
W = [w];

% TODO: Stop learning if the weight vector stabilizes?
for epoch = 1:MAX_EPOCHS
    % Shuffle the data
    data = data(:, randperm(size(data, 2)));
    % For each pattern
    for p = 1:size(data, 2)
        % Get the current pattern u
        u = data(:, p);
        % Compute the output v
        v = dot(w, u);
        % Apply online Hebb Learning rule
        delta_w = eta * (v * u);
        w = w + delta_w;
        W = cat(2, W, w);
    end
end

% ---------- (P1) ----------
figure
% Plot the training data points
scatter(data(1, :), data(2, :))
hold on
% Plot the final vector w
plotv(w, '-')
hold on
% Plot the principal eigenvector of input correlation matrix Q
% First compute Q = <u, u>
Q = data * data';
% Compute the eigenvector of Q
[V, D] = eig(Q);
% Find the max eigenvalue and its associated eigenvector
[max_columns] = max(D, [], 1);
[~, index] = max(max_columns);
max_eigenvector = V(:, index);
plotv(max_eigenvector, '--')

legend({'Training data', 'Final weight vector', 'Principal eigenvector'})
saveas(gcf, 'Assignment1/Results/Plot_P1_weight_eigenvector.png')

% ---------- (P2) ----------
% Plot the evolution of the first component of the weight vector w
% w.r.t. the time (epochs)
figure
plot(0:size(W,2)-1, W(1, :))
xlabel('Time (t)')
ylabel('w(1)')
title('Evolution of the first component of w')
saveas(gcf, 'Assignment1/Results/Plot_P2_w(1).png')

% Plot w(2) vs time
figure
plot(0:size(W,2)-1, W(2, :))
xlabel('Time (t)')
ylabel('w(2)')
title('Evolution of the second component of w')
saveas(gcf, 'Assignment1/Results/Plot_P2_w(2).png')

% Plot norm of w vs time
figure

norms = [];
for i = 1:size(W,2)
   norms(end + 1) = norm(W(:, i)); 
end

plot(0:size(W,2)-1, norms)
xlabel('Time (t)')
ylabel('|| w ||')
title('Evolution of || w ||')
saveas(gcf, 'Assignment1/Results/Plot_P2_norm_w.png')

% Save evolution of w (matrix W) as .mat
save('Assignment1/Results/W.mat', 'W');