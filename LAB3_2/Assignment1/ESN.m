clear;
rng(1);

% --- Load the data --- %
A = load('NARMA10timeseries.mat');
A = A.NARMA10timeseries;
data_X = cell2mat(A.input');
data_Y = cell2mat(A.target');

% --- Split the data into training and test set --- %
training_data_X = data_X(1:5000, :);
training_data_Y = data_Y(1:5000, :);
test_X = data_X(5001:end, :);
test_Y = data_Y(5001:end, :);

% Split the training set into training and validation set
val_X = training_data_X(4001:end, :);
val_Y = training_data_Y(4001:end, :);
train_X = training_data_X(1:4000, :);
train_Y = training_data_Y(1:4000, :);

%   Notation:
% Nu is the input size = 1
% Nr is the reservoir state size, here also called n_reservoir_units
% Ny is the the output size = 1
% 
% Win for the input-to-reservoir weight matrix
% Wr for the recurrent reservor weight matrix
% Wout for the reservoir-toreadout weight matrix
Nu = 1; Ny = 1;

% % Hyperparameters for the grid search and results
% input_scaling_values = {0.1, 0.5, 1}; % -> 0.5
% n_reservoir_units_values = {10, 100, 1000}; % -> 1000
% rho_desired_values = {0.7, 0.8, 0.9}; % -> 0.9
% regularization_values = {1e-5, 1e-4, 1e-3}; % -> 1e-5
% reservoir_connectivity_values = {0.3, 0.5, 0.7, 0.9, 1}; % -> 0.3

% Second grid search (more refined around the best hyperparameters of the
% first one.
input_scaling_values = {0.5, 0.3, 0.7};
n_reservoir_units_values = {300, 500, 750, 1000};
rho_desired_values = {0.85, 0.9};
regularization_values = {1e-8, 1e-7, 1e-6};
reservoir_connectivity_values = {0.1, 0.2, 0.3, 0.4, 0.7};

n_transient = 20;

best_val_error = inf;
best_conf = {};

% Number of times to repeat the training with the same hyperparameters
n = 5;

% Grid search
for a = 1:size(input_scaling_values, 2)
    for b = 1:size(n_reservoir_units_values, 2)
        for c = 1:size(rho_desired_values, 2)
            for d = 1:size(regularization_values, 2)
                for e = 1:size(reservoir_connectivity_values, 2)
                    % Get the current conf.
                    input_scaling = cell2mat(input_scaling_values(a));
                    n_reservoir_units = cell2mat(n_reservoir_units_values(b));
                    rho_desired = cell2mat(rho_desired_values(c));
                    reg = cell2mat(regularization_values(d));
                    reservoir_connectivity = cell2mat(reservoir_connectivity_values(e));
                    
                    % Repeat the training n times and avg. the val error
                    val_error = 0;
                    for t = 1:n
                        [Win, Wr] = create_esn(Nu, n_reservoir_units, input_scaling, rho_desired, reservoir_connectivity);
                        [Wout, X] = train_esn(train_X, train_Y, n_reservoir_units, Win, Wr, n_transient, reg);
                        % Training MSE
                        train_pred_Y = Wout * X;
                        train_mse = immse(train_pred_Y, train_Y(n_transient:end, :)');
                        fprintf('Train mse = %d\n', train_mse)
                        % Val MSE
                        X = compute_esn_states(val_X, n_reservoir_units, Win, Wr, n_transient);
                        val_pred_Y = Wout * X;
                        val_mse = immse(val_pred_Y, val_Y(n_transient:end, :)');
                        fprintf('Val mse = %d\n', val_mse)
                        val_error = val_error + val_mse;
                    end
                    
                    val_error = val_error / n;
                    if val_error < best_val_error
                        best_val_error = val_error;
                        best_conf = {input_scaling, n_reservoir_units, rho_desired, reg, reservoir_connectivity};
                    end
                end
            end
        end
    end
end

% Get the best hyperparameters
input_scaling = cell2mat(best_conf(1))
n_reservoir_units = cell2mat(best_conf(2))
rho_desired = cell2mat(best_conf(3))
reg = cell2mat(best_conf(4))
reservoir_connectivity = cell2mat(best_conf(5))

fprintf('Final Retraining\n')

% Repeat the training n times and avg. the train and test error
train_error = 0;
test_error = 0;
% Note that the final ESN model is simply the last one trained (might not be
% the one with the lowest test error), indeed we don't make any choice based on the test error.
for t = 1:n
    [Win, Wr] = create_esn(Nu, n_reservoir_units, input_scaling, rho_desired, reservoir_connectivity);
    [Wout, X] = train_esn(training_data_X, training_data_Y, n_reservoir_units, Win, Wr, n_transient, reg);
    % Training MSE
    train_pred_Y = Wout * X;
    train_mse = immse(train_pred_Y, training_data_Y(n_transient:end, :)');
    fprintf('Train mse = %d\n', train_mse)
    train_error = train_error + train_mse;
    % Test MSE
    X = compute_esn_states(test_X, n_reservoir_units, Win, Wr, n_transient);
    test_pred_Y = Wout * X;
    test_mse = immse(test_pred_Y, test_Y(n_transient:end, :)');
    fprintf('Test mse = %d\n', test_mse)
    val_error = val_error + test_mse;
end

fprintf('\n(Avg) Train mse = %d\n', train_mse)
fprintf('(Avg) Test mse = %d\n', test_mse)

% --- Plot and save the results ---

% Save the final ESN: Win, Wr, Wout
save('Assignment1/Results/Win.mat', 'Win')
save('Assignment1/Results/Wr.mat', 'Wr')
save('Assignment1/Results/Wout.mat', 'Wout')

% Save hyperparameters used
fileID = fopen('Assignment1/Results/model_hyperparameters.txt','w');
fprintf(fileID,'input scaling = %f\n',input_scaling);
fprintf(fileID,'n reservoid units = %d\n',n_reservoir_units);
fprintf(fileID,'rho desired = %f\n',rho_desired);
fprintf(fileID,'reg. coefficient = %f\n',reg);
fprintf(fileID,'reservoir connectivity = %f\n',reservoir_connectivity);
fprintf(fileID,'n_transient = %d (Number of reservoid states discarted)\n',n_transient);

% Save train, val and test error
fileID = fopen('Assignment1/Results/model_errors.txt','w');
fprintf(fileID,'Train and Test error avg over %d trials (val_mse is the avg. validation error of the selected hyperparameter during the grid search)\n', n);
fprintf(fileID,'train_mse val_mse test_mse\n');
fprintf(fileID,'%d %d %d\n',train_mse, best_val_error, test_mse);
fclose(fileID);

% Plot the training signal vs model prediction
figure
plot(training_data_Y(n_transient:end, :))
hold on
plot(train_pred_Y')
xlabel('t')
ylabel('d(t)')
legend({'Real signal', 'Model Prediction'})
title('Training signal')
saveas(gcf, 'Assignment1/Results/training_signal.png')

% Plot the test signal vs model prediction
figure
plot(test_Y(n_transient:end, :))
hold on
plot(test_pred_Y')
xlabel('t')
ylabel('d(t)')
legend({'Real signal', 'Model Prediction'})
title('Test signal')
saveas(gcf, 'Assignment1/Results/test_signal.png')


function [Win, Wr] = create_esn(Nu, Nr, input_scaling, rho_desired, reservoir_connectivity)
    % Init the input-to-reservoir Win
    Win = input_scaling*(2*rand(Nr,Nu+1)-1);

    % Init the recurrent reservoir weights Wr
    Wrandom = 2*rand(Nr,Nr)-1;
    
    % Disable recurrent units with P 'reservoir_connectivity'
    Wrandom(rand(Nr, Nr) > reservoir_connectivity) = 0;
    
    % scale the random matrix to the desired spectral radius rho
    Wr = Wrandom * (rho_desired/max(abs(eig(Wrandom))));
end

function [Wout, X] = train_esn(train_X, train_Y, Nr, Win, Wr, n_transient, lambda_r)
    % Compute training states
    X = zeros(Nr, size(train_X, 1) + 1);
    for t = 1:size(train_X, 1)
       u_t = train_X(t);
       X(:, t + 1) = tanh(Win * [u_t ; 1] + Wr * X(:, t));
    end
    
    X = [X ; ones(1, size(X,2))];
    % discard n_transient to let the network stabilize, also skip the first
    % one since it's 0 by definition (prev for loop, X size 4001 instead of 4000)
    X = X(:, n_transient+1:end);

    Ytarget = train_Y(n_transient:end, :)';
    Wout = Ytarget * X' * inv(X * X' + lambda_r*eye(Nr+1));
end

function X = compute_esn_states(inputs, Nr, Win, Wr, n_transient)
    X = zeros(Nr, 1001);
    for t = 1:size(inputs, 1)
       u_t = inputs(t);
       X(:, t + 1) = tanh(Win * [u_t ; 1] + Wr * X(:, t));
    end
    X = [X ; ones(1, size(X,2))];
    X = X(:, n_transient+1:end);
end
