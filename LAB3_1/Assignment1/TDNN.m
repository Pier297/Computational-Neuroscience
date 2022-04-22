clear;
rng(1);

% --- Load the data --- %
A = load('NARMA10timeseries.mat');
A = A.NARMA10timeseries;
X = cell2mat(A.input');
Y = cell2mat(A.target');

% --- Split the data into training and test set --- %
training_data_X = X(1:5000, :);
training_data_Y = Y(1:5000, :);
test_X = X(5001:end, :);
test_Y = Y(5001:end, :);

% Split the training set into training and validation set
val_X = training_data_X(4001:end, :);
val_Y = training_data_Y(4001:end, :);
train_X = training_data_X(1:4000, :);
train_Y = training_data_Y(1:4000, :);

% Change data types for training
X = num2cell(X');
Y = num2cell(Y');
train_X = num2cell(train_X');
train_Y = num2cell(train_Y');
val_X = num2cell(val_X');
val_Y = num2cell(val_Y');
training_data_X = num2cell(training_data_X');
training_data_Y = num2cell(training_data_Y');
test_X = num2cell(test_X');
test_Y = num2cell(test_Y');


% --- Model selection --- %
best_conf = {};
best_val_error = inf;
best_train_error = inf;
best_epoch = 0;
best_tr = {};

% First broad grid search, results -> input_delay = 0:12, hidden_sizes = 20
% input_delays = {0:1, 0:2, 0:9, 0:12}; 
% hidden_sizes = {[10], [20], [10, 10], [20,20], [50]};

input_delays = {0:12, 0:14}; 
hidden_sizes = {[20], [35]};

for i = 1:size(input_delays, 2)
    for j = 1:size(hidden_sizes, 2)
        id = cell2mat(input_delays(i));
        h  = cell2mat(hidden_sizes(j));
        
        net = timedelaynet(id, h, "trainlm");
        % Train the model on the current hyperparameter conf.
        tr_indices = 1:4000; %indices used for training
        tv_indices = 4001:5000; %indices used for validation
        ts_indices = []; % indices used for *test*
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = tr_indices;
        % Test: Used for final assessment only
        net.divideParam.testInd = ts_indices;
        % Validation: Used for early stopping
        net.divideParam.valInd = tv_indices;
        
        [Xs,Xi,Ai,Ts] = preparets(net, training_data_X, training_data_Y);
        [net, tr] = train(net,Xs,Ts,Xi,Ai,'UseParallel','yes');

        % simulate it on the training data
        [Xs,Xi,Ai,Ts] = preparets(net, train_X, train_Y);
        [train_pred_Y,Xf,Af] = net(Xs,Xi,Ai);

        train_mse = immse(cell2mat(Ts), cell2mat(train_pred_Y));
        fprintf('All Training data mse = %d\n', train_mse)

        % Simulate on the val set
        [netc,Xic,Aic] = closeloop(net,Xf,Af);
        [val_pred_Y, Xf, Af] = netc(val_X, Xic, Aic);

        val_mse = immse(cell2mat(val_Y), cell2mat(val_pred_Y));
        fprintf('Val mse = %d\n', val_mse)
        
        if val_mse < best_val_error
            best_conf = {id, h};
            best_val_error = val_mse;
            best_train_error = train_mse;
            best_epoch = tr.best_epoch;
            best_tr = tr;
            tr.best_vperf;
            tr.best_perf;
        end
    end 
end

best_id = cell2mat(best_conf(1))
best_h = cell2mat(best_conf(2))
best_epoch

% --- Train the net with the best conf. onto all the training data --- % 
net = timedelaynet(best_id, best_h, "trainlm");

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:5000;    % indices used for training
net.divideParam.testInd = 5001:10000; % Test: Used for final assessment only
net.divideParam.valInd = [];          % Validation: Used for early stopping
net.trainParam.epochs = best_epoch;   % maximum number of epochs

[Xs,Xi,Ai,Ts] = preparets(net, X, Y);
[net, tr] = train(net,Xs,Ts,Xi,Ai);


% simulate it on all the training data
[Xs,Xi,Ai,Ts] = preparets(net, training_data_X, training_data_Y);
[train_pred_Y,Xf,Af] = net(Xs,Xi,Ai);

mse = immse(cell2mat(Ts), cell2mat(train_pred_Y));
fprintf('All Training data mse = %d\n', mse)

% Simulate on the test set
[netc,Xic,Aic] = closeloop(net,Xf,Af);
[test_pred_Y, Xf, Af] = netc(test_X, Xic, Aic);

test_mse = immse(cell2mat(test_Y), cell2mat(test_pred_Y));
fprintf('Test mse = %d\n', test_mse)

% Plot the training signal vs model prediction
figure
plot(cell2mat(Ts))
hold on
plot(cell2mat(train_pred_Y))
xlabel('t')
ylabel('d(t)')
legend({'Real signal', 'Model Prediction'})
title('Training signal')
saveas(gcf, 'Assignment1/Results/TDNN/training_signal.png')

% Plot the test signal vs model prediction
figure
plot(cell2mat(test_Y))
hold on
plot(cell2mat(test_pred_Y))
xlabel('t')
ylabel('d(t)')
legend({'Real signal', 'Model Prediction'})
title('Test signal')
saveas(gcf, 'Assignment1/Results/TDNN/test_signal.png')

% Plot Model selection loss function
figure
plotperform(best_tr)
saveas(gcf, 'Assignment1/Results/TDNN/grid_search_loss_function.png')

% Plot loss function
figure
plotperform(tr)
saveas(gcf, 'Assignment1/Results/TDNN/loss_function.png')

% Save the net structure 'net' and training record 'tr'
save('Assignment1/Results/TDNN/tr.mat', 'tr')
save('Assignment1/Results/TDNN/net.mat', 'net')

fileID = fopen('Assignment1/Results/TDNN/mse.txt','w');
fprintf(fileID,'Best epoch = %d\n',best_epoch);
fprintf(fileID,'Train MSE | Val MSE | TEST MSE\n');
fprintf(fileID,'%d | %d | %d\n',best_train_error,best_val_error, test_mse);
fclose(fileID);
