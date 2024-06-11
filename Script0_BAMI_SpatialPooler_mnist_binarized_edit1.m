
% ...brainblocks-master\brainblocks-master\examples\python\experiments\image_classification

clear all

addpath(genpath('D:\20220624_Lethbridge\functions\npy-matlab-master'))
cd D:\20220624_Lethbridge\20230212_nn_HTM\save_dt
x_train  = readNPY('mnist_x_train.npy');
y_train  = readNPY('mnist_y_train.npy');
x_test   = readNPY('mnist_x_test.npy');
y_test   = readNPY('mnist_y_test.npy');
param.pixel_thresh =128;

% input neurons
param.N_in = size(x_train,2)*size(x_train,3);
param.N_in_dim = [size(x_train,2) size(x_train,3)];

% SP neurons
param.N_col = 2048 ;   % number of neurons (columns)
param.density_col = 125 ; % neurons/mm^2

param.perm_thr=0.2 ;  % receptor permanence threshold
param.perm_inc=0.03  ;  % receptor permanence increment
param.perm_dec=0.015  ;  % receptor permanence decrement

param.pct_pool=0.8;  % pooling percentage
param.pct_conn=1.0 ; % initial connected percentage
param.pct_learn=0.3; % learn percentage

%%%

% SP neurons place
sqrt_len = sqrt(param.N_col/param.density_col);
param.N_coord = [rand([1 param.N_col])*sqrt_len; rand([1 param.N_col])*sqrt_len];
clear sqrt_len

% connect input layers and SP
param.connect = zeros(param.N_in,param.N_col);
for i_in = 1:(param.N_in)
    i_col = randi(param.N_col);
    n_within = (param.N_coord(1,:) - param.N_coord(1,i_col)).^2 + (param.N_coord(2,:) - param.N_coord(2,i_col)).^2 < (0.5)^2 ;
    n_within_conn = randsample(find(n_within),round(sum(n_within)*(0.25)));
    param.connect(i_in,n_within_conn) = 1;
    %     figure
    %     scatter(param.N_coord(1,:), param.N_coord(2,:),'.k'); hold on
    %     scatter(param.N_coord(1,n_within_conn), param.N_coord(2,n_within_conn),100, '.r'); hold off

    %figure; scatter(param.N_coord(1,:), param.N_coord(2,:),200, sum(param.connect,1),'.'); colormap(jet); colorbar
    %figure; imagesc(param.connect)
end
clear n_within n_within_conn i_in i_col

% connect within SP inhibition
param.connect_inh = zeros(param.N_col,param.N_col);
param.InhRadius = 0.5;  % mm
for i_col = 1:(param.N_col)
    % i_col = 1000
    n_within_inh = (param.N_coord(1,:) - param.N_coord(1,i_col)).^2 + (param.N_coord(2,:) - param.N_coord(2,i_col)).^2 < (param.InhRadius)^2 ;
    param.connect_inh(i_col,:) = n_within_inh;

    %         figure
    %         scatter(param.N_coord(1,:), param.N_coord(2,:),'.k'); hold on
    %         scatter(param.N_coord(1,n_within_inh), param.N_coord(2,n_within_inh),100, '.r'); hold off

    % figure; imagesc(param.connect_inh)
end
clear n_within_inh i_col

% permanence
param.connect_perm = param.connect;
param.connect_perm(find(param.connect_perm)) = rand(sum(param.connect_perm(:)),1);
param.connect_syn  = param.connect_perm > param.perm_thr;
% figure; imagesc(param.connect_syn)
learn = 1;
dutyCycle_len = 1000;
dutyCycle = zeros(dutyCycle_len,param.N_col);
SP_col    = zeros(size(x_train,1),param.N_col);
for i_itr = 1:size(x_train,1)
    disp(['iteration ' num2str(i_itr)])
    % i_itr = 2022
    x_train_i = squeeze(x_train(i_itr,:,:));
    x_train_i = x_train_i > param.pixel_thresh;
    x_train_i = reshape(x_train_i, [param.N_in 1]);
    if learn == 1
        if i_itr == 1
            boost = ones(param.N_col,1);
        end
    end
    overlap = (param.connect_syn')*x_train_i;
    if learn == 1
        overlap = overlap.*boost;
    end
    active_col = zeros(param.N_col,1);
    for i_col = 1:(param.N_col)
        tmp_minOverlap = sort(overlap(find(param.connect_inh(i_col, :))));
        tmp_minOverlap = tmp_minOverlap(end-10);
        active_col(i_col) = (overlap(i_col) >= tmp_minOverlap);

        %     figure
        %     subplot(1,2,1)
        %     scatter(param.N_coord(1,:), param.N_coord(2,:),100,overlap,'.'); colormap(jet);
        %     subplot(1,2,2)
        %     scatter(param.N_coord(1,:), param.N_coord(2,:),100,overlap,'.'); colormap(jet); hold on
        %     scatter(param.N_coord(1,find(active_col)), param.N_coord(2,find(active_col)),100, '.r'); hold off
    end
    if learn == 1
        param.connect_perm(:,find(active_col)) = param.connect_perm(:,find(active_col)) + param.perm_inc;
        param.connect_perm(:,find(~active_col)) = param.connect_perm(:,find(~active_col)) - param.perm_dec;
        param.connect_perm(param.connect_perm < 0) = 0;
        param.connect_perm(param.connect_perm > 1) = 1;
        param.connect_perm(~param.connect)         = 0;
    end
    % cp_1 = param.connect_perm;
    % cp_2 = param.connect_perm;
    % figure
    % subplot(1,2,1)
    % imagesc(cp_1(1:50,1:50)); clim([0 1]); colormap(jet)
    % subplot(1,2,2)
    % imagesc(cp_2(1:50,1:50)); clim([0 1]); colormap(jet)
    if learn == 1
        dutyCycle(rem(i_itr,dutyCycle_len)+1 , :) = active_col;
        % figure; imagesc(dutyCycle); colormap(flip(gray))

        dutyCycle_mn = mean(dutyCycle);
        %     sigmoid_steep = 4;
        for i_col = 1:(param.N_col)
            tmp_diff = mean(dutyCycle_mn(find(param.connect_inh(i_col, :)))) - dutyCycle_mn(i_col);
            %         boost(i_col) = 1/(1+exp(-tmp_diff*sigmoid_steep))+0.5;
            boost(i_col) = exp(tmp_diff);
            % figure; subplot(1,2,1); histogram(dutyCycle_mn); subplot(1,2,2); histogram(boost)

        end
    end

    SP_col(i_itr,:) = active_col;
    % figure; imagesc(SP_col(1:i_itr,:)); colormap(flip(gray))
end

% SVM on both training and testing data.


