

%  ...\brainblocks-master\brainblocks-master\examples\python\sequence_learner
% bami:
% columns have cells
% segment refers to pre-synaptic input, either proximal (input presyn) or
% dital segment (ex presyn)

clear all
addpath(genpath('D:\20220624_Lethbridge\functions\npy-matlab-master'))
cd D:\20220624_Lethbridge\20230212_nn_HTM\save_dt
X  = readNPY('anomaly_X.npy');
X_bits  = readNPY('anomaly_X_bits.npy');
% 
%
% figure
% for ii = 1:4
%     subplot(5,1,ii)
%     plot(X(:,ii),'k')
% end
% subplot(5,1,5); plot(mean(X,2),'r')
% 
% 
% figure
% for ii = 1:200
%     subplot(10,20,ii)
%     imagesc(squeeze(X_bits(ii,:,:))); colormap(flip(gray))
%     axis off
% end

% input neurons
param.N_in     = size(X_bits,2)*size(X_bits,3);
param.N_in_dim = [size(X_bits,2) size(X_bits,3)]; % must be square

% HTM neurons
param.N_col          = 512; %1024; %2048 ;                 % number of columns
param.N_col_cell     = 4;                            % number of cells per columns
param.N_total_cell   = param.N_col*param.N_col_cell; % number of cells per columns
param.density_col    = 125 ;                         % neurons/mm^2

param.act_thr   =15 ;     % activation threshold to pre-synap, potentiates cell
param.learn_thr =10 ;     % matching of potentially connected, incr/decr synapse

param.perm_thr=0.2 ;    % receptor permanence threshold
param.perm_inc=0.03  ;  % receptor permanence increment
param.perm_dec=0.015  ; % receptor permanence decrement
param.perm_ini=0.15 ;   % receptor permanence after growth

param.synapse_sz =32 ;  % number of synapse which segment grows

%%% rows index: from, col index: to

% HTM column place
sqrt_len = sqrt(param.N_col/param.density_col);
param.N_coord = [rand([1 param.N_col])*sqrt_len; rand([1 param.N_col])*sqrt_len];
clear sqrt_len
% 25/(param.density_col*(pi*0.5*0.5)) % distal ex prob
% 75/(param.density_col*(pi*1.5*1.5)) % loca ex prob

% connect input layers and HTM
param.connect_in  = zeros(param.N_in,param.N_col);
for i_in = 1:(param.N_in)
    i_col    = randi(param.N_col);
    n_within = (param.N_coord(1,:) - param.N_coord(1,i_col)).^2 + (param.N_coord(2,:) - param.N_coord(2,i_col)).^2 < (0.5)^2 ;
    n_within_conn                            = randsample(find(n_within),round(sum(n_within)*(0.25)));
    param.connect_in(i_in,n_within_conn) = 1;

    %     figure; scatter(param.N_coord(1,:), param.N_coord(2,:),'.k'); hold on; scatter(param.N_coord(1,n_within_conn), param.N_coord(2,n_within_conn),100, '.r'); hold off
    %
    %     figure; scatter(param.N_coord(1,:), param.N_coord(2,:),200, sum(param.connect_in,1),'.'); colormap(jet); colorbar
    %     figure; imagesc(param.connect_in)
end
clear n_within n_within_conn i_in i_col

% connect within HTM exc
param.connect_col_ex_mask  = zeros(param.N_col,param.N_col);
param.connect_flat_ex      = zeros(param.N_total_cell,param.N_total_cell);
param.ExRadius = 1.5;  % mm
for i_col = 1:(param.N_col)
    % i_col = 1000
    n_within_ex = (param.N_coord(1,:) - param.N_coord(1,i_col)).^2 + (param.N_coord(2,:) - param.N_coord(2,i_col)).^2 < (param.ExRadius)^2 ;
    param.connect_col_ex_mask(i_col,:)  = n_within_ex;


    %    figure; scatter(param.N_coord(1,:), param.N_coord(2,:),'.k'); hold on; scatter(param.N_coord(1,n_within_ex), param.N_coord(2,n_within_ex),100, '.r'); hold off

    % figure; imagesc(param.connect_col_ex_mask)
    % figure; imagesc(param.connect_flat_ex_mask)
    % figure; imagesc(param.connect_flat_ex)
end
param.connect_flat_ex_mask = repelem(param.connect_col_ex_mask,param.N_col_cell,param.N_col_cell);
n_within_ex                = randsample(find(param.connect_flat_ex_mask),round(sum(param.connect_flat_ex_mask(:))*(0.09)));
param.connect_flat_ex(n_within_ex) = 1;
% sum(param.connect_flat_ex(:))/sum(param.connect_flat_ex_mask(:))
% any(param.connect_flat_ex_mask(find(param.connect_flat_ex)) ~= 1)
clear n_within_ex i_col

% permanence
param.connect_perm = param.connect_flat_ex;
param.connect_perm(find(param.connect_perm)) = rand(sum(param.connect_perm(:)),1);
param.connect_syn  = param.connect_perm > param.perm_thr;
% figure; subplot(1,3,1); imagesc(param.connect_flat_ex); subplot(1,3,2); imagesc(param.connect_perm); subplot(1,3,3); imagesc(param.connect_syn)
win_cell_all      = zeros(param.N_total_cell,size(X_bits,1)); % active
segment_act_all   = zeros(param.N_total_cell,size(X_bits,1)); % potentiate (pre-active learn)
segment_match_all = zeros(param.N_total_cell,size(X_bits,1)); % potentiate (learn for burst/punish)
segment_NpotSyn_all = zeros(param.N_total_cell,size(X_bits,1)); % N of potentiated synapse (match)
learn = 1;
connect_perm_ini = param.connect_perm; %%%%%%%%%% plot
for i_itr = 1:size(X_bits,1)
    disp(['iteration ' num2str(i_itr)])
    % i_itr = 1
    % i_itr = 2

    %%% input
    X_bits_i = squeeze(X_bits(i_itr,:,:));
    X_bits_i = reshape(X_bits_i, [param.N_in 1]);
    active_in_col = (param.connect_in')*X_bits_i > 0;  %%% edit later: the input to HTM should be SDR from SP
    % sum(active_in_col >= 1)/length(active_in_col)
    %     active_in_flat = repelem(active_in_col,param.N_col_cell);

    %%% accept input against prediction
    win_cell        = zeros(1,param.N_total_cell);
    punish_cell     = zeros(1,param.N_total_cell);
    conn_pre        = sum(param.connect_flat_ex,2);
    for i_a = 1:param.N_col %find(active_in_col)'
        i_a_ind = ((i_a-1)*param.N_col_cell + 1):((i_a)*param.N_col_cell);

        if i_itr == 1
            segment_act_i   = zeros(1,param.N_col_cell);
            segment_match_i = zeros(1,param.N_col_cell);
        else
            segment_act_i     = segment_act_all(i_a_ind,i_itr-1);
            segment_match_i   = segment_match_all(i_a_ind,i_itr-1);
            segment_NpotSyn_i = segment_NpotSyn_all(i_a_ind,i_itr-1);
        end

        if active_in_col(i_a) == 1
            if any(segment_act_i) % activate predicted synapse
                win_cell(i_a_ind) = segment_act_i;
            else                  % burst column, activate cell
                if any(segment_match_i)
                    [~,m_ind] = max(segment_NpotSyn_i);
                else
                    conn_pre_i  = conn_pre(i_a_ind);
                    [~,m_ind] = min(conn_pre_i);
                end

                win_cell(i_a_ind(m_ind(randi(length(m_ind))))) = 1;
            end
        else                      % punish synapse (match)
            punish_cell(i_a_ind) = segment_match_i;
        end
    end

    %%% update and grow synapse
    if i_itr > 1
        %%% update synapse
        win_cell_pre = win_cell_all(:,i_itr-1);
        for i_w = find(win_cell) % permanence update for winner cell
            % i_w = 10
            pre_syn_ind = find(param.connect_syn(:,i_w)) ;
            for i_p_sy = pre_syn_ind
                % i_p_sy = pre_syn_ind(1)
                if win_cell_pre(i_p_sy) == 1
                    param.connect_perm(i_p_sy,i_w) = param.connect_perm(i_p_sy,i_w) + param.perm_inc;
                else
                    param.connect_perm(i_p_sy,i_w) = param.connect_perm(i_p_sy,i_w) - param.perm_dec;
                end
            end
        end
        for i_pu = find(punish_cell) % permanence update for punish cell
            % i_pu = 10
            pre_syn_ind = find(param.connect_syn(:,i_pu)) ;
            for i_p_sy = pre_syn_ind
                % i_p_sy = pre_syn_ind(1)
                if win_cell_pre(i_p_sy) == 1
                    param.connect_perm(i_p_sy,i_pu) = param.connect_perm(i_p_sy,i_pu) - param.perm_dec;
                end
            end
        end
        param.connect_perm(param.connect_perm < 0) = 0;
        param.connect_perm(param.connect_perm > 1) = 1;
        param.connect_syn  = param.connect_perm > param.perm_thr;

        %%% grow synapse
        segment_NpotSyn_pre = segment_NpotSyn_all(:,i_itr-1);
        for i_w = find(win_cell)
            % i_w = 1
            i_n_grow = param.synapse_sz - segment_NpotSyn_pre(i_w);
            if i_n_grow > 0
                conn_ex_grow = param.connect_flat_ex_mask - param.connect_flat_ex ;
                conn_ex_grow = conn_ex_grow(:,i_w).*win_cell_pre;
                conn_ex_grow_ind = find(conn_ex_grow);
                if i_n_grow > length(conn_ex_grow_ind) & length(conn_ex_grow_ind) ~= 0
                    ind_grow     = conn_ex_grow_ind(randi(length(conn_ex_grow_ind),[1 length(conn_ex_grow_ind)]));
                elseif length(conn_ex_grow_ind) == 0
                    ind_grow = [];
                else
                    ind_grow     = conn_ex_grow_ind(randi(length(conn_ex_grow_ind),[1 i_n_grow]));
                end

                param.connect_flat_ex(ind_grow,i_w) = 1;
                param.connect_perm(ind_grow,i_w)    = param.perm_ini;
                param.connect_syn                   = param.connect_perm > param.perm_thr;
            end
        end
    end

    %%% potentiate cell of next time iteration
    % (param.connect_syn(:,1:100)')*win_cell'
    act_conn  = (param.connect_syn')*win_cell';
    act_poten = (param.connect_flat_ex')*win_cell';
    % figure; histogram(act_conn)
    % figure; histogram(act_poten)
    segment_act   = act_conn >= param.act_thr;
    segment_match = act_poten >= param.learn_thr;
    % sum(segment_match)
    % sum(segment_act)

    %%% store active(win), potentiation(pre-active/match) of this iteration
    win_cell_all(:,i_itr)       = win_cell;
    segment_act_all(:,i_itr)    = segment_act;
    segment_match_all(:,i_itr)  = segment_match;
    segment_NpotSyn_all(:,i_itr)= act_poten;

end

figure;
blur_sm = 3;
c_scale = 1;
tmp_img = imgaussfilt(win_cell_all,blur_sm );
c_mxx = max(tmp_img(:))*c_scale;
subplot(2,2,1); imagesc(tmp_img ); clim([0 c_mxx])% active cell
title('active cells')
tmp_img = imgaussfilt(segment_act_all,blur_sm );
c_mxx = max(tmp_img(:))*c_scale;
subplot(2,2,2); imagesc(tmp_img ); clim([0 c_mxx])% % potentiation (for predicted cell)
title('potentiated cells (predicted cell)')
tmp_img = imgaussfilt(segment_match_all,blur_sm );
c_mxx = max(tmp_img(:))*c_scale;
subplot(2,2,3); imagesc(tmp_img ); clim([0 c_mxx])% % potentiation (for punish/burst)
title('potentiated cells (punish/burst cell)')
xlabel('time step'); ylabel('cell')
tmp_img = imgaussfilt(segment_NpotSyn_all,blur_sm );
c_mxx = max(tmp_img(:))*c_scale;
subplot(2,2,4); imagesc(tmp_img ); clim([0 c_mxx])% % N of pre-synaptic potential connection
title('pre-synaptic potential connection')
colormap(flip(gray))
set(gcf,'color','w');

figure;
plot_cell = [1 100]; [1 param.N_total_cell] ;
subplot(1,3,1); imagesc(connect_perm_ini); clim([0 1]); xlim(plot_cell); ylim(plot_cell)
subplot(1,3,2); imagesc(param.connect_perm); clim([0 1]); xlim(plot_cell); ylim(plot_cell)
subplot(1,3,3); imagesc(param.connect_perm - connect_perm_ini); clim([-1 1]); xlim(plot_cell); ylim(plot_cell)
colormap(jet)


