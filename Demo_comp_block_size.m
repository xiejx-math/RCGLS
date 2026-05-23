% this file can produce Figures 1 and 2 in the manuscript

close all;
clear;
clc;

num_runs = 10; % average times

%% experimental setup 
n = 128;
d = 1024;

% n = 128;
% d = 128;

% n = 1024;
% d = 128;

lambda = 0.05;
%lambda = 0.005;

if d <= n
    denom = d; 
else
    denom = n; 
end

log2_q_vec = 0:7;
q_vec = 2.^log2_q_vec; 
num_q = length(q_vec);

%%%
epochs_rcgls = zeros(num_q, num_runs); 
time_rcgls = zeros(num_q, num_runs);
epochs_rcd   = zeros(num_q, num_runs); 
time_rcd   = zeros(num_q, num_runs);

%% executing "run_time" times of the algorithms
for run = 1:num_runs
    fprintf('Processing experimental data group %d/%d...\n', run, num_runs);
    
    %% generated data matrix A and observation b
    [A, b, x_true, x1, dev, U, sigA, V] = generate_data(n, d, ...
        'correlated',               true,...
        'matrix correlation',       0.7,...
        'matrix deviation',         5,...
        'hansen singular values',   5,...      
        'hansen max',               1,...      
        'kappa',                    4,...      
        'noise level',              0.1,...   
        'hansen input',             true,...
        'input dev',                1,...
        'prior dev',                0,...
        'prior mean',               0,...
        'structure',                false,...
        'plot',                     false); 

    if d <= n
        xLS = (A' * A + lambda * eye(d)) \ (A' * b);
    else
        xLS = A' * ((A * A' + lambda * eye(n)) \ b); 
    end
    
    %% parameter setup
    opts.x_star = xLS; 
    opts.x0 = zeros(d, 1); 
    opts.y0 = zeros(n, 1);
    opts.tol = 1e-10;
    opts.max_iter = 20000;
    
    %%%
    for i = 1:num_q
        q = q_vec(i); % size of the block
        
         %% run RidgeRCGLSU
        [~, ~, iter_rcgls, t_rcgls] = My_RidgeRCGLSU(A, b, lambda, q, opts);

        %% run RidgeRCDU
        [~, ~, iter_rcd,   t_rcd]   = My_RidgeRCDU(A, b, lambda, q, opts); 
        
        %% store the compute results
        final_iter_rcgls = iter_rcgls(end); 
        final_time_rcgls = t_rcgls(end);
        final_iter_rcd   = iter_rcd(end);   
        final_time_rcd   = t_rcd(end);
        
        epochs_rcgls(i, run) = (final_iter_rcgls * q) / denom;
        time_rcgls(i, run)   = final_time_rcgls;
        epochs_rcd(i, run)   = (final_iter_rcd * q) / denom;
        time_rcd(i, run)     = final_time_rcd;
    end
end

%%%
calc_stats = @(mat) deal(median(mat,2), min(mat,[],2), max(mat,[],2), prctile(mat,25,2), prctile(mat,75,2));
[med_ep_rcgls, min_ep_rcgls, max_ep_rcgls, q25_ep_rcgls, q75_ep_rcgls] = calc_stats(epochs_rcgls);
[med_ep_rcd,   min_ep_rcd,   max_ep_rcd,   q25_ep_rcd,   q75_ep_rcd]   = calc_stats(epochs_rcd);
[med_t_rcgls, min_t_rcgls, max_t_rcgls, q25_t_rcgls, q75_t_rcgls] = calc_stats(time_rcgls);
[med_t_rcd,   min_t_rcd,   max_t_rcd,   q25_t_rcd,   q75_t_rcd]   = calc_stats(time_rcd);


%% visualization
title_str = sprintf('n=%d, d=%d', n, d);
c_rcgls = [1 0 0];      m_rcgls = 'o';  ls_rcgls = '-';
c_rcd   = [0.4 0 0.6];  m_rcd   = '^';  ls_rcd   = '-';
alpha_minmax = 0.05; 
alpha_iqr    = 0.15; 
lw = 1.5;            

plot_shaded = @(ax, x, y_min, y_max, y_25, y_75, y_med, col, marker, ls) ...
    [fill(ax, [x, fliplr(x)], [y_min', fliplr(y_max')], col, 'FaceAlpha', alpha_minmax, 'EdgeColor', 'none'); ...
     fill(ax, [x, fliplr(x)], [y_25', fliplr(y_75')], col, 'FaceAlpha', alpha_iqr, 'EdgeColor', 'none'); ...
     plot(ax, x, y_med, 'Color', col, 'LineStyle', ls, 'LineWidth', lw, 'Marker', marker, 'MarkerSize', 7, 'MarkerFaceColor', 'none')];

x_ax = log2_q_vec;

%% RSE vs.Epochs
fig1 = figure('Position', [100, 100, 550, 450], 'Color', 'w');
ax1 = axes('Parent', fig1); hold(ax1, 'on'); 
h_rcd   = plot_shaded(ax1, x_ax, min_ep_rcd, max_ep_rcd, q25_ep_rcd, q75_ep_rcd, med_ep_rcd, c_rcd, m_rcd, ls_rcd);
h_rcgls = plot_shaded(ax1, x_ax, min_ep_rcgls, max_ep_rcgls, q25_ep_rcgls, q75_ep_rcgls, med_ep_rcgls, c_rcgls, m_rcgls, ls_rcgls);
xlabel(ax1, '$\log_2(q)$', 'Interpreter', 'latex', 'FontSize', 15); 
ylabel(ax1, 'Epochs', 'Interpreter', 'latex', 'FontSize', 15);
title(ax1, ['$', title_str, '$'], 'Interpreter', 'latex', 'FontSize', 17);
legend(ax1, [h_rcd(3), h_rcgls(3)], {'RidgeGRCDU', 'RidgeRCGLSU'}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best', 'Box', 'on');
grid(ax1, 'off'); box(ax1, 'on');
set(ax1, 'Color', 'w', 'FontSize', 12, 'LineWidth', 1.2, 'XTick', log2_q_vec);
xlim(ax1, [0, max(log2_q_vec)]);

%% RSE vs.CPU
fig2 = figure('Position', [700, 100, 550, 450], 'Color', 'w');
ax2 = axes('Parent', fig2); hold(ax2, 'on'); 
h_rcd_t   = plot_shaded(ax2, x_ax, min_t_rcd, max_t_rcd, q25_t_rcd, q75_t_rcd, med_t_rcd, c_rcd, m_rcd, ls_rcd);
h_rcgls_t = plot_shaded(ax2, x_ax, min_t_rcgls, max_t_rcgls, q25_t_rcgls, q75_t_rcgls, med_t_rcgls, c_rcgls, m_rcgls, ls_rcgls);
xlabel(ax2, '$\log_2(q)$', 'Interpreter', 'latex', 'FontSize', 15); 
ylabel(ax2, 'CPU', 'Interpreter', 'latex', 'FontSize', 15);
title(ax2, ['$', title_str, '$'], 'Interpreter', 'latex', 'FontSize', 17);
legend(ax2, [h_rcd_t(3), h_rcgls_t(3)], {'RidgeGRCDU', 'RidgeRCGLSU'}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best', 'Box', 'on');
grid(ax2, 'off'); box(ax2, 'on');
set(ax2, 'Color', 'w', 'FontSize', 12, 'LineWidth', 1.2, 'XTick', log2_q_vec);
xlim(ax2, [0, max(log2_q_vec)]);