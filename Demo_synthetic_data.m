% this file can produce Figures 3 and 4 in the manuscript

close all;
clear;
clc;

%% experimental setup
experiment_mode = 1; % overdetermined case
% experiment_mode = 2; % underdetermined case
% experiment_mode = 3; % square case

run_time = 10;       % average times

lambda = 0.05;
%lambda = 0.005;

q = 50;             

%%%
switch experiment_mode
    case 1 % overdetermined case
        d_fixed = 900;
        n_list = 1000:500:5000;
        d_list = repmat(d_fixed, 1, length(n_list));
        x_axis_vals = n_list; 
        x_label_str = 'Number of rows (n)';
        info_str = sprintf('$d=%d$', d_fixed);
        
    case 2 % underdetermined case
        n_fixed = 900;
        d_list = 1000:500:5000;
        n_list = repmat(n_fixed, 1, length(d_list));
        x_axis_vals = d_list; 
        x_label_str = 'Number of columns (d)';
        info_str = sprintf('$n=%d$', n_fixed);
        
    case 3 % square case
        n_list = 400:200:2000;
        d_list = n_list;
        x_axis_vals = n_list; 
        x_label_str = 'The value of n';
        info_str = sprintf('$n=d$');
end

%%%
num_scales = length(n_list);
Iter_A1 = zeros(run_time, num_scales);
Iter_A2 = zeros(run_time, num_scales);
CPU_A1 = zeros(run_time, num_scales);
CPU_A2 = zeros(run_time, num_scales);

%% executing "run_time" times of the algorithms
for jj = 1:run_time
    fprintf('Processing experimental data group %d/%d...\n', jj, run_time);
    
    for ii = 1:num_scales
        n = n_list(ii);
        d = d_list(ii);
        
        %% generated data matrix A and observation b
        [A, b, x_true, x1, dev, U, sigA, V] = generate_data(n, d, ...
    'correlated',             true, ...
    'matrix correlation',     0.7,  ...
    'matrix deviation',       5,    ...
    'hansen singular values', 5,    ...
    'hansen max',             1,    ...
    'kappa',                  4,    ...
    'noise level',            0.1,  ...
    'hansen input',           true, ...
    'input dev',              1,    ...
    'prior dev',              0,    ...
    'prior mean',             0,    ...
    'structure',              false,...
    'plot',                   false);
        
        if d <= n
            xLS = (A'*A + lambda*eye(d)) \ (A'*b);
        else
            xLS = A'*((A*A' + lambda*eye(n)) \ b); 
        end
        
        %% parameter setup
        opts.x_star = xLS; 
        opts.x0 = zeros(d, 1); 
        opts.y0 = zeros(n, 1);
        opts.tol = 1e-10;
        opts.max_iter = 20000;
        
        %% run RidgeRCGLSU
        [~, ~, iter1, time1] = My_RidgeRCGLSU(A, b, lambda, q, opts);
        Iter_A1(jj, ii) = iter1(end); 
        CPU_A1(jj, ii) = time1(end); 
        
        %% run HImRidgeSketchU 
        [~, ~, iter2, time2] = My_HImRidgeSketchU(A, b, lambda, q, opts);
        Iter_A2(jj, ii) = iter2(end); 
        CPU_A2(jj, ii) = time2(end); 
    end
end

%% computational complexity (FLOPs) accumulation
flops_iter_A1 = zeros(1, num_scales);
flops_iter_A2 = zeros(1, num_scales);
for i = 1:num_scales
    nv = n_list(i);
    dv = d_list(i);
    
    if nv >= dv
        flops_iter_A1(i) = (4*q + 7)*nv + 3*dv + 6*q + 6; 
        flops_iter_A2(i) = (1/3)*(q^3) + (q^2 + 5*q + 2)*nv + 2*dv + 1.5*(q^2) + 3.5*q + 6;
    else
        flops_iter_A1(i) = (4*q + 7)*dv + 3*nv + 8*q + 7; 
        flops_iter_A2(i) = (1/3)*(q^3) + (q^2 + 5*q + 2)*dv + 2*nv + 1.5*(q^2) + 4.5*q + 6;
    end
end
FLOPs_A1 = Iter_A1 .* repmat(flops_iter_A1, run_time, 1);
FLOPs_A2 = Iter_A2 .* repmat(flops_iter_A2, run_time, 1);

%% visualization
colors = [1 0 0;  
          0 0 1]; 
markers = {'o', 's'};

get_stats = @(mat) deal(median(mat, 1), min(mat, [], 1), max(mat, [], 1), prctile(mat, 25, 1), prctile(mat, 75, 1));
alpha_minmax = 0.05; 
alpha_iqr = 0.15; 
lw = 1.5;            

plot_shaded = @(ax, x, y_min, y_max, y_25, y_75, y_med, col, marker) ...
    [fill(ax, [x, fliplr(x)], [y_min, fliplr(y_max)], col, 'FaceAlpha', alpha_minmax, 'EdgeColor', 'none', 'HandleVisibility', 'off'); ...
     fill(ax, [x, fliplr(x)], [y_25, fliplr(y_75)], col, 'FaceAlpha', alpha_iqr, 'EdgeColor', 'none', 'HandleVisibility', 'off'); ...
     plot(ax, x, y_med, 'Color', col, 'LineWidth', lw, 'Marker', marker, 'MarkerSize', 6, 'MarkerFaceColor', 'none')];

x_ax = x_axis_vals;
legend_names = {'HImRidgeSketchU', 'RidgeRCGLSU'};
final_title = info_str;

%% RSE vs. FLOPs
fig1 = figure('Name', 'Theoretical FLOPs', 'Position', [100, 100, 600, 500], 'Color', 'w');
ax1 = axes('Parent', fig1); hold(ax1, 'on');

[m1, min1, max1, q25_1, q75_1] = get_stats(FLOPs_A1); 
h1 = plot_shaded(ax1, x_ax, min1, max1, q25_1, q75_1, m1, colors(1,:), markers{1});

[m2, min2, max2, q25_2, q75_2] = get_stats(FLOPs_A2); 
h2 = plot_shaded(ax1, x_ax, min2, max2, q25_2, q75_2, m2, colors(2,:), markers{2});

xlabel(ax1, x_label_str, 'FontSize', 15); 
ylabel(ax1, 'FLOPs', 'FontSize', 15); 
title(ax1, final_title, 'Interpreter', 'latex', 'FontSize', 17);
set(ax1, 'XTick', x_ax); 
xlim(ax1, [min(x_ax), max(x_ax)]); 
legend(ax1, [h2(3), h1(3)], legend_names, 'Location', 'best', 'FontSize', 14, 'Interpreter', 'none');
grid(ax1, 'off'); box(ax1, 'on');

%% RSE vs. CPU
fig2 = figure('Name', 'CPU Time', 'Position', [750, 100, 600, 500], 'Color', 'w');
ax2 = axes('Parent', fig2); hold(ax2, 'on');

[m1_t, min1_t, max1_t, q25_1t, q75_1t] = get_stats(CPU_A1); 
h1_t = plot_shaded(ax2, x_ax, min1_t, max1_t, q25_1t, q75_1t, m1_t, colors(1,:), markers{1});

[m2_t, min2_t, max2_t, q25_2t, q75_2t] = get_stats(CPU_A2); 
h2_t = plot_shaded(ax2, x_ax, min2_t, max2_t, q25_2t, q75_2t, m2_t, colors(2,:), markers{2});

xlabel(ax2, x_label_str, 'FontSize', 15); 
ylabel(ax2, 'CPU', 'FontSize', 15); 
title(ax2, final_title, 'Interpreter', 'latex', 'FontSize', 17);
set(ax2, 'XTick', x_ax); 
xlim(ax2, [min(x_ax), max(x_ax)]); 
legend(ax2, [h2_t(3), h1_t(3)], legend_names, 'Location', 'best', 'FontSize', 14, 'Interpreter', 'none');
grid(ax2, 'off'); box(ax2, 'on');