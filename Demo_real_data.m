% this file can produce Figures 5 and 6 in the manuscript
close all;
clear;
clc;

%%%
run_time = 10;       % average times
lambda = 0.05;

%% generated the matrix A using the data from LIBSVM
% dataset_name = 'real-sim';
% q = 2000;

% dataset_name = 'protein';
% q = 50;

% dataset_name = 'LEDGAR';
% q = 500;

dataset_name = 'gisette';
q = 50;

load([dataset_name, '.mat']); 
[n, d] = size(A);
if d <= n
    xLS = (A'*A + lambda*eye(d)) \ (A'*b);
else
    xLS = A'*((A*A' + lambda*eye(n)) \ b); 
end

%% parameter setup
opts.x_star = xLS; 
opts.x0 = zeros(d, 1); 
opts.y0 = zeros(n, 1);
opts.tol = 1e-4;

%%%
[err_A1_all, time_A1_all, err_A2_all, time_A2_all] = deal(cell(run_time, 1));
max_len1 = 0; max_len2 = 0;

%% executing "run_time" times of the algorithms
for jj = 1:run_time
    fprintf('Processing experimental data group %d/%d...\n', jj, run_time);
    
    %% run RidgeRCGLSU
    opts.max_iter = 50000;
    [~, err_A1, ~, time_A1] = My_RidgeRCGLSU(A, b, lambda, q, opts);
    err_A1_all{jj} = err_A1;
    time_A1_all{jj} = time_A1;
    max_len1 = max(max_len1, length(err_A1));
    
    %% run HImRidgeSketchU 
    opts.max_iter = 3000; 
    [~, err_A2, ~, time_A2] = My_HImRidgeSketchU(A, b, lambda, q, opts);
    err_A2_all{jj} = err_A2;
    time_A2_all{jj} = time_A2;
    max_len2 = max(max_len2, length(err_A2));
end

%% computational complexity (FLOPs) accumulation
flops_A1_all = cell(run_time, 1);
flops_A2_all = cell(run_time, 1);

if n >= d
    flops_iter_A1 = (4*q + 7)*n + 3*d + 6*q + 6; 
    flops_iter_A2 = (1/3)*(q^3) + (q^2 + 5*q + 2)*n + 2*d + 1.5*(q^2) + 3.5*q + 6;
else
    flops_iter_A1 = (4*q + 7)*d + 3*n + 8*q + 7; 
    flops_iter_A2 = (1/3)*(q^3) + (q^2 + 5*q + 2)*d + 2*n + 1.5*(q^2) + 4.5*q + 6;
end

for jj = 1:run_time
    flops_A1_all{jj} = max(0, (1:length(err_A1_all{jj})) - 1) * flops_iter_A1;
    flops_A2_all{jj} = max(0, (1:length(err_A2_all{jj})) - 1) * flops_iter_A2;
end

%%%
pts = 500; 
prep_t1 = median(cellfun(@(x) x(1), time_A1_all));
prep_t2 = median(cellfun(@(x) x(1), time_A2_all));


grid_f1 = linspace(0, max(cellfun(@max, flops_A1_all)), pts)';
grid_f2 = linspace(0, max(cellfun(@max, flops_A2_all)), pts)';
grid_t1 = linspace(prep_t1, max(cellfun(@max, time_A1_all)), pts)';
grid_t2 = linspace(prep_t2, max(cellfun(@max, time_A2_all)), pts)';


Err1_f_mat = zeros(pts, run_time); Err1_t_mat = zeros(pts, run_time);
Err2_f_mat = zeros(pts, run_time); Err2_t_mat = zeros(pts, run_time);


do_interp = @(raw_x, raw_y, grid_x) interp1(raw_x, log10(max(raw_y, 1e-16)), grid_x, 'linear', 'extrap');

%%%
for jj = 1:run_time
    [u_f1, id_f1] = unique(flops_A1_all{jj}, 'first');
    log_e = do_interp(u_f1, err_A1_all{jj}(id_f1), grid_f1);
    
    if err_A1_all{jj}(end) <= opts.tol
        log_e(grid_f1 > u_f1(end)) = -20; 
    else
        log_e(grid_f1 > u_f1(end)) = log10(max(err_A1_all{jj}(end), 1e-16));
    end
    Err1_f_mat(:, jj) = 10.^log_e;
    
    [u_t1, id_t1] = unique(time_A1_all{jj}, 'first');
    if length(u_t1) < 2
        u_t1 = [0, max(1e-6, u_t1(end))]; 
        err_vals = [err_A1_all{jj}(1), err_A1_all{jj}(end)]; 
    else
        err_vals = err_A1_all{jj}(id_t1); 
    end
    
    log_e = do_interp(u_t1, err_vals, grid_t1);
    if err_vals(end) <= opts.tol
        log_e(grid_t1 > u_t1(end)) = -20;
    else
        log_e(grid_t1 > u_t1(end)) = log10(max(err_vals(end), 1e-16));
    end
    Err1_t_mat(:, jj) = 10.^log_e;
    
    [u_f2, id_f2] = unique(flops_A2_all{jj}, 'first');
    log_e = do_interp(u_f2, err_A2_all{jj}(id_f2), grid_f2);
    
    if err_A2_all{jj}(end) <= opts.tol
        log_e(grid_f2 > u_f2(end)) = -20;
    else
        log_e(grid_f2 > u_f2(end)) = log10(max(err_A2_all{jj}(end), 1e-16)); 
    end
    Err2_f_mat(:, jj) = 10.^log_e;
    
    [u_t2, id_t2] = unique(time_A2_all{jj}, 'first');
    if length(u_t2) < 2
        u_t2 = [0, max(1e-6, u_t2(end))]; 
        err_vals = [err_A2_all{jj}(1), err_A2_all{jj}(end)]; 
    else
        err_vals = err_A2_all{jj}(id_t2); 
    end
    
    log_e = do_interp(u_t2, err_vals, grid_t2);
    if err_vals(end) <= opts.tol
        log_e(grid_t2 > u_t2(end)) = -20;
    else
        log_e(grid_t2 > u_t2(end)) = log10(max(err_vals(end), 1e-16));
    end
    Err2_t_mat(:, jj) = 10.^log_e;
end

%%%
get_stats = @(mat) deal(median(mat, 2), min(mat, [], 2), max(mat, [], 2), prctile(mat, 25, 2), prctile(mat, 75, 2));

[m1_f_e, min1_f_e, max1_f_e, q25_1f_e, q75_1f_e] = get_stats(Err1_f_mat);
[m2_f_e, min2_f_e, max2_f_e, q25_2f_e, q75_2f_e] = get_stats(Err2_f_mat);
[m1_t_e, min1_t_e, max1_t_e, q25_1t_e, q75_1t_e] = get_stats(Err1_t_mat);
[m2_t_e, min2_t_e, max2_t_e, q25_2t_e, q75_2t_e] = get_stats(Err2_t_mat);

%% visualization
colors = [1 0 0;  
          0 0 1]; 
alpha_minmax = 0.05; 
alpha_iqr = 0.15; 
lw = 1.5;            

plot_shaded = @(ax, x, y_min, y_max, y_25, y_75, y_med, col) ...
    [fill(ax, [x; flipud(x)], [y_min; flipud(y_max)], col, 'FaceAlpha', alpha_minmax, 'EdgeColor', 'none', 'HandleVisibility', 'off'); ...
     fill(ax, [x; flipud(x)], [y_25; flipud(y_75)], col, 'FaceAlpha', alpha_iqr, 'EdgeColor', 'none', 'HandleVisibility', 'off'); ...
     plot(ax, x, y_med, 'Color', col, 'LineWidth', lw)];
     
legend_names = {'HImRidgeSketchU', 'RidgeRCGLSU'};
y_upper_limit = 1; 

clean_dataset_name = strrep(dataset_name, '_', '\_');
final_title = sprintf('{\\tt %s}, $n=%d, d=%d$', clean_dataset_name, n, d);

%% RSE vs. FLOPs
fig1 = figure('Name', 'RSE vs FLOPs', 'Position', [100, 100, 550, 450], 'Color', 'w');
ax1 = axes('Parent', fig1); hold(ax1, 'on'); 
set(ax1, 'YScale', 'log'); 

h2 = plot_shaded(ax1, grid_f2, min2_f_e, max2_f_e, q25_2f_e, q75_2f_e, m2_f_e, colors(2,:));
h1 = plot_shaded(ax1, grid_f1, min1_f_e, max1_f_e, q25_1f_e, q75_1f_e, m1_f_e, colors(1,:));

ylim(ax1, [opts.tol, y_upper_limit]);
xlabel(ax1, 'Flops', 'FontSize', 15); 
ylabel(ax1, 'RSE', 'FontSize', 15); 
title(ax1, final_title, 'Interpreter', 'latex', 'FontSize', 17);
legend(ax1, [h2(3), h1(3)], legend_names, 'Location', 'northeast', 'FontSize', 14, 'Interpreter', 'none');
grid(ax1, 'off'); box(ax1, 'on');

%% RSE vs. CPU
fig2 = figure('Name', 'RSE vs CPU Time', 'Position', [700, 100, 550, 450], 'Color', 'w');
ax2 = axes('Parent', fig2); hold(ax2, 'on'); 
set(ax2, 'YScale', 'log'); 

h2_t = plot_shaded(ax2, grid_t2, min2_t_e, max2_t_e, q25_2t_e, q75_2t_e, m2_t_e, colors(2,:));
h1_t = plot_shaded(ax2, grid_t1, min1_t_e, max1_t_e, q25_1t_e, q75_1t_e, m1_t_e, colors(1,:));

xlim(ax2, [0, max(grid_t2(end), grid_t1(end))]); 
ylim(ax2, [opts.tol, y_upper_limit]);
xlabel(ax2, 'CPU', 'FontSize', 15); 
ylabel(ax2, 'RSE', 'FontSize', 15); 
title(ax2, final_title, 'Interpreter', 'latex', 'FontSize', 17);
legend(ax2, [h2_t(3), h1_t(3)], legend_names, 'Location', 'northeast', 'FontSize', 14, 'Interpreter', 'none');
grid(ax2, 'off'); box(ax2, 'on');

fprintf('Finished!\n');