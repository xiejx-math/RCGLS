function [x_out, err_hist, iter_hist, time_hist] = My_HImRidgeSketchU(A, b, lambda, q, opts)
% The RidgeSketch method using heuristic increasing momentum with uniform subsampling
% for solving the ridge regression problem
%            min_x ||Ax-b||^2 + lambda ||x||^2
%
% Input: the coefficient matrix A, the vector b, the regularization parameter lambda, block size q
% and opts
% opts.max_iter: the maximum number of iterations
% opts.tol: the stopping rule
% opts.x_star: the exact solution x*
%
% Output:
% x_out: the approximate solution
% err_hist: the error history
% iter_hist: the iteration history
% time_hist: the time history
%

t_algo_start = tic;
%% setting some parameters
[n, d] = size(A);
flag = exist('opts', 'var');
%%%% setting the max iteration
if (flag && isfield(opts, 'max_iter'))
    max_iter = opts.max_iter;
else
    max_iter = 2000000;
end
%%%% setting the tolerance
if (flag && isfield(opts, 'tol'))
    tol = opts.tol;
else
    tol = 1e-10;
end
%%%% determining what to use as the stopping rule
if (flag && isfield(opts, 'x_star'))
    strategy = 1;
    x_star = opts.x_star;
    norm_x_star_sq = norm(x_star)^2;
    if norm_x_star_sq == 0
        norm_x_star_sq = 1; 
    end
else
    strategy = 0;
    if d <= n
        norm_Atb_sq_plus_1 = norm(A' * b)^2 + 1;
    else
        norm_b_sq_plus_1 = norm(b)^2 + 1;
    end
end
%%%
max_records = max_iter + 1;
err_hist = zeros(max_records, 1);
iter_hist = zeros(max_records, 1);
time_hist = zeros(max_records, 1);
%% executing the HImRidgeSketchU method
if d <= n
    %%%% Option III
    x = zeros(d, 1);
    g = -b;
    
    dx = zeros(d, 1);
    dg = zeros(n, 1);
    
    %%%
    hist_cnt = 1;
    iter_hist(hist_cnt) = 0;
    time_hist(hist_cnt) = toc(t_algo_start);
    
    if strategy == 1
        err_hist(hist_cnt) = norm(x - x_star)^2 / norm_x_star_sq;
    else
        grad = A' * g + lambda * x;
        err_hist(hist_cnt) = norm(grad)^2 / norm_Atb_sq_plus_1;
    end
    
    %%%
    for k = 1:max_iter
        mom_beta = min(0.5, 1 - 1.005 / (0.005 * k + 1));
        
        idx = randperm(d, q); 
        A_J = A(:, idx);
        
        v_k = A_J' * g + lambda * x(idx);
        H_J = A_J' * A_J + lambda * eye(q);
        
        delta_k = - (H_J \ v_k); 
        
        dx_new = mom_beta * dx;
        dx_new(idx) = dx_new(idx) + delta_k;
        
        dg_new = mom_beta * dg + (A_J * delta_k);
        
        x = x + dx_new;
        g = g + dg_new;
        
        dx = dx_new;
        dg = dg_new;
        
        %%%% stopping rule
        hist_cnt = hist_cnt + 1;
        iter_hist(hist_cnt) = k;
        time_hist(hist_cnt) = toc(t_algo_start);
        
        if strategy == 1
            rel_err_sq = norm(x - x_star)^2 / norm_x_star_sq;
        else
            %%%% Note that we do not use this stopping rule during our test
            grad = A' * g + lambda * x;
            rel_err_sq = norm(grad)^2 / norm_Atb_sq_plus_1;
        end
        
        err_hist(hist_cnt) = rel_err_sq;
        if rel_err_sq < tol
            break; 
        end
    end
    x_out = x;
    
else
    %%%% Option IV
    At = A'; 
    y = zeros(n, 1);
    x = zeros(d, 1); 
    
    dy = zeros(n, 1);
    dx = zeros(d, 1);
    
    %%%
    hist_cnt = 1;
    iter_hist(hist_cnt) = 0;
    time_hist(hist_cnt) = toc(t_algo_start);
    
    if strategy == 1
        err_hist(hist_cnt) = norm(x - x_star)^2 / norm_x_star_sq;
    else
        grad = At' * x + lambda * y - b;
        err_hist(hist_cnt) = norm(grad)^2 / norm_b_sq_plus_1;
    end
    
    %%%
    for k = 1:max_iter
        mom_beta = min(0.5, 1 - 1.005 / (0.005 * k + 1));
        
        idx = randperm(n, q); 
        At_I = At(:, idx);
        
        v_k = At_I' * x + lambda * y(idx) - b(idx);
        H_I = At_I' * At_I + lambda * eye(q);
        
        delta_k = - (H_I \ v_k); 
        
        dy_new = mom_beta * dy;
        dy_new(idx) = dy_new(idx) + delta_k;
        
        dx_new = mom_beta * dx + (At_I * delta_k);
        
        y = y + dy_new;
        x = x + dx_new;
        
        dy = dy_new;
        dx = dx_new;
        
        %%%% stopping rule
        hist_cnt = hist_cnt + 1;
        iter_hist(hist_cnt) = k;
        time_hist(hist_cnt) = toc(t_algo_start);
        
        if strategy == 1
            rel_err_sq = norm(x - x_star)^2 / norm_x_star_sq;
        else
            %%%% Note that we do not use this stopping rule during our test
            grad = At' * x + lambda * y - b;
            rel_err_sq = norm(grad)^2 / norm_b_sq_plus_1;
        end
        
        err_hist(hist_cnt) = rel_err_sq;
        if rel_err_sq < tol
            break; 
        end
    end
    x_out = x;
end
%%%% setting output
err_hist = err_hist(1:hist_cnt); 
iter_hist = iter_hist(1:hist_cnt); 
time_hist = time_hist(1:hist_cnt);
end
