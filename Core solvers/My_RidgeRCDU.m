function [x_out, err_hist, iter_hist, time_hist] = My_RidgeRCDU(A, b, lambda, q, opts)
% The generalized randomized coordinate descent method with uniform subsampling
% for solving the ridge regression problem
%            min_x ||Ax-b||^2 + lambda ||x||^2
%
% Input: the coefficient matrix A, the vector b, the regularization parameter lambda, block size q
% and opts
% opts.max_iter: the maximum number of iterations
% opts.tol: the stopping rule
% opts.x0: the initial vector x^0
% opts.y0: the initial vector y^0
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
%%%% setting the initial point
if (flag && isfield(opts, 'x0'))
    x0 = opts.x0;
else
    x0 = zeros(d, 1);
end
if (flag && isfield(opts, 'y0'))
    y0 = opts.y0;
else
    y0 = zeros(n, 1);
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
        norm_b_sq_plus_1 = lambda * norm(b)^2 + 1;
    end
end
%%%
max_records = max_iter + 1;
err_hist = zeros(max_records, 1);
iter_hist = zeros(max_records, 1);
time_hist = zeros(max_records, 1);
eps_tol = eps;
%% executing the RidgeRCDU method
if d <= n
    %%%% n \geq d
    x = x0;
    g = b - A * x;
    
    %%%
    hist_cnt = 1;
    iter_hist(hist_cnt) = 0;
    time_hist(hist_cnt) = toc(t_algo_start);
    
    if strategy == 1
        err_hist(hist_cnt) = norm(x - x_star)^2 / norm_x_star_sq;
    else
        grad = -A' * g + lambda * x;
        err_hist(hist_cnt) = norm(grad)^2 / norm_Atb_sq_plus_1;
    end
    
    %%%
    for k = 1:max_iter
        idx = randperm(d, q);
        
        U = A(:, idx);
        w = U' * g - lambda * x(idx);
        v = U * w;
        
        norm_w_sq = w' * w;        
        den = (v' * v) + lambda * norm_w_sq; 
        if den > eps_tol
            mu = norm_w_sq / den;
        else
            mu = 0; 
        end
        
        x(idx) = x(idx) + mu * w;
        g = g - mu * v;
        
        %%%% stopping rule
        hist_cnt = hist_cnt + 1;
        iter_hist(hist_cnt) = k;
        time_hist(hist_cnt) = toc(t_algo_start);
        
        if strategy == 1
            rel_err_sq = norm(x - x_star)^2 / norm_x_star_sq;
        else
            %%%% Note that we do not use this stopping rule during our test
            grad = -A' * g + lambda * x;
            rel_err_sq = norm(grad)^2 / norm_Atb_sq_plus_1;
        end
        
        err_hist(hist_cnt) = rel_err_sq;
        if rel_err_sq < tol
            break;
        end
    end
    x_out = x;
    
else
    %%%% n < d
    sqrt_lam = sqrt(lambda);
    At = A'; 
    y = y0;
    x = (At * y) / sqrt_lam;
    
    %%%
    hist_cnt = 1;
    iter_hist(hist_cnt) = 0;
    time_hist(hist_cnt) = toc(t_algo_start);
    
    if strategy == 1
        err_hist(hist_cnt) = norm(x - x_star)^2 / norm_x_star_sq;
    else
        grad = At' * (sqrt_lam * x) + lambda * y - sqrt_lam * b;
        err_hist(hist_cnt) = norm(grad)^2 / norm_b_sq_plus_1;
    end
    
    %%%
    for k = 1:max_iter
        idx = randperm(n, q);
        
        U = At(:, idx); 
        w = sqrt_lam * (b(idx) - U' * x) - lambda * y(idx);
        v = U * w;
        
        norm_w_sq = w' * w;        
        den = (v' * v) + lambda * norm_w_sq; 
        if den > eps_tol
            mu = norm_w_sq / den;
        else
            mu = 0;
        end
        
        y(idx) = y(idx) + mu * w;
        x = x + (mu / sqrt_lam) * v;
        
        %%%% stopping rule
        hist_cnt = hist_cnt + 1;
        iter_hist(hist_cnt) = k;
        time_hist(hist_cnt) = toc(t_algo_start);
        
        if strategy == 1
            rel_err_sq = norm(x - x_star)^2 / norm_x_star_sq;
        else
            %%%% Note that we do not use this stopping rule during our test
            grad = At' * (sqrt_lam * x) + lambda * y - sqrt_lam * b;
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

