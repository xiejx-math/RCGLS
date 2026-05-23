function [data, b, x0, x1, DEV, U, sigA, V ] = generate_data(n, d, varargin)

% generates data matrix and observation vector with given specifications
%
% Input: n (number of rows), d (number of columns), varargin (variable arguments)
% varargin options:
%   'singular values'        : double(n,1)
%   'correlated'             : boolean (default: false)
%   'matrix correlation'     : double, 0 < corr < 1 (default: 0.6)
%   'matrix deviation'       : double (default: 3)
%   'hansen singular values' : double scalar 1-10 (default: 0)
%   'hansen input'           : boolean (default: false)
%   'hansen max'             : double (default: 1e6)
%   'kappa'                  : double in log10 scale (default: 6)
%   'input dev'              : double (default: 1)
%   'prior dev'              : double (default: 0)
%   'prior mean'             : double (default: 0)
%   'noise level'            : double (default: 0)
%   'structure'              : boolean (default: false)
%   'plot'                   : boolean (default: false)
%
% Output: data, b, x0, x1, DEV, U, sigA, V
%

%% setting default values
KAPPA           = 6;
CORRELATED      = false;
CORRELATION     = .6;
A_DEV           = 3;
HANSEN_IN       = false;
HANSEN          = 0;
HANSEN_MAX      = 1e6;
INPUT_DEV       = 1;
PRIOR_MEAN      = 0;
PRIOR_DEV       = 0;
NOISE_LEVEL     = 0;
STRUCT          = false;
PLOT            = false;

%% parsing user inputs
for i = 1:2:length(varargin)
    str = varargin{i};
    switch str
        case 'singular values'
            sigA = varargin{i+1};
        case 'hansen singular values'
            HANSEN = varargin{i+1};
        case 'hansen input'
            HANSEN_IN = varargin{i+1};
        case 'hansen max'
            HANSEN_MAX = varargin{i+1};
        case 'kappa'
            KAPPA = varargin{i+1};
        case 'correlated'
            CORRELATED = varargin{i+1};
        case 'matrix correlation'
            CORRELATION = varargin{i+1};
        case 'matrix deviation'
            A_DEV = varargin{i+1};
        case 'input dev'
            INPUT_DEV = varargin{i+1};
        case 'prior mean'
            PRIOR_MEAN = varargin{i+1};
        case 'prior dev'
            PRIOR_DEV = varargin{i+1};
        case 'noise level'
            NOISE_LEVEL = varargin{i+1};
        case 'structure'
            STRUCT = varargin{i+1};
        case 'plot'
            PLOT = varargin{i+1};
        otherwise
            error(['generate data: un-identified input!!!:  ' varargin{i}])
    end
end

%% processing hansen data
if(HANSEN > 0)
    HAN_DATA = load('hansen_data.mat', 'HANSEN_CUT');
    HAN_DATA = HAN_DATA.HANSEN_CUT;
    sing_han = HAN_DATA{HANSEN,1};
    
    %%%% sampling hansen data
    n_han = size(sing_han,1);
    ind_han = linspace(1, n_han, min(n,d));
    sigA = interp1(1:n_han, sing_han, ind_han, 'linear');
    sigA = sigA/max(sigA);
    SIG_LOG = log10(sigA);
    SIG_REC = KAPPA*SIG_LOG/abs(SIG_LOG(end));
    sigA = (10.^SIG_REC)*HANSEN_MAX;
end

%% generating correlated data matrix
switch CORRELATED
    case 0 
        %%% uncorrelated case
        A = A_DEV*randn(n,d);
    case 1 
        %%% correlated case
        n1       = ones(n,d);
        
        %%%% generate covariance matrix strictly based on feature dimension d
        R        = corr(d, CORRELATION, A_DEV);
        T        = chol(R);
        
        %%%% uniformly right-multiply T to ensure feature correlation
        A        = randn(n,d)*T + n1;
end

[U,~,V]  = svd(A, 'econ');
A        = U * (sigA'.* V');

%% generating input vector x0
x1 = PRIOR_DEV*randn(d,1) + PRIOR_MEAN;
x1 = V*(V'*x1);

if(HANSEN_IN)
    han_x = size(HAN_DATA{HANSEN,2},1);
    if(han_x > 1)
        x_han = HAN_DATA{HANSEN,2};
        x0 = interp1(1:n_han, x_han, linspace(1, n_han, d), 'linear');
        x0 = reshape(x0, d, 1);
    else
        x0 = x1 + INPUT_DEV*randn(d,1);
    end
else
    x0 = x1 + INPUT_DEV*randn(d,1);
end
x0 = V*(V'*x0);

%% generating exact measurement
b0      = A*x0;
b0_nrm  = norm(b0);

%% adding noise
w = zeros(n, numel(NOISE_LEVEL));
DEV = zeros(numel(NOISE_LEVEL),1);
for i = 1:numel(NOISE_LEVEL)
    r           = randn(n,1);
    DEV(i)      = (NOISE_LEVEL(i)*b0_nrm)/norm(r);
    w(:,i)      = DEV(i)*r;
end

%% generating noisy measurement
b = b0 + w;
sigA = sigA(:);

%% setting output
if(STRUCT)
    data.size       = [n, d];
    data.A          = A;
    data.b          = b;
    data.x0         = x0;
    data.x1         = x1;
    data.dev        = DEV;
    data.MC         = numel(unique(NOISE_LEVEL));
    data.metric     = @(xx)(sqrt(sum((x0 - xx).^2, 1))/norm(x0));
    data.nlevel     = NOISE_LEVEL;
    data.matdev     = A_DEV;
    data.matcor     = CORRELATION;
else
    data = A;
end

%% plotting
if(PLOT)
    D = length(DEV);
    if D ==1
        D = [];
    end
    if(STRUCT)
        data.fig = figure('Name', 'Theoric Plot');
    else
        fig = figure('Name', 'Theoric Plot');
    end
    
    %%% coherence plot
    subplot(1,3,1);
    imagesc(normA(A), [0 1]); colorbar;
    title('coherence of A: $<\hat{a_i}, \hat{a_j}>$', 'Interpreter', 'latex');
    xlabel('column index i'); ylabel('column index j')
    axis square
    set(gca,'fontsize',16)
    
    %%% singular values plot
    subplot(1,3,2);
    semilogy(1:min(d,n), sigA, 'linewidth', 3); hold on;
    semilogy(1:min(d,n), abs(U'*b(:,[D, 1])), 'linewidth', .5); hold on;
    semilogy(1:min(d,n), DEV([D, 1]).*ones(min(d,n),1), 'linewidth', .5); hold on;
    title('Energy of Measurement'); xlabel('index i');
    if(numel(DEV) == 1)
        legend('\sigma_i', '|u_i^Ty|', '\sigma_\omega');
    else
        legend('\sigma_i', 'L:|u_i^Ty|', 'S:|u_i^Ty|', 'L:\sigma_\omega', 'S:\sigma_\omega', 'Location', 'northeastoutside');
    end
    set(gca,'fontsize',16)
    
    %%% energy of input and noise plot
    subplot(1,3,3);
    semilogy(1:min(d,n), sigA, 'linewidth', 3); hold on;
    semilogy(1:min(d,n), abs((V'*x0)), 'linewidth', 0.5)
    semilogy(1:min(d,n), abs((U'*w(:, [D, 1]))./sigA), 'linewidth', 0.5);
    title('Eneryg of Iput and Noise'); xlabel('index i');
    if(numel(DEV) == 1)
        legend('\sigma_i', '|v_i^Tx_0|', '|u_i^T\omega/\sigma_i|');
    else
        legend('\sigma_i', '|v_i^Tx_0|', 'L: |u_i^T\omega/\sigma_i|', 'S: |u_i^T\omega/\sigma_i|', 'Location', 'northeastoutside');
        set(gca,'fontsize',16)
        
    end
end

end

function R = corr(n, a, dev)

% generates correlation matrix
% R_jk = dev*a^|j-k|

P = zeros(n);
ind = 0:n-1;
for i = 1:n
    P(i:end,i) = ind(1:end-i+1);
end
P = P + P.';
R = dev*(a*ones(n)).^P;

end

function [ o ] = normA( A )

% calculates the coherence matrix of A

[n,d]   = size(A);
if(n>d)
    A = A ./ repmat(sqrt(sum(A.^2, 1)), n,1);
    o=A.'*A;
else
    A = A ./ repmat(sqrt(sum(A.^2, 2)), 1,d);
    o=A*A.';
end

end