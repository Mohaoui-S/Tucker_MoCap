function Y = ttm(X, U, varargin)
    % TTM Tensor times matrix (n-mode product)
    %
    % Y = ttm(X, U, n) computes the n-mode product of tensor X with matrix U.
    % If U is a cell array of matrices, the product is computed for each mode
    % specified in n, or for all modes if n = -1.
    %
    % Y = ttm(X, U, [n1, n2, ...]) computes the n-mode products for the modes
    % specified in the vector [n1, n2, ...].
    %
    % Y = ttm(X, U, -n) multiplies by all factors except the nth mode.

    N = ndims(X); % Number of dimensions in X
    
    if nargin == 2
        % Default case: apply the product to all modes
        n = 1:N;
        tflag = '';
    elseif nargin == 3
        % Specified modes or transpose flag
        if ischar(varargin{1})
            tflag = varargin{1};
            n = 1:N;
        else
            n = varargin{1};
            tflag = '';
        end
    elseif nargin == 4
        % Specified modes and transpose flag
        n = varargin{1};
        tflag = varargin{2};
    end

    % Handle the case where n is negative
    if numel(n) == 1 && n < 0
        n = setdiff(1:N, -n); % Use all modes except the -n mode
    end

    % If U is a cell array, multiply in each specified mode
    if iscell(U)
        Y = X;
        for i = 1:length(n)
            Y = ttm(Y, U{n(i)}, n(i), tflag);
        end
        return;
    end

    % Single matrix multiplication in a specific mode
    if numel(n) ~= 1 || (n < 1) || (n > N)
        error('Dimension N must be between 1 and NDIMS(X).');
    end

    sz = size(X); % Original size of the tensor
    order = [n, 1:n-1, n+1:N]; % Order to bring the nth dimension first
    newdata = permute(X, order); % Permute dimensions to bring nth dimension first
    newdata = reshape(newdata, sz(n), []); % Reshape to matrix for multiplication

    % Apply the matrix multiplication
    if strcmp(tflag, 't')
        newdata = U' * newdata;
    else
        newdata = U * newdata;
    end

    % Reshape the result back to the correct size
    newsz = [size(newdata, 1), sz(1:n-1), sz(n+1:N)];
    Y = reshape(newdata, newsz);
    Y = ipermute(Y, order); % Permute dimensions back to original order
end
