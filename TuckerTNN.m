function [Fseq, core, Out] = TuckerTNN(mcsequence, Data, opts)

    % TuckerTNN completion algorithm.
    % Souad Mohaoui

    %% Parameters
    if isfield(opts, 'maxit'), maxit = opts.maxit; else maxit = 500; end
    if isfield(opts, 'tol'), tol = opts.tol; else tol =1e-3; end 
     if isfield(opts, 'tol3'), tol3 = opts.tol3; else tol3 =1e-2; end 
   
    
    if isfield(opts, 'tol2'), tol2 = opts.tol2; else tol2 = 1e-7; end
    if isfield(opts, 'rank_inc'), rank_inc = opts.rank_inc; else rank_inc =1; end
    if isfield(opts, 'rank_max'), rank_max = opts.rank_max; else rank_max = [100, 40, 3]; end
    %if isfield(opts, 'init_rank'), init_rank = opts.init_rank; else init_rank = 10; end
    if isfield(opts, 'alpha'), alpha = opts.alpha; else alpha =.001; end
    
    if isfield(opts, 'rho_u'), rho_u = opts.rho_u; else rho_u = [.1,.1,.1]; end
    
    if isfield(opts, 'rho_v'), rho_v = opts.rho_v; else rho_v = [.1,.1,.1]; end
    if isfield(opts, 'rho_a'), rho_a = opts.rho_a; else rho_a = [.1,.1,.1]; end
    
    if isfield(opts, 'rho_g'), rho_g = opts.rho_g; else rho_g = 1.5; end
    if isfield(opts, 'rho_x'), rho_x = opts.rho_x; else rho_x = 1.5; end
    
        
  %% C3D to Tensor format
    n_f = mcsequence.nFrames;
    n_m = mcsequence.nMarkers;

    incomplete = reshape(mcsequence.data, n_f, n_m,3);
    M=reshape(Data, n_f, n_m,3);

    known = ~isnan(incomplete);
    
    Nway=[size(M,1), size(M,2), size(M,3)];
  
    % Initialize rank    
    if isfield(opts, 'R_init')
        R = opts.R_init;
    else
        ratio=0.01;
        R=AdapN_Rank(M,ratio);
    end
    % Data preprocessing
    R=[R(1),R(2),Nway(3)];
    data = zeros(Nway);
    data(known) = M(known); X=data;
    N = ndims(X) ; % Number of dimensions
  
   
    At = cell(1, N);
    Ap = cell(1, N);
    for n = 1:N-1   
       A{n} = randn(Nway(n), R(n));  A{n}=A{n}/norm(A{n});  
       At{n} = A{n}';         
    end
    A{N} = eye(Nway(N), R(N)); At{N} = A{N}; 
    core=randn(R);
    Ap=A;  corep=core;Xp=X;
    
    Y = ttm(core, A, 1:N);  % Update core tensor
    E_o = norm((Y(known)  - M(known) ),'fro');
    Out.rank = R; % Record the initial rank
    
    best_rmse = Inf;  % Initialize the best RMSE with a large value
    Cherr=zeros(1,maxit);
    RMSE=zeros(1,maxit);
    ReE=zeros(1,maxit);
    %% Main optimization loop
max_iter=50;
for k = 1:maxit
    % Update the facto
    for n = 1:N 
      Z = ttm(X, At, -n);  % Tensor-times-matrix for all modes except n
      Zn = Unfold(Z, Nway, n);
      coren = Unfold(core, R, n);
        
      % Update A_n using pseudoinverse
      A{n} = (Zn*coren'  + rho_a(n)*Ap{n}) * pinv(coren*coren' + rho_a(n));
      [Q, ~] = qr(A{n}, 0); 
      A{n} = Q;
      At{n} = A{n}';    
    end
    % Update the core tensor with the new orthonormal A_n matrices
     core = ttm(X, At, 1:N);  % Apply ttm over all modes        
     core=(core + rho_g*corep)/(1+rho_g);

   % compute X_1
    Y = ttm(core, A, 1:N);  % Update core tensor
    T1 = Unfold(Y, Nway, 1);
    Tp1 = Unfold(Xp, Nway, 1);
    gradf = @(x) ((1+rho_x)*x - T1 - rho_x*Tp1);
    X1 = APG_SVT(gradf, 1/alpha, max_iter, Tp1);
    
    % reconstruct tensor
    XX =  Fold(X1,Nway, 1);   
    X(~known) = XX(~known); 
    X(known) = M(known);    
    
    %%
    
    % Calculate RMSE over the entire tensor
    rmse = sqrt(mean((X(:) - M(:)).^2));  % RMSE over the entire tensor
    rel_err = norm(X(:) - M(:))/norm(M(:));
    rel_ch = norm(Xp(:) - X(:))/norm(Xp(:)); 
    E_n = norm(Y(known) - M(known),'fro');  % Residual between current tensor and original tensor
    Err_change=abs(1 - E_n /E_o);

     % Check if the current RMSE is an improvement
      if rmse < best_rmse
       % Update the best RMSE and save the best reconstruction
        best_rmse = rmse;
        best_X= X;  % Save the current reconstruction tensor
        best_core = core;  % Save the core tensor
        best_A = A;  % Save the factor matrices A
      end
    
 % Rank estimation
    if  Err_change < tol3 
        for n = 1:N
            if R(n) < rank_max(n)
                R(n) = R(n) + rank_inc;  % Increase Tucker rank for mode n
                rndx2=randn(size(A{n}, 1), rank_inc);   
                A{n} = [ A{n}, rndx2]; 
                [Q, ~] = qr(A{n}, 0); A{n} = Q; 
                At{n} = A{n}';      
            end
        end
        core = ttm(X, At, 1:N);  % Recompute core with updated ranks
    end
    Ap=A;   corep=core; Xp=X;
    E_o =E_n;  % Residual between current tensor and original tensor

   fprintf('%d: RMSE = %4.3e \n', k, rmse);
    
    % Check convergence
%     if rel_ch < tol2
%         fprintf('Converged at iteration %d\n', k);
%         break;
%     end
 
 
 RMSE(k)=rmse;
 Cherr(k)=rel_ch;
 ReE(k) =rel_err;
end
Out.Cherr=Cherr;
Out.RMSE=RMSE;
Out.Rer=ReE;
fprintf('Best RMSE achieved: %f\n', best_rmse);
X = best_X;
core = best_core;
A = best_A;


%% Tensor to C3D format 
F_matrix = reshape(X, [size(X, 1) 3*size(X, 2)]);
mcsequence.data =double(F_matrix);
Fseq=mcsequence;  
    %% Utility functions
    function W = Fold(W, dim, i)
        dim = circshift(dim, [1-i, 1-i]);
        W = shiftdim(reshape(W, dim), length(dim) + 1 - i);
    end

    function W = Unfold(W, dim, i)
        W = reshape(shiftdim(W, i-1), dim(i), []);
    end



end

