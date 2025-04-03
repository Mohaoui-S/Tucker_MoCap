function [X] = APG_SVT(grad_f, lambda, max_iter,  X0)
   
    
    % Initialize variables
    X = X0;
    Y = X;
    t = 1;L=1;
    tol=.0001;
    %L = 1; % This should be estimated based on the specific problem
    
    for k = 1:max_iter
        % Gradient descent step
        G = Y - (1 / L) * grad_f(Y);
        
        % Proximal operator of the nuclear norm
        X_new = prox_NN(G, lambda / L);
        
        % Update t and Y
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;
        Y = X_new + ((t - 1) / t_new) * (X_new - X);
        
        % Check convergence
        if norm(X_new - X, 'fro') / norm(X, 'fro') < tol
            break;
        end
        
        % Update variables for next iteration
        X = X_new;
        t = t_new;
    end
end


