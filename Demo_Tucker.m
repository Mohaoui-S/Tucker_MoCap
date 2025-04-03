%% Motion Capture Data Processing with Tensor Completion
% This script processes motion capture data and demonstrates tensor completion techniques
% for recovering missing markers in motion capture sequences.


% Clear workspace and initialize

clear all;
close all;
clc;

%% Add required paths
addpath('mocaptoolbox/mocaptoolbox-1.9.8');
addpath('mocaptoolbox/private');
addpath('MoCapToolboxExtension');
addpath('tensor_toolbox-master');
addpath('aLgorithms_CP');
addpath('data');
rng(80);  % Set the random seed for reproducibility

%% Load and prepare motion capture data
% Available datasets:
% - 'HDM_mm_03-02_01_120.c3d': Kicking and punching

dataFile = 'HDM_mm_01-02_03_120.c3d';  
original = mcread(['data/' dataFile]);

%check if there are Nans in your data 
if any(any(isnan(original.data)))
    fprintf('There are already missing data in your file.\n');
    Mrk_miss = find(isnan(original.data));
    original.data(Mrk_miss)=0;
end

%% Display original data
p = mcinitanimpar;
p.conn = mcautomaticbones2(original);    
fprintf('Automatic connection...\n');

% Uncomment to visualize original data
% p.scrsize = [1900/2 800];
% myfighandle = figure(1);
% mc3dplot(original,p,myfighandle);
% title('Original data');

%% Simulate gaps on your data:
nmissing =20;  %nombre of missing markers
Freq=original.freq;
Fmissing=100; %nombre of missing frames
Gapsec = Fmissing/Freq; %duration of gaps
%Gapsec =1.2; % Or you can fix the duration of gaps
%Fmissing = Gapsec*Freq;


% Multiple gaps configuration
multiple_gaps = true; % Enable multiple gaps per marker trajectory
Number_gaps =10;      % Maximum possible number of gaps per marker


incomplete = original;
indices = zeros(size(incomplete.data));
n_frames=incomplete.nFrames;
n_markers = incomplete.nMarkers;
fps = incomplete.freq;

% Randomly select markers for introducing gaps
j=randperm(n_markers);
j=j(1:nmissing);
Gapframe=round(Gapsec*fps);
marker_names = original.markerName;
gaps=ones(n_markers,1);

% Configure number of gaps per marker
if multiple_gaps    
    gaps=round(rand(n_markers,1)*Number_gaps);
    gaps(gaps<1)=1;
end


% Create gaps in the data
ind_perm=[];
id1 = 1+floor(rand(1)*(n_frames - Gapsec*fps));
id2 = min(n_frames,id1+Gapframe);
for m=j
 ind_perm{m} = zeros(incomplete.nFrames,1);
 for k=1:gaps(m)
      if multiple_gaps
          %gaps at different times
          %id1 = 1+round(rand(1) * nframes * scaling_factor);
           id1 = 1+round(rand(1)*n_frames*4/5.1);
           id2 = min(n_frames,id1+Gapframe);
      end
      ind_perm{m}(id1:id2)=true;
      indices(id1:id2,(3*m-2):(3*m))=true;  %indexes of fake gaps
  end
  ind_perm{m}=logical(ind_perm{m});
end

indices = logical(indices);
incomplete.data(indices)=nan;
data=original.data;


% Uncomment to visualize the gap pattern
% figure(6);imagesc(indices);
% colormap(jet); % Apply a color map
% % Set a specific color for NaN values, like gray
% 
% set(gca, 'Color', [0.8, 0.8, 0.8]); % Light gray for NaNs
% %set(gca, 'Color', [0.85, 0.85, 0.85]); % Sets the background to light gray for NaNs
% title('Simulated gaps');
% xlabel('Markers (interleaved x,y,z)');
% ylabel('Frames');

%%Display incomplete data
%myfighandle = figure(3);
%mc3dplot(incomplete,p,myfighandle);
%title('incomplete data');


Data=original.data;
gap_length=Fmissing;

%% TuckerTNN Method
fprintf('\n=== Running TuckerTNN Method ===\n');
 
% Configure parameters for TuckerTNN
opts.alpha = .01;
opts.mu=0;
opts.rho_u=[.1,.1,.1];
opts.rho_v=[.1,.1,.1];
opts.rho_a=[.1,.1,.1];
opts.rho_x=.001;
opts.rho_g=.01; % [1.1,1.1,1.1]; for M2/M1
opts.maxit =1000; 
opts.tol=1e-3; %1e-3
opts.rank_max = [100, 40, 3];
opts.tol3=1e-4;

% Execute TuckerTNN algorithm
t0 = tic;
[TM_rec, core, Out] = TuckerTNN(incomplete,Data,opts); 
timeTM= toc(t0);

% Calculate performance metrics
X_t=TM_rec.data;
TM_RMSE = sqrt(mean((X_t(:) - Data(:)).^2)); 
TM_AvE = mean(abs(X_t(:) - Data(:))); 
TM_ReE= (norm(X_t - Data))/norm(Data);
TM_STD= std(X_t(:) - Data(:));
% TMRMSE(i)=TM_RMSE; 
% TMAvE(i)=TM_AvE;
fprintf ('TM_RMSE %d, average_error %d, std  %0.4f \r',TM_RMSE, TM_AvE,TM_STD)
RMSE_Frame = zeros(size(Data, 1), 1);
for frame = 1:size(Data, 1)
  RMSE_Frame(frame) = sqrt(mean((Data(frame,:) - X_t(frame,:)).^2));
end

%% Tucker_Tu Method
fprintf('\n=== Running Tucker_Tu Method ===\n');

% Configure parameters for Tucker_TEM
opts.mu=0;
opts.rho_a=[.1,.1,.1];
opts.rho_x=.5;%.1
opts.rho_g=[1.5,1.5,1.5]; % [1.1,1.1,1.1]; for M2/M1
opts.maxit =1000; 
opts.tol=1e-3; %1e-3
opts.rank_max = [100, 40, 3];
opts.tol3=1e-4;

% Execute Tucker_Tu algorithm
t0 = tic;
[Tu_rec,out] =  Tucker_TEM(incomplete,Data,opts); 
timeT= toc(t0);

% Calculate performance metrics
X_t=Tu_rec.data;
Tu_RMSE = sqrt(mean((X_t(:) - Data(:)).^2)); 
Tu_AvE = mean(abs(X_t(:) - Data(:))); 
Tu_ReE= (norm(X_t - Data))/norm(Data);
Tu_STD= std(X_t(:) - Data(:));


%% Compare results
fprintf('\n=== Comparison of Methods ===\n');
fprintf ('Tu_RMSE %d, average_error %d, std  %0.4f \r',Tu_RMSE, Tu_AvE,Tu_STD)
fprintf ('TM_RMSE %d, average_error %d, std  %0.4f \r',TM_RMSE, TM_AvE,TM_STD)

%% CP Methods (Commented Out - Uncomment to Use)
%
% fprintf('\n=== Running CP Method ===\n');
% %% CP 
% opts.Rmax =Inf; 
% opts.rho =[0, 0, 0];
% opts.eps= 1e-3;  
% opts.maxiter= 1000;        
% opts.tol= 1e-4;        
% opts.tol_cnv=1e-6;
% opts.gama= 0; %  opts.gama= .001, 0.01; % for spaseCP and smoothCP
% opts.model='Cp'; % 'sparse';  'smooth';
% 
% t0 = tic;
% [CP_rec,out]  = CP_algs(incomplete,Data,opts);
% timeCP= toc(t0);
% 
% Recv=CP_rec.data;
% CP_RMSE = sqrt(mean((Recv(:) - Data(:)).^2));
% CP_ReE= (norm(Recv - Data))/norm(Data);
% CP_AvE = mean(abs(Recv(:) - Data(:)));
% CP_STD= std(Recv(:) - Data(:));
% 
% %% SPARSITY
% opts.rho     = [.001, .001, .001]; % tune this parameter [.0001, 0.0001, .0001];
% opts.gama = .001; % tune this parameter 
% opts.model='sparse';
% t0 = tic;
% [SP_rec,out]  = CP_algs(incomplete,Data,opts);
% timeSP= toc(t0);
% 
% Recv=SP_rec.data;
% SP_RMSE = sqrt(mean((Recv(:) - Data(:)).^2));
% SP_ReE= (norm(Recv - Data))/norm(Data);
% SP_AvE = mean(abs(Recv(:) - Data(:)));
% SP_STD= std(Recv(:) - Data(:));
% 
% 
% 
% fprintf ('CP_RMSE %d, average_error %d, std %0.4f \r',CP_RMSE, CP_AvE, CP_STD)
% fprintf ('SP_RMSE %d, average_error %d, std  %0.4f \r',SP_RMSE,SP_AvE, SP_STD)
%end
%% End of script

