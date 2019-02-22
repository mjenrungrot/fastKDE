if ~exist('data_ofiles','var')
    data_ofiles = 'data0';
end
if ~exist('DIMS','var')
    DIMS = 2; % dimension of data
end

N_MIX = 10; % number of mixture components
N_SAMP_MIX = 1E5; % samples per mixture component
N_SAMP = N_SAMP_MIX*N_MIX; % total number of samples


sigmas = zeros(DIMS,DIMS,N_MIX);
mus = zeros(DIMS,N_MIX);
samples = nan(N_SAMP,DIMS);
if DIMS==2
    [X1,X2] = meshgrid(linspace(-12,12,200),linspace(-12,12,200));
    Z = zeros(size(X1));
end
for i=1:N_MIX
    A = (rand+1)*2*rand(DIMS)-1; 
    A = A*A'+diag(rand(DIMS,1));
    sigmas(:,:,i) = A;
    B = 10*rand(1,DIMS)-5;
    mus(:,i) = B;
    if DIMS==2
        Z = Z + reshape(mvnpdf([X1(:) X2(:)],B, A),size(X1))/N_MIX;
    end
    samples((i-1)*(N_SAMP_MIX)+1:i*(N_SAMP_MIX),:) = mvnrnd(B, A,N_SAMP_MIX);
end
if DIMS==2 % plot if in 2d
    h1 = figure;
    r=randi(N_SAMP,5000,1);
    scatter(samples(r,1),samples(r,2),'+');
    axis([-12 12 -12 12]);
    saveas(h1, ['../Data/' data_ofiles '-sample.fig']);
    h2 = figure; 
    contourf(X1,X2,Z);
    saveas(h2, ['../Data/' data_ofiles '-contour.fig']);
    h3 = figure; 
    surf(X1,X2,Z);
    saveas(h3, ['../Data/' data_ofiles '-surf.fig']);
end
samplesr = samples(randperm(length(samples)),:);
save(['../Data/' data_ofiles],'sigmas','mus','samples','samplesr');
dlmwrite(['../Data/' data_ofiles '.txt'], samplesr, 'delimiter', '\t', 'precision', 5);