function WP = updateWPOPLFMVC(HP,Y,C,lambda)

k = size(HP,2);
numker = size(HP,3);
WP = zeros(k,k,numker);

Hstar = Y*C;
for p = 1 : numker
    Tp = lambda(p)*HP(:,:,p)'*Hstar;
    [Up,Sp,Vp] = svd(Tp,'econ');
    WP(:,:,p) = Up*Vp';
end