function beta = updateBetaOPLFMVC(HP,RP,Y,C)
numker = size(RP,3);
HHPRP = zeros(numker,1);
Hstar = Y*C;
for  p=1:numker
    HHPRP(p) = trace(Hstar'*(HP(:,:,p)*RP(:,:,p)));
end
beta = HHPRP./norm(HHPRP);
beta((beta<eps))=0;
beta = beta./norm(beta);