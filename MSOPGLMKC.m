function [YY,iter,obj] = MSOPGLMKC(KH,k,alpha,beta,gamma)
num = size(KH,2);
numker = size(KH,3);
maxIter = 100;

RP = repmat(eye(k),1,1,numker);
lambda = ones(numker,1)*sqrt(1/numker);

opt.disp = 0;
flag = 1;
iter = 0;
viewnum = numker+1;

G = cell(1,viewnum);
W = cell(1,viewnum);

for iv = 1:viewnum
    G{iv} = zeros(num,k);
    W{iv} = zeros(num,k);
end

if 10*k < num
    num_anchors = 10*k;
else
    num_anchors = floor(num/2);
end

accelerated_flag = 0;

rInd_temp = zeros(numker,num_anchors);
Z = zeros(num,num_anchors,numker);
apprxK_KKM = zeros(num,num,numker);

for i = 1:numker
    [rInd_temp(i,:),C] = recursiveNystrom_kernel( ...
        KH(:,:,i),num_anchors,accelerated_flag);
    Wsmall = KH(rInd_temp(i,:),rInd_temp(i,:),i);
    Wsmall = (Wsmall+Wsmall')/2;
    [U,D] = eig(Wsmall);
    d = diag(D);
    tol = max(1e-12,1e-10*max(abs(d)));
    invSqrt = zeros(num_anchors,1);
    positive = d > tol;
    invSqrt(positive) = 1./sqrt(d(positive));
    Z(:,:,i) = bsxfun(@times,C*U,invSqrt');
    apprxK_KKM(:,:,i) = Z(:,:,i)*Z(:,:,i)';
end

KH = apprxK_KKM;

K0 = zeros(num,num);
for p = 1:numker
    KH(:,:,p) = (KH(:,:,p)+KH(:,:,p)')/2;
    [Hp,~] = eigs(KH(:,:,p),k,'la',opt);
    K0 = K0+(1/numker)*KH(:,:,p);
    HP(:,:,p) = Hp;
end

beta_hi = beta;
YA = cell(1,numker);
H = cell(1,numker);

for i = 1:num
    H{1}(i,mod(i,k)+1) = 1;
end

YA{1} = H{1};
H{1} = H{1}./sqrt(sum(H{1}));

for v = 1:numker
    YA{v} = zeros(num,k);
    H{v} = H{1};
end

rho = 1e-5;
miu = 1.2;

J = zeros(num,k);
for p = 1:numker
    J = J+lambda(p)*(HP(:,:,p)*RP(:,:,p));
end

while flag
    iter = iter+1;

    for v = 1:numker
        Temp = Z(:,:,v)*(Z(:,:,v)'*H{v})+beta_hi*YA{v}./2;
        [U,~,V] = svd(Temp,"econ");
        H{v} = U*V';
    end
    clear Temp

    for v = 1:numker
        Temp = G{v}-W{v}./rho+(beta_hi/rho)*H{v};
        YA{v} = (Temp == max(Temp')');
    end
    clear Temp

    YB = G{viewnum}-W{viewnum}./rho+(alpha/rho)*J;
    Y = mySolving(YB);

    Y_tensor = cat(3,YA{:},Y);
    W_tensor = cat(3,W{:});
    Yv = Y_tensor(:);
    Wv = W_tensor(:);

    [Gv,objV] = wshrinkObj( ...
        Yv+1/rho*Wv,gamma/rho,[num,k,viewnum],0,3);
    AA(iter) = objV;
    G_tensor = reshape(Gv,[num,k,viewnum]);

    for iv = 1:viewnum
        G{iv} = G_tensor(:,:,iv);
        W{iv} = W{iv}+rho*(Y_tensor(:,:,iv)-G{iv});
    end

    clear G_tensor Y_tensor W_tensor Gv Yv Wv
    rho = min(miu*rho,1e10);

    CB = Y'*J;
    [Uh,~,Vh] = svd(CB,'econ');
    C = Uh*Vh';

    lambda = updateBetaOPLFMVC(HP,RP,Y,C);
    RP = updateWPOPLFMVC(HP,Y,C,lambda);

    J = zeros(num,k);
    for p = 1:numker
        J = J+lambda(p)*HP(:,:,p)*RP(:,:,p);
    end

    YY_YA = zeros(num,k);
    for v = 1:numker
        YY_YA = YY_YA+1/numker*YA{v};
    end

    YY_score = YY_YA+Y;
    [~,YY] = max(YY_score,[],2);

    term1 = 0;
    for i = 1:numker
        Zi = Z(:,:,i);
        term1 = term1+norm(Zi,'fro')^2-norm(Zi'*H{i},'fro')^2;
    end

    term2 = 0;
    for i = 1:numker
        term2 = term2+trace(H{i}'*YA{i});
    end
    term2 = -beta_hi*term2;

    term3 = -alpha*trace(C'*Y'*J);
    obj(iter) = term1+term2+term3+gamma*objV;

    if (iter > 2) && ...
            (abs((obj(iter)-obj(iter-1))/obj(iter)) < 1e-5 || ...
            iter > maxIter)
        flag = 0;
    end
end
