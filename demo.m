clear
clc
warning off

dataName = 'UCI_DIGIT';
alphaset = 2.^(-1:1:7);
betaset  = 2.^(-1:1:7);
gammaset = 2.^(-1:1:7);

addpath(genpath('./'));

load([dataName, '_Kmatrix'], 'KH', 'Y');

true_Y = double(Y(:));
numclass = length(unique(true_Y));
true_Y(true_Y == 0) = numclass;

KH = kcenter(KH);
KH = knorm(KH);

bestACC = -inf;
runIndex = 0;
totalRuns = numel(alphaset)*numel(betaset)*numel(gammaset);

for alpha = alphaset
    for beta = betaset
        for gamma = gammaset
            runIndex = runIndex+1;
            s = RandStream('mt19937ar', 'Seed', 2);
            RandStream.setGlobalStream(s);

            tic;
            [Yout, ~, ~] = MSOPGLMKC( ...
                KH, numclass, alpha, beta, gamma);
            tcost = toc;

            Yout = double(Yout(:));
            if length(Yout) ~= length(true_Y)
                error('Yout length = %d, true_Y length = %d.', ...
                    length(Yout), length(true_Y));
            end

            res = ClusteringMeasure(true_Y, Yout);
            ACC = res(1);
            NMI = res(2);
            Purity = res(3);
            ARI = res(7);

            if ACC > bestACC
                bestACC = ACC;
                bestNMI = NMI;
                bestPurity = Purity;
                bestARI = ARI;
                bestAlpha = alpha;
                bestBeta = beta;
                bestGamma = gamma;
                bestTime = tcost;
            end

            fprintf(['[%d/%d] alpha = %g | beta = %g | gamma = %g | ', ...
                'ACC = %.4f | NMI = %.4f | Purity = %.4f | ', ...
                'ARI = %.4f | Time = %.2fs | ', ...
                'Best ACC = %.4f\n'], ...
                runIndex, totalRuns, alpha, beta, gamma, ...
                ACC, NMI, Purity, ARI, tcost, ...
                bestACC);
        end
    end
end

fprintf(['Best | Dataset: %s | alpha = %g | beta = %g | gamma = %g\n', ...
    'ACC = %.4f | NMI = %.4f | Purity = %.4f | ', ...
    'ARI = %.4f | Time = %.2fs\n'], ...
    dataName, bestAlpha, bestBeta, bestGamma, ...
    bestACC, bestNMI, bestPurity, bestARI, bestTime);
