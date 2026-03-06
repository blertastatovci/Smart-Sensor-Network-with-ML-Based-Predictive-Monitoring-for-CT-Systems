function permutation_importance(modelsMat, csvPath, metricName, kTop)
% Permutation Importance për SVM & ANN (test set 80/20, rng=42).
% metricName: 'AUPRC' (default) | 'AUROC' | 'F1'
% Përdor:
%   permutation_importance('pdm_models.mat','FeatureTable_10min_2024_2025.csv','AUPRC',15)

if nargin<1 || isempty(modelsMat), modelsMat = fullfile(pwd,'pdm_models.mat'); end
if nargin<2 || isempty(csvPath),   csvPath   = fullfile(pwd,'FeatureTable_10min_2024_2025.csv'); end
if nargin<3 || isempty(metricName), metricName = 'AUPRC'; end
if nargin<4 || isempty(kTop),       kTop = 15; end
metricName = upper(metricName);

S = load(modelsMat); M = S.modelsFinal;

% ---------- Lexo CSV (ruaj emrat origjinalë) ----------
T = readtable(csvPath, 'VariableNamingRule','preserve');

% ---------- Rindërto 'label' nëse mungon ----------
if ~ismember('label', T.Properties.VariableNames)
    T.label = makeSurrogateLabelsLikePipeline(T);
end

% ---------- Map i emrave të veçorive nga modeli -> CSV ----------
modelVars = M.vars(:)';                                 % nga modeli
csvVars   = T.Properties.VariableNames;                 % në CSV
mapped    = modelVars;

% map 1: përputhja direkte
miss = ~ismember(mapped, csvVars);
if any(miss)
    % ndërto indeks për emrat "të pastruar"
    clean = @(s) regexprep(lower(s), '\W', '_');        % heq / hapësira etj.
    csvMap = containers.Map;
    for i=1:numel(csvVars)
        csvMap(clean(csvVars{i})) = csvVars{i};
    end
    for i=1:numel(modelVars)
        if ~ismember(modelVars{i}, csvVars)
            key = clean(modelVars{i});
            if isKey(csvMap, key)
                mapped{i} = csvMap(key);
            end
        end
    end
end

% verifikim final
stillMiss = setdiff(modelVars, mapped);
if ~isempty(stillMiss)
    error("Nuk u gjetën në CSV këto veçori nga modeli:\n  - %s", strjoin(stillMiss, '\n  - '));
end

% ---------- Ndërto X, y dhe standardizo si në scaler ----------
X = T{:, mapped};
y = double(T.label(:) > 0);

X = fillmissing(X,'constant',0);
mu = M.scaler.mu; sg = M.scaler.sigma; sg(sg==0)=1;
Xs = (X - mu) ./ sg;

% ---------- Split 80/20 si pipeline ----------
rng(42);
n   = size(Xs,1);
idx = randperm(n);
nTr = round(0.8*n);
te  = idx(nTr+1:end);

Xte = Xs(te,:); yte = y(te);

% ---------- Baseline scores ----------
[~,scoreSVM] = predict(M.svm, Xte); baseSVM = scoreSVM(:,2);
predSVM = baseSVM >= 0.5;

scoreANN = M.ann(Xte')'; baseANN = scoreANN;
predANN = baseANN >= 0.5;

base_svm = metric_of(yte, baseSVM, predSVM, metricName);
base_ann = metric_of(yte, baseANN, predANN, metricName);

% ---------- Permutation loop ----------
p = size(Xte,2);
impSVM = zeros(p,1);
impANN = zeros(p,1);

for j = 1:p
    Xperm = Xte;
    Xperm(:,j) = Xperm(randperm(size(Xperm,1)), j); % permuto vetëm kolonën j

    % SVM
    [~,sc] = predict(M.svm, Xperm); sc = sc(:,2);
    yhat = sc >= 0.5;
    s = metric_of(yte, sc, yhat, metricName);
    impSVM(j) = base_svm - s;

    % ANN
    sc2 = M.ann(Xperm')'; yhat2 = sc2 >= 0.5;
    s2 = metric_of(yte, sc2, yhat2, metricName);
    impANN(j) = base_ann - s2;
end

% ---------- Vizatim: top-k ----------
[sv,si] = sort(impSVM,'descend');
[av,ai] = sort(impANN,'descend');

kS = min(kTop, numel(si)); kA = min(kTop, numel(ai));
namesS = mapped(si(1:kS));
namesA = mapped(ai(1:kA));

figure('Color','w','Position',[100 100 1100 460]);

subplot(1,2,1);
barh(sv(1:kS)); grid on;
set(gca,'YTick',1:kS,'YTickLabel',namesS,'YDir','reverse');
xlabel(sprintf('Drop in %s (higher = more important)', metricName));
title(sprintf('SVM – Permutation Importance (baseline %s = %.3f)', metricName, base_svm));

subplot(1,2,2);
barh(av(1:kA)); grid on;
set(gca,'YTick',1:kA,'YTickLabel',namesA,'YDir','reverse');
xlabel(sprintf('Drop in %s (higher = more important)', metricName));
title(sprintf('ANN – Permutation Importance (baseline %s = %.3f)', metricName, base_ann));

sgtitle(sprintf('Permutation Importance on test set (80/20) – metric: %s', metricName));

end

% ===================== helpers =====================

function m = metric_of(yTrue, score, yPred, metricName)
switch upper(metricName)
    case 'F1'
        tp = sum(yPred==1 & yTrue==1);
        fp = sum(yPred==1 & yTrue==0);
        fn = sum(yPred==0 & yTrue==1);
        prec = tp / max(tp+fp,1);
        rec  = tp / max(tp+fn,1);
        m = 2*prec*rec / max(prec+rec, eps);
    case 'AUROC'
        try
            [~,~,~,auc] = perfcurve(yTrue, score, 1);
        catch, auc = NaN; end
        m = auc;
    otherwise % 'AUPRC'
        try
            [~,~,~,auprc] = perfcurve(yTrue, score, 1, 'xCrit','reca','yCrit','prec');
        catch, auprc = NaN; end
        m = auprc;
end
end

function lbl = makeSurrogateLabelsLikePipeline(A)
    z1 = zsafe(A,"events_total");

    sevHigh = 0;
    if ismember("sev_4_cnt", A.Properties.VariableNames), sevHigh = sevHigh + A.("sev_4_cnt"); end
    if ismember("sev_7_cnt", A.Properties.VariableNames), sevHigh = sevHigh + A.("sev_7_cnt"); end
    if isequal(sevHigh,0)
        sevCols = A.Properties.VariableNames(contains(A.Properties.VariableNames,"sev_") & contains(A.Properties.VariableNames,"_cnt"));
        for i=1:numel(sevCols)
            tok = regexp(sevCols{i}, 'sev_(\d+)_cnt', 'tokens','once');
            if ~isempty(tok) && str2double(tok{1})>=4
                sevHigh = sevHigh + A.(sevCols{i});
            end
        end
    end
    sevRatio = sevHigh ./ max(A.("events_total"),1);
    z2 = zscore_fallback(sevRatio);

    z3    = zsafe(A,"burst_len_max_1m");
    zTilt = zscore_fallback(abs(getOrZero(A,"mean_gantry_tilt")));

    lbl = double((z1>2) | (z2>2) | (z3>2) | (zTilt>2));
end

function z = zsafe(A,name)
    v = getOrZero(A,name);
    z = zscore_fallback(v);
end

function v = getOrZero(A,name)
    if ismember(name, A.Properties.VariableNames), v = A.(name); else, v = zeros(height(A),1); end
end

function z = zscore_fallback(x)
    x = double(x);
    m = mean(x,'omitnan'); s = std(x,[],'omitnan'); if s==0 || isnan(s), s = 1; end
    z = (x - m) ./ s; z(isnan(z)) = 0;
end
