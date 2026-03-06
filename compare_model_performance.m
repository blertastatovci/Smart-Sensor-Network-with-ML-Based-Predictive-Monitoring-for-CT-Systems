function compare_model_performance(modelsMat, csvPath)
% Krahasim SVM vs ANN në test set (80/20 si pipeline).
% Rindërton 'label' nëse mungon në CSV (surrogate rule si në pipeline).
%
% Përdorim:
%   compare_model_performance('pdm_models.mat','FeatureTable_10min_2024_2025.csv');

if nargin<1, modelsMat = fullfile(pwd,'pdm_models.mat'); end
if nargin<2, csvPath   = fullfile(pwd,'FeatureTable_10min_2024_2025.csv'); end

S = load(modelsMat); M = S.modelsFinal;

% --- Lexo CSV duke ruajtur emrat origjinalë të kolonave
T = readtable(csvPath, 'VariableNamingRule','preserve');

% --- Nëse mungon 'label', e krijojmë sipas rregullit të pipeline-it
if ~ismember('label', T.Properties.VariableNames)
    T.label = makeSurrogateLabelsLikePipeline(T);
end

% --- Përgatit X, y me të njëjtat veçori që pret modeli
vars = M.vars(:)';

% Map nëse ndonjë emër te CSV është "sanitizuar"
missing = setdiff(vars, T.Properties.VariableNames);
if ~isempty(missing)
    allCSV = T.Properties.VariableNames;
    map = containers.Map;
    for i=1:numel(allCSV)
        map(regexprep(allCSV{i}, '\W', '_')) = allCSV{i};
    end
    still = {};
    for i=1:numel(missing)
        key = regexprep(missing{i}, '\W', '_');
        if isKey(map, key)
            vars(strcmp(vars, missing{i})) = {map(key)};
        else
            still{end+1} = missing{i}; %#ok<AGROW>
        end
    end
    if ~isempty(still)
        error(['Kolonat e mëposhtme nuk u gjetën as me "preserve" e as me mapping:' newline '  - ' strjoin(still, newline+'  - ')]);
    end
end

X = T{:, vars};
y = double(T.label(:) > 0);

% --- Standardizo si në scaler-in e ruajtur
mu = M.scaler.mu; sg = M.scaler.sigma; sg(sg==0)=1;
X = fillmissing(X,'constant',0);
Xs = (X - mu) ./ sg;

% --- Split 80/20 me rng 42 si në pipeline
rng(42);
n  = size(Xs,1); idx = randperm(n);
nTr = round(0.8*n);
te = idx(nTr+1:end);
Xte = Xs(te,:); yte = y(te);

% --- SVM
[~,scoreSVM] = predict(M.svm, Xte);
pSVM = scoreSVM(:,2);
yhatSVM = pSVM >= 0.5;

% --- ANN
pANN = M.ann(Xte')';
yhatANN = pANN >= 0.5;

% --- Metrikat
metrics = @(ytrue,score,ypred) local_metrics(ytrue, score, ypred);
msvm = metrics(yte, pSVM, yhatSVM);
mann = metrics(yte, pANN, yhatANN);

% --- Graf krahasues
cats = ["Accuracy","Precision","Recall","F1","AUPRC","AUROC"];
svmVals = [msvm.acc, msvm.prec, msvm.rec, msvm.f1, msvm.auprc, msvm.auroc];
annVals = [mann.acc, mann.prec, mann.rec, mann.f1, mann.auprc, mann.auroc];

figure('Color','w');
bar([svmVals; annVals]'); grid on;
set(gca,'XTickLabel',cats);
legend('SVM','ANN','Location','southoutside','Orientation','horizontal');
title('Model performance comparison (test set, 80/20)');
ylabel('Score'); ylim([0 1]);

% ====== Helpers ======
function m = local_metrics(yTrue, yScore, yPred)
    yTrue = yTrue(:); yPred = yPred(:); yScore = yScore(:);
    acc = mean(yPred==yTrue);
    tp = sum(yPred==1 & yTrue==1);
    fp = sum(yPred==1 & yTrue==0);
    fn = sum(yPred==0 & yTrue==1);
    prec = tp / max(tp+fp,1);
    rec  = tp / max(tp+fn,1);
    f1   = 2*prec*rec / max(prec+rec,eps);
    try
        [~,~,~,auroc] = perfcurve(yTrue,yScore,1);
        [~,~,~,auprc] = perfcurve(yTrue,yScore,1,'xCrit','reca','yCrit','prec');
    catch
        auroc = NaN; auprc = NaN;
    end
    m = struct('acc',acc,'prec',prec,'rec',rec,'f1',f1,'auroc',auroc,'auprc',auprc);
end

function lbl = makeSurrogateLabelsLikePipeline(A)
    % z-score për events_total
    z1 = zsafe(A, "events_total");

    % severity e lartë: nëse ka sev_4_cnt / sev_7_cnt përdori,
    % përndryshe mblidh të gjitha kolonat sev_>=4
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
    sevRatio = sevHigh ./ max(A.("events_total"), 1);
    z2 = zscore_fallback(sevRatio);

    % burst dhe gantry tilt
    z3    = zsafe(A, "burst_len_max_1m");
    zTilt = zscore_fallback(abs(getOrZero(A,"mean_gantry_tilt")));

    lbl = double((z1>2) | (z2>2) | (z3>2) | (zTilt>2));
end

function z = zsafe(A, name)
    v = getOrZero(A, name);
    z = zscore_fallback(v);
end

function v = getOrZero(A, name)
    if ismember(name, A.Properties.VariableNames), v = A.(name); else, v = zeros(height(A),1); end
end

function z = zscore_fallback(x)
    x = double(x);
    m = mean(x,'omitnan'); s = std(x,[],'omitnan');
    if s==0 || isnan(s), s = 1; end
    z = (x - m) ./ s; z(isnan(z)) = 0;
end

end
