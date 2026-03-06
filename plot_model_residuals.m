function plot_model_residuals()
% === Plot residuals (errors) for SVM and ANN models ===
% Krahason parashikimet e dy modeleve dhe vizualizon gabimet (FP/FN)

csvFile = 'FeatureTable_10min_2024_2025.csv';
matFile = 'pdm_models.mat';
fprintf('Lexoj: %s dhe %s\n', csvFile, matFile);

% ---- 1) Leximi i CSV ----
opts = detectImportOptions(csvFile);
opts.VariableNamingRule = 'preserve';
T = readtable(csvFile, opts);

% ---- 2) Timestamp si datetime ----
if ~isdatetime(T.timestamp)
    try
        T.timestamp = datetime(T.timestamp,'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
    catch
        T.timestamp = datetime(T.timestamp);
    end
end

% ---- 3) Etiketa surrogate nëse mungon 'label' ----
if ~ismember('label', T.Properties.VariableNames)
    T.label = makeSurrogateLabelsLikePipeline(T);
end

% ---- 4) Ngarkimi i modeleve ----
S = load(matFile);
models = S.modelsFinal;
vars = models.vars;

% Filtrimi i kolonave numerike
X = zeros(height(T), numel(vars));
for i=1:numel(vars)
    v = vars{i};
    if ismember(v, T.Properties.VariableNames)
        X(:,i) = numify(T.(v));
    end
end
yTrue = double(T.label(:) > 0);

% Standardizim
mu = models.scaler.mu;
sg = models.scaler.sigma;
Xs = (X - mu) ./ sg;

% ---- 5) Parashikimet ----
[~,scoreSVM] = predict(models.svm, Xs);
yPredSVM = scoreSVM(:,2);

scoreANN = models.ann(Xs')';
yPredANN = scoreANN;

% ---- 6) Rezidualet ----
resSVM = yTrue - yPredSVM;
resANN = yTrue - yPredANN;

% ---- 7) Vizualizimi ----
figure('Color','w','Position',[100 100 950 420]);

subplot(1,2,1)
scatter(T.timestamp, resSVM, 8, 'filled');
title('SVM Residuals');
ylabel('Gabimi (y_{true} - y_{pred})');
xlabel('Koha');
grid on; ylim([-1.5 1.5]);

subplot(1,2,2)
scatter(T.timestamp, resANN, 8, 'filled');
title('ANN Residuals');
ylabel('Gabimi (y_{true} - y_{pred})');
xlabel('Koha');
grid on; ylim([-1.5 1.5]);

sgtitle('Residuals (SVM vs ANN) në kohë');

exportgraphics(gcf, 'Residuals_SVM_ANN.png', 'Resolution', 200);
fprintf('✓ Figura u ruajt si Residuals_SVM_ANN.png\n');
end

% ==================== FUNKSIONE NDIHMËSE ====================
function lbl = makeSurrogateLabelsLikePipeline(T)
    ev   = numify(getField(T,'events_total'));
    sev4 = numify(getField(T,'sev_4_cnt'));
    sev7 = numify(getField(T,'sev_7_cnt'));
    burst= numify(getField(T,'burst_len_max_1m'));
    tilt = numify(getField(T,'mean_gantry_tilt'));

    sev_ratio = (sev4 + sev7) ./ max(ev, 1);
    z1 = zscoreSafe(ev);
    z2 = zscoreSafe(sev_ratio);
    z3 = zscoreSafe(burst);
    zTilt = zscoreSafe(abs(tilt));

    lbl = double((z1>2) | (z2>2) | (z3>2) | (zTilt>2));
end

function v = getField(T, name)
    if ismember(name, T.Properties.VariableNames)
        v = T.(name);
    else
        v = zeros(height(T),1);
    end
end

function x = numify(v)
    if iscell(v)
        x = str2double(v);
    elseif isstring(v)
        x = str2double(v);
    else
        x = double(v);
    end
    x(~isfinite(x)) = 0;
end

function z = zscoreSafe(x)
    x = double(x);
    m = mean(x,'omitnan');
    s = std(x,[],'omitnan');
    if s==0 || isnan(s)
        s=1;
    end
    z = (x - m) ./ s;
    z(~isfinite(z)) = 0;
end
