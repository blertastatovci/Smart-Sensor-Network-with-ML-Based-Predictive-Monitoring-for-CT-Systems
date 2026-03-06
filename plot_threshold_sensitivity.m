function plot_threshold_sensitivity()
    % plot_threshold_sensitivity.m
    % Vizualizon ndjeshmërinë e performancës ndaj pragut (F1-score vs threshold)

    % 1) lexo datasetin
    T = readtable('FeatureTable_10min_2024_2025.csv', 'VariableNamingRule','preserve');

    % 2) krijo etiketa surrogate si në pipeline
    ytrue = makeLabelsLocal(T);

    % 3) ngarko modelet e ruajtura
    S = load('pdm_models.mat');

    % nëse ekziston struktura modelsFinal, përdore
    if isfield(S,'modelsFinal')
        models = S.modelsFinal;
    else
        error('Nuk u gjet struktura modelsFinal brenda pdm_models.mat.');
    end

    % 4) përgatit të dhënat numerike sipas varësive që janë ruajtur
    vars = models.vars;
    X = zeros(height(T), numel(vars));
    for i = 1:numel(vars)
        if ismember(vars{i}, T.Properties.VariableNames)
            X(:,i) = numify(T.(vars{i}));
        end
    end

    % standardizo me scaler-in e ruajtur
    mu = models.scaler.mu;
    sg = models.scaler.sigma;
    Xs = (X - mu) ./ sg;

    % 5) gjenero probabilitetet e parashikimit për SVM dhe ANN
    [~, scoreSVM] = predict(models.svm, Xs);
    yscore_svm = scoreSVM(:,2);

    scoreANN = models.ann(Xs')';
    yscore_ann = scoreANN;

    % 6) llogarit F1 për çdo prag
    th = linspace(0,1,101);
    f1_svm = zeros(size(th));
    f1_ann = zeros(size(th));

    for i = 1:numel(th)
        ypred_svm = yscore_svm >= th(i);
        ypred_ann = yscore_ann >= th(i);

        f1_svm(i) = f1score_local(ytrue, ypred_svm);
        f1_ann(i) = f1score_local(ytrue, ypred_ann);
    end

    % 7) vizualizo
    figure('Color','w','Position',[100 100 800 400]);
    plot(th, f1_svm, 'b-', 'LineWidth', 1.5); hold on;
    plot(th, f1_ann, 'r-', 'LineWidth', 1.5);
    grid on;
    xlabel('Threshold');
    ylabel('F1-score');
    title('F1-score në varësi të pragut të vendimmarrjes');
    legend('SVM','ANN','Location','best');
end

%% ===== helper functions =====
function y = makeLabelsLocal(T)
    % krijon etiketa surrogate bazuar në kolonat më të rëndësishme
    names = {'events_total','sev_4_cnt','sev_5_cnt','sev_6_cnt','burst_len_max_1m'};
    vals = zeros(height(T), numel(names));
    for i = 1:numel(names)
        if ismember(names{i}, T.Properties.VariableNames)
            vals(:,i) = numify(T.(names{i}));
        end
    end
    z = zscore(vals, [], 1);
    score = mean(z, 2, 'omitnan');
    y = double(score > 2);
end

function f1 = f1score_local(ytrue, ypred)
    tp = sum(ytrue==1 & ypred==1);
    fp = sum(ytrue==0 & ypred==1);
    fn = sum(ytrue==1 & ypred==0);
    prec = tp / max(tp+fp,1);
    rec  = tp / max(tp+fn,1);
    f1 = 2*prec*rec / max(prec+rec,1);
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
