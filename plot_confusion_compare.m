function plot_confusion_compare()
    % ------------------------------------------------------------
    % Krahasim vizual i performancës për SVM dhe ANN
    % Përfshin: Confusion Matrix për secilin + hartë delta
    % ------------------------------------------------------------

    % 1) Lexo tabelën bazë (vetëm për etiketa)
    T = readtable('FeatureTable_10min_2024_2025.csv', ...
        'VariableNamingRule','preserve');

    % 2) Krijo etiketat si në pipeline (z-score > 2)
    ytrue = double( ...
        zscore(numify(T.events_total)) > 2 | ...
        zscore(numify(T.sev_4_cnt)) > 2 | ...
        zscore(numify(T.burst_len_max_1m)) > 2 );

    % 3) Ngarko modelet e ruajtura
    S = load('pdm_models.mat');
    fn = fieldnames(S);
    disp('Fushat në pdm_models.mat:');
    disp(fn);

    % 4) Lexo parashikimet (zgjidh variantin që ekziston)
    if isfield(S,'yPredSVM')
        y_svm = S.yPredSVM;
    elseif isfield(S,'svm_pred')
        y_svm = S.svm_pred;
    elseif isfield(S,'modelsFinal')
        [~,scoreSVM] = predict(S.modelsFinal.svm, ...
            normalize_features(T, S.modelsFinal));
        y_svm = scoreSVM(:,2) >= 0.5;
    else
        error('Nuk u gjetën parashikimet për SVM.');
    end

    if isfield(S,'yPredANN')
        y_ann = S.yPredANN;
    elseif isfield(S,'ann_pred')
        y_ann = S.ann_pred;
    elseif isfield(S,'modelsFinal')
        scoreANN = S.modelsFinal.ann( ...
            normalize_features(T, S.modelsFinal)');
        y_ann = scoreANN' >= 0.5;
    else
        error('Nuk u gjetën parashikimet për ANN.');
    end

    y_svm = round(y_svm(:));
    y_ann = round(y_ann(:));
    ytrue = ytrue(:);

    % 5) Ndërto matricat e konfuzionit (numerikisht)
    C_svm = confusionmat(ytrue, y_svm);
    C_ann = confusionmat(ytrue, y_ann);
    C_svm_pct = 100 * C_svm ./ sum(C_svm,2);
    C_ann_pct = 100 * C_ann ./ sum(C_ann,2);
    deltaC = C_ann_pct - C_svm_pct;

    % 6) Vizualizimet -------------------------------------------

    figure('Color','w','Position',[100 100 1150 380]);

    % --- SVM Confusion ---
    subplot(1,3,1)
    imagesc(C_svm_pct)
    title('SVM – Confusion Matrix (%)')
    xlabel('Parashikuar'); ylabel('Reale');
    xticks(1:2); yticks(1:2);
    xticklabels({'Klasa 0','Klasa 1'});
    yticklabels({'Klasa 0','Klasa 1'});
    colorbar
    text_labels(C_svm_pct);

    % --- ANN Confusion ---
    subplot(1,3,2)
    imagesc(C_ann_pct)
    title('ANN – Confusion Matrix (%)')
    xlabel('Parashikuar'); ylabel('Reale');
    xticks(1:2); yticks(1:2);
    xticklabels({'Klasa 0','Klasa 1'});
    yticklabels({'Klasa 0','Klasa 1'});
    colorbar
    text_labels(C_ann_pct);

    % --- Diferenca ANN - SVM ---
    subplot(1,3,3)
    imagesc(deltaC)
    title('\Delta (ANN - SVM)')
    xlabel('Parashikuar'); ylabel('Reale');
    xticks(1:2); yticks(1:2);
    xticklabels({'Klasa 0','Klasa 1'});
    yticklabels({'Klasa 0','Klasa 1'});
    colormap(subplot(1,3,3), flipud(parula)); % paletë standarde MATLAB, pa toolbox shtesë
    colorbar
    text_labels(deltaC);
    sgtitle('Krahasimi i performancës SVM vs ANN');

    exportgraphics(gcf,'Confusion_Compare_Delta.png','Resolution',200);
    fprintf('✓ Figura u ruajt si Confusion_Compare_Delta.png\n');
end

% ======= funksione ndihmëse =======
function Xs = normalize_features(T, model)
    vars = model.vars;
    mu = model.scaler.mu;
    sg = model.scaler.sigma;
    X = zeros(height(T), numel(vars));
    for i = 1:numel(vars)
        if ismember(vars{i}, T.Properties.VariableNames)
            X(:,i) = numify(T.(vars{i}));
        end
    end
    Xs = (X - mu) ./ sg;
end

function text_labels(M)
    % vendos vlerat përqindore në katrorë
    for i = 1:size(M,1)
        for j = 1:size(M,2)
            text(j,i,sprintf('%.1f%%',M(i,j)), ...
                'HorizontalAlignment','center', ...
                'Color','w','FontWeight','bold');
        end
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
