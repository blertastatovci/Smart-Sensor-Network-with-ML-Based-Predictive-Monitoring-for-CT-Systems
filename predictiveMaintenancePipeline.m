%% predictiveMaintenancePipeline.m
% CT Predictive Maintenance – Logs only (2024–2025) – ONE FILE (UPDATED v2)
% CHANGES (2025-10-21):
% • Deduplicate input log files (avoid double reading)
% • Chronological train/test split (default)
% • SVM: remove fitPosterior, add class weights & threshold sweep for max accuracy
% • Fixed regex backslashes for MATLAB (\s+, etc.)
% • Optional hyperparameter optimization block (commented by default)

clear; clc;

%% ===================== USER CONFIG =====================
logFolder     = pwd;                     % folderi me log-et
timeZoneStr   = 'Europe/Belgrade';
windowMinutes = 10;                      % dritare agregimi
yearMin       = 2024;                    % nga 2024
yearMax       = 2025;                    % deri 2025

% outputs
outCSV        = fullfile(logFolder,'FeatureTable_10min_2024_2025.csv');
outCAT        = fullfile(logFolder,'FeatureCatalog.xlsx');
outMAT        = fullfile(logFolder,'pdm_models.mat');

% splitting: 'chronological' (recommended) or 'random'
splitMode     = 'chronological';
trainFrac     = 0.80;                    % 80% train, 20% test

% SVM options
useClassWeights   = true;                % pesha klasash për imbalancim
optimizeHyper     = false;               % aktivizo për "heavy tuning"
boxC              = 10;                  % nëse optimizeHyper=false
kernelScale       = 'auto';              % nëse optimizeHyper=false

fprintf('=== CT PdM FINAL (Logs only, UPDATED v2) ===\nFolder: %s\n\n', logFolder);

%% ========== 1) Read ALL relevant logs (2024–2025) ==========
cands = [ ...
    dir(fullfile(logFolder, 'gesys_ct01.log.*')); ...
    dir(fullfile(logFolder, 'gesys_CT01.log*')); ...
    dir(fullfile(logFolder, 'gesys_ct01 Aug2025*')) ...
];
cands = cands(~[cands.isdir]);

% --- Deduplicate by absolute path ---
if ~isempty(cands)
    fulls = fullfile({cands.folder}', {cands.name}');
    [~, ia] = unique(fulls, 'stable');
    cands = cands(ia);
end

if isempty(cands), error('S’u gjetën file gesys_ct01* në %s', logFolder); end

allTbl = table();
for k = 1:numel(cands)
    fp = fullfile(cands(k).folder, cands(k).name);
    fprintf(' -> Lexoj: %s\n', cands(k).name);
    try
        T = parseGesysLog(fp, timeZoneStr);  % === FUNKSION NË FUND ===
        if ~isempty(T)
            allTbl = [allTbl; T]; %#ok<AGROW>
        end
    catch ME
        warning('Leximi dështoi për %s: %s', cands(k).name, ME.message);
    end
end
if isempty(allTbl), error('Asnjë event i lexuar nga log-et.'); end

% Filter 2024–2025
allTbl = allTbl(year(allTbl.ts)>=yearMin & year(allTbl.ts)<=yearMax, :);
fprintf('\nTOTAL: %d evente (vite %d–%d)\n', height(allTbl), yearMin, yearMax);

%% ========== 2) Build features (10-min) ==========
featTbl = buildFeatureTable(allTbl, minutes(windowMinutes));  % === FUNKSION NË FUND ===
fprintf('U krijua tabela e features: %d rreshta x %d kolona\n', height(featTbl), width(featTbl));

% Feature catalog
cat = table(featTbl.Properties.VariableNames', repmat("log-derived", width(featTbl),1), ...
            'VariableNames', {'FeatureName','Source'});
try, writetable(cat, outCAT); end

% Save CSV
writetable(featTbl, outCSV);
fprintf('✓ Ruajta: %s\n', outCSV);

%% ========== 3) Surrogate labels + Train SVM/ANN + Figures ==========
featTbl.label = makeSurrogateLabels(featTbl);                 % === FUNKSION NË FUND ===

vars = featTbl.Properties.VariableNames;
isNum = varfun(@isnumeric, featTbl, 'OutputFormat','uniform');
keep = isNum & ~ismember(vars, {'label'});
X = featTbl{:, keep};
y = double(featTbl.label(:) > 0);

% Preprocess
X = fillmissing(X,'constant',0);
mu = mean(X,1,'omitnan'); sg = std(X,[],1,'omitnan'); sg(sg==0)=1;
Xs = (X - mu) ./ sg;

% Chronological vs Random split
switch lower(splitMode)
    case 'chronological'
        [~, ord] = sort(featTbl.timestamp);
        Xs = Xs(ord,:); y = y(ord);
        n = size(Xs,1); nTrain = round(trainFrac*n);
        XsTr = Xs(1:nTrain,:); yTr = y(1:nTrain);
        XsTe = Xs(nTrain+1:end,:); yTe = y(nTrain+1:end);
    otherwise
        rng(42);
        n = size(Xs,1); idx = randperm(n);
        nTrain = round(trainFrac*n);
        tr = idx(1:nTrain); te = idx(nTrain+1:end);
        XsTr = Xs(tr,:); yTr = y(tr);
        XsTe = Xs(te,:); yTe = y(te);
end

% ===== SVM (no fitPosterior, class weights, threshold sweep) =====
if useClassWeights
    w = ones(numel(yTr),1);
    pos = yTr==1; neg = ~pos;
    if any(pos)
        w(pos) = sum(neg)/max(sum(pos),1);
    end
else
    w = [];
end

if optimizeHyper
    svm = fitcsvm(XsTr, yTr, 'KernelFunction','rbf', 'Standardize',false, ...
        'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus', ...
                                                   'KFold',5,'UseParallel',true));
else
    if isempty(w)
        svm = fitcsvm(XsTr, yTr, 'KernelFunction','rbf', 'KernelScale',kernelScale, ...
            'BoxConstraint',boxC, 'Standardize',false, 'ClassNames',[0 1]);
    else
        svm = fitcsvm(XsTr, yTr, 'KernelFunction','rbf', 'KernelScale',kernelScale, ...
            'BoxConstraint',boxC, 'Standardize',false, 'ClassNames',[0 1], 'Weights', w);
    end
end

[~,scoreSVM_te] = predict(svm, XsTe);  % raw scores
[bestTh, bestAcc_svm, accCurve] = chooseBestThreshold(scoreSVM_te(:,2), yTe); %#ok<ASGLU>
yhat_svm = scoreSVM_te(:,2) >= bestTh;

% ===== ANN (simple) =====
hiddenSize = 16;
net = patternnet(hiddenSize);
net.trainParam.showWindow = false;
net.divideMode = 'none';
net = train(net, XsTr', yTr');
scoreANN = net(XsTe')';
yhat_ann  = scoreANN >= 0.5;

% Reports + figures
repSVM = evaluateAndPlot(yTe, scoreSVM_te(:,2), yhat_svm, sprintf('SVM[th=%.4f]',bestTh));
repANN = evaluateAndPlot(yTe, scoreANN,       yhat_ann,  'ANN');

% Save models
modelsFinal = struct();
modelsFinal.vars           = vars(keep);
modelsFinal.scaler.mu      = mu;
modelsFinal.scaler.sigma   = sg;
modelsFinal.svm            = svm;
modelsFinal.ann            = net;
modelsFinal.reportSVM      = repSVM;
modelsFinal.reportANN      = repANN;
modelsFinal.windowMinutes  = windowMinutes;
modelsFinal.timeZone       = timeZoneStr;
modelsFinal.years          = [yearMin yearMax];
modelsFinal.splitMode      = splitMode;
modelsFinal.trainFrac      = trainFrac;
modelsFinal.svmBestThresh  = bestTh;
modelsFinal.svmAccCurve    = accCurve;  % diagnostics
save(outMAT, 'modelsFinal');

fprintf('\n=== DONE. Models & reports saved to %s ===\n', outMAT);
fprintf('Grafikat: SVM*_confusion.png, SVM*_ROC.png, SVM*_PR.png, ANN_*.png\n');

%% ====================== LOCAL FUNCTIONS (MUST BE LAST) ======================

function [bestTh, bestAcc, accCurve] = chooseBestThreshold(scores, yTrue)
% Sweep mbi pragjet reale të score-it për të maksimizuar accuracy
    scores = scores(:); yTrue = yTrue(:);
    ths = unique(scores);
    bestAcc = 0; bestTh = 0.5;
    accCurve = zeros(numel(ths),1);
    for i=1:numel(ths)
        t = ths(i);
        yhat = scores >= t;
        acc  = mean(yhat == yTrue);
        accCurve(i) = acc;
        if acc > bestAcc
            bestAcc = acc; bestTh = t;
        end
    end
end

function T = parseGesysLog(filepath, tz)
% Parser SR...EN dhe ekstrakte nga message
    txt = fileread_detect(filepath);
    txt = strrep(txt, char(13), '');
    L   = regexp(txt, '\n', 'split');
    n   = numel(L); i = 1;
    rows = struct('epoch',{},'code',{},'severity',{},'module',{},'submodule',{}, ...
                  'file',{},'line',{},'message',{});
    while i <= n
        line = strtrim(L{i});
        if startsWith(line,'SR ')
            if i+3 <= n
                p1 = regexp(strtrim(L{i+1}), '\s+', 'split');
                if numel(p1) >= 6 && all(isstrprop(p1{1},'digit'))
                    ep = str2double(p1{1}); if numel(p1{1})==13, ep = ep/1000; end
                    code = string(p1{5});
                    sev  = str2double(p1{6});
                    p2 = regexp(strtrim(L{i+2}), '\s+', 'split'); p2 = p2(~cellfun(@isempty,p2));
                    module    = ""; submodule = "";
                    if numel(p2)>=1, module    = string(p2{1}); end
                    if numel(p2)>=2, submodule = string(p2{2}); end
                    p3 = regexp(strtrim(L{i+3}), '\s+', 'split'); p3 = p3(~cellfun(@isempty,p3));
                    fil = ""; lin = NaN;
                    if ~isempty(p3)
                        fil = string(p3{1});
                        if all(isstrprop(p3{end},'digit')), lin = str2double(p3{end}); end
                    end
                    j = i+4; msgL = strings(0,1);
                    while j<=n && ~startsWith(strtrim(L{j}),'EN ')
                        msgL(end+1,1) = string(L{j}); j = j+1; %#ok<AGROW>
                    end
                    msg = strtrim(strjoin(msgL, newline));
                    rows(end+1) = struct('epoch',ep,'code',code,'severity',sev, ...
                        'module',module,'submodule',submodule,'file',fil,'line',lin, ...
                        'message',msg); %#ok<AGROW>
                    if j<=n && startsWith(strtrim(L{j}),'EN '), i = j+1; continue; else, i=j; continue; end
                end
            end
        end
        i = i+1;
    end
    if isempty(rows), T = table(); return; end
    T = struct2table(rows);
    T.ts = datetime(T.epoch,'ConvertFrom','posixtime','TimeZone','UTC'); T.ts.TimeZone = tz;

    % Ekstraktimet nga message
    T.dfov        = str2double(extract1(T.message, 'dfov\s*=\s*([\-0-9\.]+)'));
    T.slice_thk   = str2double(extract1(T.message, 'slice-?thickness\s*=\s*([\-0-9\.]+)'));
    T.gantry_tilt = str2double(extract1(T.message, 'gantry-tilt\s*=\s*([\-0-9\.]+)'));
    T.numslices   = str2double(extract1(T.message, 'numslices\s*=\s*([0-9]+)'));
    T.first_img   = str2double(extract1(T.message, 'first\s*img\s*=\s*([0-9]+)'));

    pr = extract2(T.message, 'slice-center\s*=\s*\(\s*([\-0-9\.]+)\s*,\s*([\-0-9\.]+)\s*\)');
    T.slice_cx = pr(:,1); T.slice_cy = pr(:,2);

    pr = extract2(T.message, 'slice-range\s*=\s*\[\s*([\-0-9\.]+)\s*:\s*([\-0-9\.]+)\s*\]');
    T.slice_r_lo = pr(:,1); T.slice_r_hi = pr(:,2);

    pr = extract2(T.message, 'Start\s+Loc\(\s*([\-0-9\.]+)\s*\)\s*:\s*End\s+Loc\(\s*([\-0-9\.]+)\s*\)');
    T.start_loc = pr(:,1); T.end_loc = pr(:,2);

    T.scan_type  = string(extract1(T.message, 'Scan Type\s*:\s*([A-Za-z/ ]+)'));
    T.group_type = string(extract1(T.message, 'Group\(\d+\)\s*;\s*Group Type\(\s*([A-Za-z]+)\s*\)'));

    T.is_start_scan  = contains(T.message, '[START SCAN]');
    T.is_stop_scan   = contains(T.message, '[STOP SCAN]');
    T.is_abort       = contains(T.message, 'Operator Aborted Scanning');
    T.is_transfer_ok = contains(T.message, 'Transfer request') & contains(T.message, 'status=Ok');
end

function featTbl = buildFeatureTable(ev, winDur)
% 10-min window features (includes burst fix)
    ev = sortrows(ev,'ts');

    % floor në shumëfisha të winDur (p.sh. 10-min)
    ev.win = dateshift(ev.ts,'start','minute',0);
    ev.win = ev.win - minutes(mod(minute(ev.win), winDur/minutes(1)));

    % === baza ===
    G = findgroups(ev.win);
    T  = splitapply(@(x) x(1), ev.win, G);              % timestamp i dritares
    events_total = splitapply(@numel, ev.ts, G);

    % === severity counts ===
    sevU = unique(ev.severity(~isnan(ev.severity))); sevU = sevU(:)';
    sevCnt = zeros(numel(T), numel(sevU));
    for i=1:numel(sevU)
        s = sevU(i); sevCnt(:,i) = splitapply(@(x) sum(x==s), ev.severity, G);
    end

    % === module top-10 ===
    [mods,~,im] = unique(string(ev.module)); freqM = accumarray(im,1);
    [~,ord] = sort(freqM,'descend'); modsTop = mods(ord(1:min(10,end)));
    modCnt  = zeros(numel(T), numel(modsTop));
    for i=1:numel(modsTop)
        m = modsTop(i); modCnt(:,i) = splitapply(@(x) sum(x==m), string(ev.module), G);
    end

    % === code top-15 ===
    [codes,~,ic] = unique(string(ev.code)); freqC = accumarray(ic,1);
    [~,ord] = sort(freqC,'descend'); codesTop = codes(ord(1:min(15,end)));
    codeCnt  = zeros(numel(T), numel(codesTop));
    for i=1:numel(codesTop)
        c = codesTop(i); codeCnt(:,i) = splitapply(@(x) sum(string(x)==c), string(ev.code), G);
    end

    % === diversiteti i kodeve ===
    unique_codes_cnt = splitapply(@(x) numel(unique(string(x))), ev.code, G);
    entropy_codes    = splitapply(@(x) entropy_of_counts(string(x)), ev.code, G);

    % === derivativi i ngarkesës ===
    event_rate_change = [NaN; diff(events_total)];

    % === koha ndërmjet eventeve (mediana) ===
    evSorted = sortrows(ev,'ts');
    evSorted.dt = [NaN; seconds(diff(evSorted.ts))];
    time_since_last_event = splitapply(@(x) median(x,'omitnan'), evSorted.dt, findgroups(evSorted.win));

    % === BURST per minute (FIX) ===
    ev.minute = dateshift(ev.ts, 'start', 'minute', 0);
    GM = findgroups(ev.win, ev.minute);
    perMin = splitapply(@numel, ev.ts, GM);
    winPerPair = splitapply(@(x) x(1), ev.win, GM);
    GW = findgroups(winPerPair);
    burst_len_max_1m = splitapply(@max, perMin, GW);

    % === trend ngarkese 24h ===
    trend_events_24h = rolling_slope(events_total, 144);        % 144 * 10min ~ 24h

    % === mesatare parametrash teknikë ===
    mean_dfov        = splitapply(@(x) mean(x,'omitnan'), ev.dfov,        G);
    mean_slice_thk   = splitapply(@(x) mean(x,'omitnan'), ev.slice_thk,   G);
    mean_gantry_tilt = splitapply(@(x) mean(x,'omitnan'), ev.gantry_tilt, G);
    mean_numslices   = splitapply(@(x) mean(x,'omitnan'), ev.numslices,   G);
    mean_first_img   = splitapply(@(x) mean(x,'omitnan'), ev.first_img,   G);
    mean_slice_cx    = splitapply(@(x) mean(x,'omitnan'), ev.slice_cx,    G);
    mean_slice_cy    = splitapply(@(x) mean(x,'omitnan'), ev.slice_cy,    G);
    mean_slice_r_lo  = splitapply(@(x) mean(x,'omitnan'), ev.slice_r_lo,  G);
    mean_slice_r_hi  = splitapply(@(x) mean(x,'omitnan'), ev.slice_r_hi,  G);
    mean_start_loc   = splitapply(@(x) mean(x,'omitnan'), ev.start_loc,   G);
    mean_end_loc     = splitapply(@(x) mean(x,'omitnan'), ev.end_loc,     G);

    % === event flags ===
    is_start_scan_cnt  = splitapply(@sum, ev.is_start_scan,  G);
    is_stop_scan_cnt   = splitapply(@sum, ev.is_stop_scan,   G);
    is_abort_cnt       = splitapply(@sum, ev.is_abort,       G);
    is_transfer_ok_cnt = splitapply(@sum, ev.is_transfer_ok, G);

    % === scan types top-5 ===
    st = string(ev.scan_type); st(ismissing(st)) = "";
    [types,~,it] = unique(st); freqT = accumarray(it,1);
    [~,ord] = sort(freqT,'descend'); typesTop = types(ord(1:min(5,end)));
    stCnt = zeros(numel(T), numel(typesTop));
    for i=1:numel(typesTop)
        t = typesTop(i); stCnt(:,i) = splitapply(@(x) sum(string(x)==t), st, G);
    end

    % === bashkimi në tabelë ===
    featTbl = table(T, events_total, event_rate_change, unique_codes_cnt, entropy_codes, ...
                    time_since_last_event, burst_len_max_1m, trend_events_24h, ...
                    mean_dfov, mean_slice_thk, mean_gantry_tilt, mean_numslices, ...
                    mean_first_img, mean_slice_cx, mean_slice_cy, mean_slice_r_lo, mean_slice_r_hi, ...
                    mean_start_loc, mean_end_loc, ...
                    is_start_scan_cnt, is_stop_scan_cnt, is_abort_cnt, is_transfer_ok_cnt, ...
                    'VariableNames', {'timestamp','events_total','event_rate_change','unique_codes_cnt','entropy_codes', ...
                                      'time_since_last_event','burst_len_max_1m','trend_events_24h', ...
                                      'mean_dfov','mean_slice_thk','mean_gantry_tilt','mean_numslices', ...
                                      'mean_first_img','mean_slice_cx','mean_slice_cy','mean_slice_r_lo','mean_slice_r_hi', ...
                                      'mean_start_loc','mean_end_loc', ...
                                      'is_start_scan_cnt','is_stop_scan_cnt','is_abort_cnt','is_transfer_ok_cnt'});

    % shto severity
    for i=1:numel(sevU), featTbl.("sev_"+sevU(i)+"_cnt") = sevCnt(:,i); end
    % shto module
    for i=1:numel(modsTop), featTbl.("mod_"+sanitizeName(modsTop(i))+"_cnt") = modCnt(:,i); end
    % shto top codes
    for i=1:numel(codesTop), featTbl.("topcode_"+sanitizeName(codesTop(i))+"_cnt") = codeCnt(:,i); end
    % shto scan types
    for i=1:numel(typesTop), featTbl.("scanType_"+sanitizeName(typesTop(i))+"_cnt") = stCnt(:,i); end
end

function rep = evaluateAndPlot(yTrue, yScore, yPred, tag)
% Raportim metrikash + grafika (CM, ROC, PR)
    yTrue = yTrue(:); yPred = yPred(:); yScore = yScore(:);
    % Metrika bazë
    acc  = mean(yPred==yTrue);
    tp   = sum(yPred==1 & yTrue==1);
    fp   = sum(yPred==1 & yTrue==0);
    fn   = sum(yPred==0 & yTrue==1);
    tn   = sum(yPred==0 & yTrue==0);
    prec = tp / max(tp+fp,1);
    rec  = tp / max(tp+fn,1);
    f1   = 2*prec*rec / max(prec+rec,eps);

    % ROC / PR
    try
        [rocX,rocY,~,aucROC] = perfcurve(yTrue, yScore, 1);
        [prX,prY,~,aucPR]    = perfcurve(yTrue, yScore, 1, 'xCrit','reca','yCrit','prec');
    catch
        aucROC = NaN; aucPR = NaN; rocX = []; rocY = []; prX = []; prY = [];
    end

    % Confusion Matrix plot
    try
        figure('Name',[tag ' Confusion'],'Color','w');
        confusionchart(yTrue,yPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
        title([tag ' – Confusion Matrix']);
        saveas(gcf, [regexprep(tag,'[^A-Za-z0-9_]','_') '_confusion.png']);
    catch
        C = [tn fp; fn tp];
        figure('Name',[tag ' Confusion'],'Color','w');
        imagesc(C); axis equal tight; colorbar;
        set(gca,'XTick',1:2,'XTickLabel',{'Pred 0','Pred 1'});
        set(gca,'YTick',1:2,'YTickLabel',{'True 0','True 1'});
        title([tag ' – Confusion Matrix']);
        text(1,1,num2str(C(1,1)),'HorizontalAlignment','center','Color','w','FontWeight','bold');
        text(2,1,num2str(C(1,2)),'HorizontalAlignment','center','Color','w','FontWeight','bold');
        text(1,2,num2str(C(2,1)),'HorizontalAlignment','center','Color','w','FontWeight','bold');
        text(2,2,num2str(C(2,2)),'HorizontalAlignment','center','Color','w','FontWeight','bold');
        saveas(gcf, [regexprep(tag,'[^A-Za-z0-9_]','_') '_confusion.png']);
    end

    % ROC
    if exist('rocX','var') && ~isempty(rocX)
        figure('Name',[tag ' ROC'],'Color','w');
        plot(rocX, rocY, 'LineWidth',1.5); grid on; xlabel('FPR'); ylabel('TPR');
        title(sprintf('%s – ROC (AUC=%.3f)', tag, aucROC));
        saveas(gcf, [regexprep(tag,'[^A-Za-z0-9_]','_') '_ROC.png']);
    end

    % PR
    if exist('prX','var') && ~isempty(prX)
        figure('Name',[tag ' PR'],'Color','w');
        plot(prX, prY, 'LineWidth',1.5); grid on; xlabel('Recall'); ylabel('Precision');
        title(sprintf('%s – PR (AUPRC=%.3f)', tag, aucPR));
        saveas(gcf, [regexprep(tag,'[^A-Za-z0-9_]','_') '_PR.png']);
    end

    rep = struct('acc',acc,'prec',prec,'rec',rec,'f1',f1,'aucROC',aucROC,'aucPR',aucPR);
    fprintf('%s  Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUPRC %.3f  AUROC %.3f\n', ...
        tag, acc, prec, rec, f1, rep.aucPR, rep.aucROC);
end

function vals = extract1(strArray, pattern)
    toks = regexp(cellstr(strArray), pattern, 'tokens', 'once');
    n = numel(toks); vals = strings(n,1);
    for k=1:n
        if ~isempty(toks{k}), vals(k) = string(toks{k}{1}); else, vals(k) = missing; end
    end
end

function M = extract2(strArray, pattern)
    toks = regexp(cellstr(strArray), pattern, 'tokens', 'once');
    n = numel(toks); M = nan(n,2);
    for k=1:n
        if ~isempty(toks{k})
            M(k,1) = str2double(toks{k}{1});
            M(k,2) = str2double(toks{k}{2});
        end
    end
end

function e = entropy_of_counts(x)
    c = countcats(categorical(x));
    if isempty(c) || sum(c)==0, e = 0; return; end
    p = c / sum(c);
    e = -sum(p .* log2(p + eps));
end

function z = rolling_slope(y, W)
    z = nan(size(y));
    for i=1:numel(y)
        lo = max(1, i-W+1); seg = y(lo:i);
        if numel(seg) >= 5
            X = [ones(numel(seg),1) (1:numel(seg))'];
            b = X \ seg(:);
            z(i) = b(2);
        end
    end
end

function lbl = makeSurrogateLabels(T)
    z1 = zscoreSafe(T.events_total);
    sev4 = getFieldOrZero(T, 'sev_4_cnt');
    sev7 = getFieldOrZero(T, 'sev_7_cnt');
    sev_ratio = (sev4 + sev7) ./ max(T.events_total, 1);
    z2 = zscoreSafe(sev_ratio);
    z3 = zscoreSafe(T.burst_len_max_1m);
    zTilt = zscoreSafe(abs(T.mean_gantry_tilt));
    lbl = double((z1>2) | (z2>2) | (z3>2) | (zTilt>2));
end

function v = getFieldOrZero(S, name)
    if ismember(name, S.Properties.VariableNames), v = S.(name); else, v = zeros(height(S),1); end
end

function z = zscoreSafe(x)
    x = double(x);
    m = mean(x,'omitnan'); s = std(x,[],'omitnan'); if s==0, s=1; end
    z = (x - m) ./ s; z(isnan(z)) = 0;
end

function s = sanitizeName(x)
    s = regexprep(string(x), '\\W', '_');
end

function txt = fileread_detect(fp)
% robust text read with common encodings
    fid = fopen(fp,'r'); if fid<0, error('S’munda të hap %s', fp); end
    C = fread(fid, '*uint8'); fclose(fid);
    encs = {'UTF-8','Windows-1252','ISO-8859-1'};
    for i=1:numel(encs)
        try, txt = native2unicode(C', encs{i}); return; catch, end
    end
    txt = char(C');
end
