function plot_temporal_anomaly_heatmap()
% === Temporal Heatmap for Out-of-Tolerance (OOT) events ===
% Lexon FeatureTable_10min_2024_2025.csv dhe tregon shpërndarjen e anomalive
% sipas orës dhe muajit, me ngjyra të normalizuara brenda çdo muaji.

csvFile = 'FeatureTable_10min_2024_2025.csv';
normalizeFlag = true; % nëse do version të normalizuar (përqindje)
fprintf('Lexoj: %s\n', csvFile);

% ---- 1. Leximi i CSV ----
opts = detectImportOptions(csvFile);
opts.VariableNamingRule = 'preserve';
T = readtable(csvFile, opts);

% ---- 2. Siguro që timestamp është datetime ----
if ~isdatetime(T.timestamp)
    try
        T.timestamp = datetime(T.timestamp,'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
    catch
        T.timestamp = datetime(T.timestamp);
    end
end

% ---- 3. Krijo etiketa surrogate nëse mungon 'label' ----
if ~ismember('label', T.Properties.VariableNames)
    T.label = makeSurrogateLabels(T);
end

% ---- 4. Nxjerr muajin dhe orën ----
T.hour = hour(T.timestamp);
T.ym = dateshift(T.timestamp,'start','month');

% Vetëm anomalitë
A = T(T.label > 0, {'ym','hour'});
A.hour = double(A.hour);

% ---- 5. Numëro ngjarjet për çdo (muaj, orë) ----
months = unique(A.ym,'stable');
hours = 0:23;
Z = zeros(numel(months), numel(hours));

for i = 1:height(A)
    m = find(months == A.ym(i));
    h = A.hour(i)+1;
    Z(m,h) = Z(m,h) + 1;
end

% ---- 6. Normalizo (përqindje brenda muajit) ----
if normalizeFlag
    rowSum = sum(Z,2);
    rowSum(rowSum==0)=1;
    Z = 100*(Z./rowSum);
    labelC = 'Përqindja brenda muajit (%)';
else
    labelC = 'Numri i ngjarjeve OOT';
end

% ---- 7. Krijo heatmap ----
monthLabels = cellstr(datestr(months,'yyyy-mmm'));
figure('Color','w','Position',[100 100 950 480]);
h = heatmap(hours, monthLabels, Z);
h.Colormap = turbo; % ngjyra më kontrast
h.CellLabelColor = 'none';
h.ColorbarVisible = 'on';
h.XLabel = 'Ora e ditës';
h.YLabel = 'Muaji (YYYY-MMM)';
h.Title = 'Shpërndarja kohore e ngjarjeve OOT (normalizuar)';
h.MissingDataColor = [0.95 0.95 0.95];
h.MissingDataLabel = 'Pa të dhëna';

% Ruaje automatikisht figurën
exportgraphics(gcf, 'OOT_TemporalHeatmap.png', 'Resolution', 200);
fprintf('✓ Figura u ruajt si OOT_TemporalHeatmap.png\n');
end


% === Funksione ndihmëse për etiketa ===
function lbl = makeSurrogateLabels(T)
    ev   = numify(getField(T,'events_total'));
    sev4 = numify(getField(T,'sev_4_cnt'));
    sev7 = numify(getField(T,'sev_7_cnt'));
    burst= numify(getField(T,'burst_len_max_1m'));
    tilt = numify(getField(T,'mean_gantry_tilt'));

    sev_ratio = (sev4 + sev7) ./ max(ev,1);
    z1 = zscoreSafe(ev);
    z2 = zscoreSafe(sev_ratio);
    z3 = zscoreSafe(burst);
    zTilt = zscoreSafe(abs(tilt));

    lbl = double((z1>2) | (z2>2) | (z3>2) | (zTilt>2));
end

function v = getField(T,name)
    if ismember(name, T.Properties.VariableNames)
        v = T.(name);
    else
        v = zeros(height(T),1);
    end
end

function x = numify(v)
    if iscell(v), x = str2double(v);
    elseif isstring(v), x = str2double(v);
    else, x = double(v);
    end
    x(~isfinite(x)) = 0;
end

function z = zscoreSafe(x)
    x = double(x);
    m = mean(x,'omitnan');
    s = std(x,[],'omitnan');
    if s==0 || isnan(s), s=1; end
    z = (x - m) ./ s;
    z(~isfinite(z)) = 0;
end
