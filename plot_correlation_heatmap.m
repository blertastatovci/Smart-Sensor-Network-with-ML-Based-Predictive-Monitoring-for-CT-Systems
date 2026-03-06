function plot_correlation_heatmap(csvPath, varList)
% Correlation heatmap për veçoritë kryesore nga FeatureTable
% Përdorim: plot_correlation_heatmap('FeatureTable_10min_2024_2025.csv');

if nargin==0
    csvPath = fullfile(pwd,'FeatureTable_10min_2024_2025.csv');
end
T = readtable(csvPath);

% nëse s’jepet varList, zgjedh një bërthamë të kuptueshme
if nargin<2 || isempty(varList)
    varList = [
        "events_total","burst_len_max_1m","entropy_codes","trend_events_24h", ...
        "time_since_last_event","is_abort_cnt","is_start_scan_cnt","is_stop_scan_cnt", ...
        "mean_dfov","mean_slice_thk","mean_gantry_tilt"
    ];
    varList = varList(ismember(varList, T.Properties.VariableNames));
end

X = T{:, varList};
R = corr(X, 'Rows','pairwise');

figure('Color','w'); 
h = heatmap(varList, varList, R, 'Colormap', parula, 'ColorLimits',[-1 1]);
title('Correlation matrix of extracted features (10-min windows)');
end
