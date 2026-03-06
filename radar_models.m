function radar_models_filled_real()
% Radar chart i mbushur me kompozit = mes(AUPRC, AUROC)
% ANN/SVM nga rezultatet e tua; të tjerat baseline të rregullueshme.

% ---- Rezultatet reale (nga pipeline) ----
AUPRC_SVM = 0.981; AUROC_SVM = 0.996;
AUPRC_ANN = 0.976; AUROC_ANN = 0.993;
score_SVM = mean([AUPRC_SVM, AUROC_SVM]);   % 0.9885
score_ANN = mean([AUPRC_ANN, AUROC_ANN]);   % 0.9845

labels = { ...
    'GENETIC ALGORITHM','PRT','FUZZY','LSTM','NAÏVE BAYES', ...
    'REGRESSION ANALYSIS','DECISION TREE','KNN','RANDOM FOREST', ...
    'SVM','ANN'};

% ---- Baseline (ndrysho lirisht) ----
values = [ ...
    0.80, ... % Genetic Algorithm
    0.76, ... % PRT
    0.78, ... % Fuzzy
    0.93, ... % LSTM (baseline)
    0.72, ... % Naïve Bayes
    0.75, ... % Regression
    0.80, ... % Decision Tree
    0.85, ... % KNN
    0.90, ... % Random Forest
    score_SVM, ...
    score_ANN ...
];

% ---- Koordinatat polare (mbyll poligonin) ----
N     = numel(labels);
theta = linspace(0, 2*pi, N+1);
rho   = [values, values(1)];

% ---- Figura + polaraxes ----
figure('Color','w','Position',[120 120 720 600]);
ax = polaraxes; hold(ax,'on');
ax.ThetaZeroLocation = 'top';
ax.ThetaDir          = 'clockwise';
ax.ThetaTick         = rad2deg(theta(1:N));
ax.ThetaTickLabel    = labels;
ax.RLim              = [0 1];
ax.RTick             = 0:0.2:1;
title('CT Predictive Maintenance – Composite Score (avg AUPRC & AUROC)');

% Vija kryesore
p = polarplot(ax, theta, rho, 'LineWidth', 2);

% Mbushja me overlay kartesian
fill_on_polar(ax, theta, rho, p.Color, 0.25);

% Kufiri i theksuar
polarplot(ax, theta, rho, 'k', 'LineWidth', 1.5);

% Ruaje (opsionale)
exportgraphics(gcf, 'CT_PM_Radar_Composite_MATLAB.png', 'Resolution', 300);
exportgraphics(gcf, 'CT_PM_Radar_Composite_MATLAB.pdf');
end

function fill_on_polar(axPolar, theta, rho, colorRGB, alphaVal)
    if nargin < 5, alphaVal = 0.25; end
    if nargin < 4 || isempty(colorRGB), colorRGB = [0 0.4470 0.7410]; end
    [xFill, yFill] = pol2cart(theta, rho);
    axCart = axes('Position', axPolar.Position, 'Color','none', 'Visible','off','HitTest','off');
    axis(axCart,'equal'); axis(axCart,[-1 1 -1 1]);
    fill(axCart, xFill, yFill, colorRGB, 'FaceAlpha',alphaVal, 'EdgeColor','none');
    uistack(axPolar,'top');
end

