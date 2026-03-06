function plot_ct_monthly_util_vs_risk(csvPath)
if nargin==0
    csvPath = fullfile(pwd,'FeatureTable_10min_2024_2025.csv');
end
T = readtable(csvPath);
T.timestamp = datetime(T.timestamp,'InputFormat','yyyy-MM-dd''T''HH:mm:ss', 'TimeZone','local');

% Grupim mujor
T.YM = dateshift(T.timestamp,'start','month');
G = findgroups(T.YM);
months = splitapply(@(d) d(1), T.YM, G);

% --- 1) UTILIZATION: events_total (shuma mujore)
evt = splitapply(@sum, T.events_total, G);

% --- 2) RISK: severity-high (>=5) + aborts (nëse kolonat ekzistojnë)
sevCols = T.Properties.VariableNames(contains(T.Properties.VariableNames,"sev_") & contains(T.Properties.VariableNames,"_cnt"));
sevNums = regexp(sevCols,'sev_(\d+)_cnt','tokens','once');
sevMask = false(size(sevCols));
for i=1:numel(sevCols)
    if ~isempty(sevNums{i}) && str2double(sevNums{i}{1})>=5
        sevMask(i) = true;
    end
end
sevHighMonthly = zeros(size(months));
if any(sevMask)
    for i=find(sevMask)
        sevHighMonthly = sevHighMonthly + splitapply(@sum, T.(sevCols{i}), G);
    end
end

abortMonthly = zeros(size(months));
if ismember('is_abort_cnt', T.Properties.VariableNames)
    abortMonthly = splitapply(@sum, T.is_abort_cnt, G);
end
risk = sevHighMonthly + abortMonthly;

% --- zbutje opsionale 3-mujore (moving average)
ma = @(x) movmean(x,3,'Endpoints','shrink');
evt_s = ma(evt);
risk_s = ma(risk);

% --- vizatim
figure('Color','w','Position',[100 100 900 480]);

subplot(2,1,1); hold on; grid on;
plot(months, evt, '--s','Color',[0 0 0],'LineWidth',1.2,'MarkerSize',5);
plot(months, evt_s, '-','Color',[0 0 0],'LineWidth',2);
[~,imax]=max(evt); plot(months(imax), evt(imax),'ko','MarkerFaceColor',[0 0 0]);
text(months(imax), evt(imax), " peak", 'VerticalAlignment','bottom','HorizontalAlignment','left');
title('CT monthly utilization (events\_total)'); ylabel('events per month');
xticks([]); % etiketat e muajit vetëm poshtë

subplot(2,1,2); hold on; grid on;
plot(months, risk, ':^','Color',[0 0 0],'LineWidth',1.2,'MarkerSize',5);
plot(months, risk_s, '-','Color',[0 0 0],'LineWidth',2);
title('Monthly incident risk (severity≥5 + aborts)');
ylabel('count per month'); xlabel('Month'); xtickformat('yyyy-MMM');

legend({'raw','3-mo MA'},'Location','northwest'); box on;
sgtitle('CT logs (2024–2025): utilization vs. risk','FontWeight','bold');
end
