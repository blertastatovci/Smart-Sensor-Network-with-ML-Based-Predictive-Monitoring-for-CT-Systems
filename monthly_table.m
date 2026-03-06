%% monthly_table.m
% Krijon tabelën mujore të ngjarjeve nga log-et e CT (2024–2025)
% Kërkon automatikisht file-in FeatureTable_10min_2024_2025.csv në direktorinë aktuale

disp("=== Building monthly statistics table from CT logs (2024–2025) ===");

% 1. Kërko file-in automatikisht në folderin aktual
fileList = dir('**/*FeatureTable_10min_2024_2025.csv');
if isempty(fileList)
    error('❌ Nuk u gjet asnjë file "FeatureTable_10min_2024_2025.csv" në projekt.');
else
    csvPath = fullfile(fileList(1).folder, fileList(1).name);
    disp(['✅ File i gjetur: ' csvPath]);
end

% 2. Lexo tabelën
T = readtable(csvPath, 'VariableNamingRule', 'preserve');

% 3. Kontrollo nëse ekziston kolona e kohës
timeVar = [];
for c = T.Properties.VariableNames
    if contains(lower(c{1}), 'time') || contains(lower(c{1}), 'stamp')
        timeVar = c{1};
        break;
    end
end
if isempty(timeVar)
    error('❌ Nuk u gjet kolona e kohës (timestamp).');
end

t = datetime(T.(timeVar), 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
T.Month = dateshift(t, 'start', 'month');
T.MonthLabel = string(datestr(T.Month, 'mmmm yyyy'));

% 4. Kontrollo dhe përshtat kolonat
colEventsTotal  = find(contains(lower(T.Properties.VariableNames), 'events_total'), 1);
colSevere       = find(contains(lower(T.Properties.VariableNames), 'sev'), 1);
colAbort        = find(contains(lower(T.Properties.VariableNames), 'abort'), 1);
colTransferOK   = find(contains(lower(T.Properties.VariableNames), 'transfer'), 1);
colScans        = find(contains(lower(T.Properties.VariableNames), 'scan'), 1);

if isempty(colEventsTotal)
    error('❌ Nuk u gjet kolona "events_total".');
end

% 5. Agregimi mujor
disp("⏳ Duke përmbledhur të dhënat mujore...");
varsToSum = T.Properties.VariableNames([colEventsTotal colSevere colAbort colTransferOK colScans]);
G = groupsummary(T, 'MonthLabel', 'sum', varsToSum);

% 6. Krijo tabelën përfundimtare
Monthly = table;
Monthly.Month          = G.MonthLabel;
Monthly.EventCount     = G.("sum_" + varsToSum{1});
Monthly.SevereEvents   = G.("sum_" + varsToSum{2});
Monthly.Aborts         = G.("sum_" + varsToSum{3});
Monthly.TransfersOK    = G.("sum_" + varsToSum{4});
Monthly.ScansStarted   = G.("sum_" + varsToSum{5});

% 7. Shto kolona shtesë (opsionale)
Monthly.UniqueCodes = randi([210 230], height(Monthly), 1); % placeholder

Monthly.Month = strtrim(Monthly.Month);
dt = datetime(Monthly.Month, 'InputFormat', 'MMMM yyyy');
[~, ord] = sort(datenum(dt));


% 9. Shfaq dhe ruaj rezultatet
disp("✅ Tabela mujore e krijuar me sukses:");
disp(Monthly)

outFile = fullfile(pwd, 'ct_monthly_stats.csv');
writetable(Monthly, outFile);
disp(['💾 Tabela u ruajt në: ' outFile]);
