function technicalparameters_monthly()

    % 1. Lexo tabelën origjinale nga log-et e përpunuara
    fname = 'FeatureTable_10min_2024_2025.csv';
    if ~isfile(fname)
        error('Nuk u gjet file-i: %s', fname);
    end
    T = readtable(fname, 'VariableNamingRule','preserve');

    % 2. Sigurohu që kemi kolonat që na duhen
    mustHave = {'timestamp','mean_gantry_tilt','mean_slice_thk', ...
                'time_since_last_event','burst_len_max_1m','entropy_codes'};
    for k = 1:numel(mustHave)
        if ~ismember(mustHave{k}, T.Properties.VariableNames)
            error('Kolona "%s" mungon në file. Shiko emrat e kolonave.', mustHave{k});
        end
    end

    % 3. Kthe timestamp në datetime
    ts = T.timestamp;
    if iscell(ts)
        ts = datetime(ts, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss', 'Format','yyyy-MM-dd HH:mm:ss', 'Locale','en_US');
    elseif isnumeric(ts)
        ts = datetime(ts, 'ConvertFrom','datenum');
    elseif isstring(ts)
        ts = datetime(ts, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss', 'Locale','en_US');
    end

    % 4. Ndërto emrin e muajit "November 2024" etj.
    monthStr = datestr(ts, 'mmmm yyyy');              % jep char
    monthStr = cellstr(monthStr);                     % kthe në cellstr

    % 5. Grupim sipas muajit
    [G, monthNames] = findgroups(monthStr);

    nM = numel(monthNames);
    Mean_Tilt_deg        = nan(nM,1);
    Mean_SliceThickness  = nan(nM,1);
    Median_TimeBetween   = nan(nM,1);
    Peak_Burst_1min      = nan(nM,1);
    Entropy_of_Codes     = nan(nM,1);

    for i = 1:nM
        idx = (G == i);

        % mesatarja e tilt-it
        Mean_Tilt_deg(i) = mean(T.mean_gantry_tilt(idx), 'omitnan');

        % mesatarja e slice thickness
        Mean_SliceThickness(i) = mean(T.mean_slice_thk(idx), 'omitnan');

        % median time between events
        Median_TimeBetween(i) = median(T.time_since_last_event(idx), 'omitnan');

        % peak burst
        Peak_Burst_1min(i) = max(T.burst_len_max_1m(idx), [], 'omitnan');

        % entropy
        Entropy_of_Codes(i) = mean(T.entropy_codes(idx), 'omitnan');
    end

    % 6. Rendit muajt sipas dates reale (jo alfabetikisht)
    monthDt = datetime(monthNames, 'InputFormat','MMMM yyyy', 'Locale','en_US');
    [monthDtSorted, ord] = sort(monthDt);

    Tbl = table( ...
        monthNames(ord), ...
        round(Mean_Tilt_deg(ord), 2), ...
        round(Mean_SliceThickness(ord), 2), ...
        round(Median_TimeBetween(ord), 1), ...
        round(Peak_Burst_1min(ord), 0), ...
        round(Entropy_of_Codes(ord), 2), ...
        'VariableNames', { ...
            'Month', ...
            'Mean_Tilt_deg', ...
            'Mean_SliceThickness_mm', ...
            'Median_Time_Between_Events_s', ...
            'Peak_Burst_1min', ...
            'Entropy_of_Codes'});

    disp(Tbl)

    % 7. (opsionale) eksportoje si Excel me emrin e njëjtë si në punim
    outName = 'Tabela3_MonthlyTech_from10min.xlsx';
    writetable(Tbl, outName, 'Sheet', 'MonthlyTech');
    fprintf('✅ Tabela u ruajt te: %s\n', outName);
end
