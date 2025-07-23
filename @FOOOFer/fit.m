function [results, ap_fitter, p_fitter] = fit(obj, freqs, spectrum, pv)%include_freq_range, exclude_freq_range, apriori_peak_range)

arguments

    obj FOOOFer
    freqs (:, 1) double
    spectrum (:, :) double

    pv.included_frequencies (:, 2) double = gen.range(freqs) %all by default
    pv.excluded_frequencies (:, 2) double = obj.excluded_frequencies
    pv.apriori_peak_range = []
    
    pv.plot = false
    pv.rng_seed = randi(2^8)

end

rng(pv.rng_seed);

line_styles = ["-", ":"];
% Main fitting method.
include_freq_range = pv.included_frequencies;
exclude_freq_range = pv.excluded_frequencies;
apriori_peak_range = pv.apriori_peak_range;

% --- Preparations ---
isFreqIn = freqs~=0;

isFreqExcld = false(size(isFreqIn));
for ii = 1:size(exclude_freq_range,1)
    isFreqExcld = isFreqExcld | gen.ifwithin(freqs, exclude_freq_range(ii,:));
end


if ~isempty(include_freq_range)

    isFreqIn = isFreqIn & gen.ifwithin(freqs, include_freq_range);

end

analysis_freqs = freqs(isFreqIn & ~isFreqExcld);
original_spectrum = log10(spectrum(isFreqIn & ~isFreqExcld,:));
analysis_spectrum = original_spectrum;
% If there are more than 15 observations, determine outliers and assign
% weights
if size(analysis_spectrum,2) > 15 % in the futue, make this optional

    z_spectrum = gen.robust_z(analysis_spectrum, 2); % amount of noise depends on the frequency, calculate robust z per each frequency
    weights = abs(z_spectrum) < 5;
else
    weights = 1;

end
isFreqExcldFromAp = false(size(analysis_freqs));
for ii = 1:size(apriori_peak_range,1)
    isFreqExcldFromAp = isFreqExcldFromAp | gen.ifwithin(analysis_freqs, apriori_peak_range(ii,:));
end

obj.iter = 0; % Step 3 is the iterative fine-tuning of Step 1 and 2
n_peaks = obj.max_n_peaks;
includeKnee = false;

if pv.plot

    figure;
    plot(analysis_freqs, original_spectrum, LineAlpha=.5);
    hold on;
    drawnow();

end

isConverged = false;
while obj.iter <= obj.max_refit_n_iter && ~isConverged

    %% -- Step 1: Aperiodic Fits ---

    % initiate the periodic and aperiodic model
    p_mdl = SumOfGaussians(n_peaks = n_peaks, ...
        min_peak_width = obj.min_peak_width,...
        min_peak_distance = obj.min_peak_distance,...
        min_peak_frequency = obj.min_peak_frequency,...
        max_peak_frequency = obj.max_peak_frequency,...
        verbose = false);
    ap_mdl = ExponentialPowerLaw(includeKnee = includeKnee, inLogScale=true, verbose=false);
    
    % Provide data to models
    p_mdl.X = analysis_freqs;
    p_mdl.Y = analysis_spectrum;   
    p_mdl.W = weights;

    ap_mdl.X = analysis_freqs(~isFreqExcldFromAp);
    ap_mdl.Y = analysis_spectrum(~isFreqExcldFromAp,:);
    ap_mdl.W = weights;
    
    p_fitter = NLLFitter(p_mdl);
    ap_fitter = NLLFitter(ap_mdl);
    % --- Step 1: Aperiodic Fit (Robust) ---
    if obj.iter <= 1
        % In Steps 0 & 1 re-estimate aperiodic params from
        % data which is 'original_spectrum' in Step #0 and
        % 'original_spectrum/periodic_fit' in Step #1
        if obj.iter == 0
            
            p_p0 = p_mdl.estimate(analysis_freqs, analysis_spectrum);
            n_peaks = (numel(p_p0)-1)/3;
            p_p0(end) = 0; % no baseline to correct for
            if n_peaks ~= p_mdl.n_peaks
                % triggers re-solving the model
                p_mdl.n_peaks = n_peaks;
            end
            p_init_fit = p_mdl.predict(p_p0, analysis_freqs);

            % update results
            obj.append_to_results( ...
                iter = obj.iter,...
                type = "initial_periodic_estimate",...
                seed = p_p0, ...
                nextEntry=true);
            
            % update y_data as the subtraction
            ap_mdl.Y = analysis_spectrum(~isFreqExcldFromAp,:) - p_init_fit(~isFreqExcldFromAp);
            ap_mdl.W = weights;

        end
        
        ap_p0 = ap_fitter.estimate();
            
    else

        % For the subsequent fits, always use the previously
        % fitted parameters.        
        ap_p0 = ap_params;

    end 

    % update results
    obj.append_to_results( ...
        iter = obj.iter,...
        type = "aperiodic",...
        seed = ap_p0, ...
        nextEntry = true);

    % --- Aperiodic Fit ---

    % set up optimizer options
    ap_optimopts = optimoptions('fmincon',...
        'Display', 'off',...
        'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
        'SpecifyObjectiveGradient', true, ...
        'HessianFcn', @(p, ~) ap_fitter.compute_hessian(p), ...
        'MaxFunctionEvaluations', obj.max_func_eval, ...
        'MaxIterations', obj.max_fit_iter);

    % estimate reasonable bounds from data
    [ap_lb, ap_ub] = ap_fitter.estimate_parameter_bounds();

    if obj.verbose

        msg = 'Fitting the aperiodic component with';
        if ap_mdl.includeKnee
            msg = [msg, ' knee...\n\t'];
        else
            msg = [msg, 'out knee...\n\t'];
        end
        fprintf(msg);

    end
    
    tic;
    
    [ap_params, fval, flag, op] = fmincon(@(p) ap_fitter.objective_function(p), ...
            ap_p0, [], [], [], [], ap_lb, ap_ub, [], ap_optimopts);
    
    if obj.verbose, toc; end

    % Calculate AIC

    curr_ap_aic = obj.calculate_aic_(ap_mdl.wR, numel(ap_params));

    % update results
    obj.append_to_results( ...
        iter = obj.iter,...
        type = 'aperiodic',...
        fit = ap_params, ...
        fit_flag=flag, ...
        fit_n_iter=op.iterations, ...
        fit_n_func_eval=op.funcCount, ...
        fit_n_pcg_iter=op.cgiterations, ...
        gof=curr_ap_aic, ...
        gof_metric='aic');

    if includeKnee

        prevModelSurvived = curr_ap_aic >= simple_ap_aic && flag > 0;

        % update the entry for the previous model
        obj.append_to_results(modelSurvived = prevModelSurvived, step_from_current_entry = -1 - (obj.iter < 1));

    end


    % If includeKnee == false, save as simple_ap_mdl, and continue next
    % iteration with includeKnee = true

    if obj.iter <= 1 && ~includeKnee

        includeKnee = true;
        
        simple_ap_fitter = ap_fitter;
        simple_ap_params = ap_params;
        simple_ap_aic = curr_ap_aic;

        % simple_ap_iter = ap_iter;
        % simple_ap_isConverged = ap_isConverged;
        
        continue;

    % If includeKnee = true, compare AICs, and select ap_model
    elseif obj.iter <= 1 && prevModelSurvived

            ap_fitter = simple_ap_fitter;
            ap_mdl = simple_ap_fitter.inner_model;
            ap_params = simple_ap_params;

    else
        
        modelSurvived = ~prevModelSurvived;
        if isempty(modelSurvived), modelSurvived = NaN; end
        obj.append_to_results(modelSurvived=modelSurvived);
    
    end
    
    % --- Periodic Fit ---

    % Fit aperiodic function, and flatten the original data
    ap_fit = ap_mdl.predict([], analysis_freqs);
    p_mdl.Y = original_spectrum - ap_fit;
    p_mdl.W = weights;

    if pv.plot

        plot(analysis_freqs, ap_fit, LineWidth = 3);
        hold on;
        drawnow();

    end
        
    % Get initial estimates
    if obj.iter <= 1
        p_fitter.inner_model.n_peaks = obj.max_n_peaks;
        p_p0 = p_fitter.estimate();
        
    else
        p_p0 = p_params; 
    end
    n_peaks = (numel(p_p0)-2)/3;

    % Estimate bounds from data  

    % Iteratively add more peaks and compare AIC's to determine whether
    % add a new gaussian or not
    iKeepPeak = [];
    % first gof comes from the flattened spectrum since it is the residual after
    % aperiodic fit.
    gof = obj.calculate_aic_(p_mdl.Y .* p_mdl.W , numel(ap_params));
    gof_prev = gof;

    curr_p0 = p_p0;
    iTestPeak = 1;

    p_optimopts = optimoptions('fmincon',...
        'Display', 'off',...
        'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
        'SpecifyObjectiveGradient', true, ...
        'HessianFcn', @(p, ~) p_fitter.compute_hessian(p), ...
        'MaxFunctionEvaluations', obj.max_func_eval, ...
        'MaxIterations', obj.max_fit_iter);
    % isempty(iTestPeak) = true during refitting
    while iTestPeak <= n_peaks 

        n_peaksN = numel(iKeepPeak) + numel(iTestPeak);
        p_mdl.n_peaks = n_peaksN;

        kept_peaks = arrayfun(@(i) curr_p0(i:n_peaks:end-2), iKeepPeak, UniformOutput=false);
        
        
        kept_peaks = vertcat(kept_peaks{:});

        pN = [kept_peaks; curr_p0(iTestPeak:n_peaks:end-2)];
        bN = curr_p0(end-1);
        sigmaN = curr_p0(end);
        pN = [pN(:); bN; sigmaN]';

        % Estimate bounds from data
        [lbN, ubN] = p_fitter.estimate_parameter_bounds();

        % Get linear inequality matrix to ensure each peak is searched
        % within its own bounds
        %
        % The class method assumes inner_model parameters are represented 
        % at the begining of pN vector
        [A, b] = p_mdl.return_linineq_for_peak_centers(pN);
        
        % Perform fitting
        if obj.verbose

            msg = 'Fitting the periodic component with Peak #%d....\n\t';
            if isempty(iTestPeak)
                fprintf("Fiting the periodic component with final estimates...\n\t");
            else
                fprintf(msg, iTestPeak);
            end
        end
        
        tic;
        [p_paramsN, fval, flag, op] = fmincon(@(p) p_fitter.objective_function(p), ...
            pN, A, b, [], [], lbN, ubN, [], p_optimopts);
        if obj.verbose, toc; end

        % calculate gof
        gofN = obj.calculate_aic_(p_mdl.wR, numel(p_paramsN));

        % if better gof, keep the peak
        modelSurvived = gofN < gof_prev && flag > 0;
        if ~isempty(iTestPeak) && modelSurvived

            iKeepPeak(end+1) = iTestPeak;          
            gof_prev = gofN;
            gof(end + 1) = gofN;

            % update inital estimates
            for iKept = 1:n_peaksN
                curr_p0(iKeepPeak(iKept):n_peaks:end-2) = p_paramsN(iKept:n_peaksN:end-2);
            end
            curr_p0(end-1:end) = p_paramsN(end-1:end);
            
        end

        % Update the results here,
        obj.append_to_results( ...
            iter = obj.iter, ...
            seed = pN,...
            type = 'periodic', ...
            fit = p_paramsN, ...
            fit_flag = flag, ...
            fit_n_iter=op.iterations, ...
            fit_n_func_eval=op.funcCount, ...
            fit_n_pcg_iter=op.cgiterations, ...
            gof=gofN, ...
            gof_metric='aic', ...
            modelSurvived=modelSurvived, ...
            nextEntry=true);

        iTestPeak = iTestPeak + 1;        
        
    end   

    % Select the last serviving model parameters
    if isempty(iKeepPeak)

        % No periodic components
        % end fitting loop
        break;


    else

        % last surviving model's index in results struct
        iRes = numel(obj.results) - (n_peaks - max(iKeepPeak));
        p_params = obj.results(iRes).fit;

        % update the model with the correct no of fits        
        n_peaks = numel(iKeepPeak);
        p_mdl.n_peaks = n_peaksN;
        
        % predict the periodic component and subtract it from the original
        % y to fit the aperiodic component in the next iteration, add the
        % baseline estimate for a correct estimate of intercept
        p_fit = p_mdl.predict(p_params(1:end-1));
        analysis_spectrum = original_spectrum - p_fit + p_params(end-1);

        includeKnee = false;

    end

    if pv.plot

        plot(analysis_freqs, ap_fit+p_fit, line_styles((mod(obj.iter,2)==1)+1), LineWidth = 3);
        hold on;
        drawnow();

    end

    % continue to next iteration
    obj.next();
    if obj.iter > 1
        break;
    end
    
end

results = obj.results;
end % fit()


