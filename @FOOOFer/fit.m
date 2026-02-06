function [results, ap_fitter, p_fitter] = fit(self, freqs, spectrum, pv)%include_freq_range, exclude_freq_range, apriori_peak_range)

arguments

    self FOOOFer
    freqs (:, 1) double
    spectrum (:, :) double

    pv.included_frequencies (:, 2) double = do.range(freqs) %all by default
    pv.excluded_frequencies (:, 2) double = self.excluded_frequencies
    pv.apriori_peak_range = []

    pv.apriori_fooofer FOOOFer = FOOOFer.empty()% fooofer object (e.g. previous fitting 
    % results on a neighboring channel's data) to use as estimates for the 
    % current dataset    
    pv.skip_aperiodic_lrt = false % if true, must provide estimates
    pv.peak_center_offset = 2 % when synching initial periodic estimates with data-driven estimates match the center frequencies within tol
    pv.max_peak_frequency_offset = NaN % determines bounds of center frequency from the initial guesses
    pv.synch_tol = 2
    pv.includeKnee = false
    pv.lrt_knee = true % whether to compare aperiodic models w and w/o knee    
    pv.lrt_p_threshold = .05;

    pv.stop_aperiodic_lrt_after = 3;
    pv.stop_periodic_lrt_after = 2;
    pv.periodic_prior_updates_after = 0;
    
    pv.plot = false
    pv.rng_seed = randi(2^8)
    pv.mode {mustBeMember(pv.mode, {'exploratory', 'apriori'})} = 'exploratory'

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
    isFreqExcld = isFreqExcld | do.ifwithin(freqs, exclude_freq_range(ii,:));
end


if ~isempty(include_freq_range)

    isFreqIn = isFreqIn & do.ifwithin(freqs, include_freq_range);

end

analysis_freqs = freqs(isFreqIn & ~isFreqExcld);
original_spectrum = log10(spectrum(isFreqIn & ~isFreqExcld,:));
analysis_spectrum = original_spectrum;
% If there are more than 15 observations, determine outliers and assign
% weights
if size(analysis_spectrum,2) > 15 % in the futue, make this optional

    z_spectrum = do.robust_z(analysis_spectrum, 2); % amount of noise depends on the frequency, calculate robust z per each frequency
    weights = abs(z_spectrum) < 5;
else
    weights = 1;

end
isFreqExcldFromAp = false(size(analysis_freqs));
for ii = 1:size(apriori_peak_range,1)
    isFreqExcldFromAp = isFreqExcldFromAp | do.ifwithin(analysis_freqs, apriori_peak_range(ii,:));
end

%% Configure fitting procedure
% if there is an apriori_fooorer skip the initial iter
if  ~isempty(pv.apriori_fooofer)

    init_ap_res = pv.apriori_fooofer.retrieve('estimates', type='aperiodic');
    init_p_res = pv.apriori_fooofer.retrieve('estimates', type='periodic');
    ap_params = [init_ap_res(1).fit,... % best model without knee
            NaN; init_ap_res(2).fit];;
    findPeaks = true;
    p_lrt = 'reduced';
    self.iter = 1; % Step 3 is the iterative fine-tuning of Step 1 and 2
    n_peaks = self.max_n_peaks;
    includeKnee = false;
    ap_lrt = ~pv.skip_aperiodic_lrt;
    pv.periodic_prior_updates_after = 0;


else

    init_p_res = struct([]);
    findPeaks = true;
    p_lrt = 'null';
    self.iter = 0; % Step 3 is the iterative fine-tuning of Step 1 and 2
    n_peaks = self.max_n_peaks;
    includeKnee = pv.includeKnee;
    ap_lrt = ~pv.skip_aperiodic_lrt;
    ap_params = [];

end
%%

p_mdl = SumOfGaussians(n_peaks = n_peaks, ...
    min_peak_width = self.min_peak_width,...
    min_peak_distance = self.min_peak_distance,...
    min_peak_frequency = self.min_peak_frequency,...
    max_peak_frequency = self.max_peak_frequency,...
    max_peak_frequency_offset = pv.max_peak_frequency_offset,...
    verbose = false);
ap_mdl = ExponentialPowerLaw(includeKnee = includeKnee, ...
    inLogScale=true, verbose=false);
p_fitter = NLLFitter(p_mdl);
ap_fitter = NLLFitter(ap_mdl);

% Provide data to models


p_mdl.X = analysis_freqs;
p_mdl.Y = analysis_spectrum;
p_mdl.W = weights;


ap_mdl.X = analysis_freqs(~isFreqExcldFromAp);
ap_mdl.Y = analysis_spectrum(~isFreqExcldFromAp,:);
ap_mdl.W = weights;

if ~isempty(init_p_res)
    % update Y
    p_init_fit = p_mdl.predict(init_p_res.fit, X=ap_mdl.Y);
    ap_mdl.Y = ap_mdl.Y - p_init_fit;
end

if pv.plot()

    fig = figure;
    tiles = tiledlayout('flow', Padding = 'compact', ...
        TileSpacing = 'compact');

    ylims = do.range(original_spectrum);
    ylims = ylims + [-1,1] * diff(ylims)*.1;

end


%% Main Loop
hasConverged = false;
runInitialFit = ~self.iter;
while self.iter <= self.max_refit_n_iter && ~hasConverged

    if pv.plot
        
        tileN = nexttile();
        plot(analysis_freqs, original_spectrum, LineWidth=1.5);
        title(tileN, sprintf('Iteration \\#%d', self.iter), Interpreter='latex');
        xlabel(tileN, 'Frequency (Hz)', Interpreter='latex');
        ylabel(tileN, 'Amplitude ($\log_{10}(\mu V)$)', Interpreter='latex')
        ylim(ylims)
        hold on;
        drawnow();

    end
    
    %% Estimate periodic component and subtract from the original spectrum
    if self.iter == 0
        p_p0 = p_mdl.estimate();
        n_peaks = (numel(p_p0)-1)/3;
        p_p0(end) = 0; % no baseline to correct for
        if n_peaks ~= p_mdl.n_peaks
            % triggers re-solving the model
            p_mdl.n_peaks = n_peaks;
        end
        p_init_fit = p_mdl.predict(p_p0, X=analysis_freqs(~isFreqExcldFromAp));

        % update results
        self.append_to_results( ...
            iter = self.iter,...
            type = "initial_periodic_estimate",...
            seed = p_p0, ...
            nextEntry=true);

        % update y_data as the subtraction
        ap_mdl.Y = ap_mdl.Y - p_init_fit;
    end

    %% --- Aperiodic Fit ---    
    [ap_fitter, ap_res] = self.aperiodic_fit_(ap_fitter, ...
        lrt = ap_lrt,...
        p0 = ap_params);
    ap_mdl = ap_fitter.inner_model;

    %% --- Prep for Periodic Fit ---
    % Subtract aperiodic component from the original data
    ap_fit = ap_mdl.predict(X=analysis_freqs);
    p_mdl.Y = original_spectrum - ap_fit;

    if pv.plot

        plot(tileN, analysis_freqs, ap_fit, "--", LineWidth=1.5);
        drawnow();

    end

    %% --- Periodic Fit ---
    [p_fitter, p_res] = self.periodic_fit_(p_fitter, ...
        findPeaks=findPeaks, lrt=p_lrt, reduced_model = init_p_res,...
        synch_tol=pv.synch_tol);
    p_mdl = p_fitter.inner_model;
    p_fit = p_mdl.predict(X=analysis_freqs);
    ap_mdl.Y = original_spectrum - p_fit;   

    if pv.plot

        plot(tileN, analysis_freqs, ap_fit + p_fit, "-.", LineWidth=1.5);        
        drawnow();

    end

    %% --- Check Convergence ---

    % check GOF convergence
    true_residuals = original_spectrum - (ap_fit+p_fit);
    curr_aic = self.calculate_aic_(true_residuals, numel(p_res.fit)+numel(ap_res.fit));
    if self.iter >= (1+~runInitialFit)
        % check GOF convergence
        convergence =   curr_aic./prev_aic;        
        hasConverged = abs(1 - convergence) < 1e-4;
        if self.verbose

            fprintf("Iteration #%d\t, Convergence rate: %.2f%%\t, " + ...
                "Convergence = %d\n", self.iter, 100*convergence, hasConverged);

        end
    end
    prev_aic = curr_aic;

    %% --- Prep for next iter ---
    if self.iter <= pv.stop_aperiodic_lrt_after

        % in the next iteration aperiodic estimates must be best params        
        ap_params = self.retrieve('estimates', 'fit', type ='aperiodic');
        ap_params = [ap_params(1).fit,... % best model without knee
            NaN; ap_params(2).fit];
    else

        ap_params = ap_res.fit;
        ap_lrt = false;

    end

    if self.iter >= pv.periodic_prior_updates_after
        
        % start synching previous guesses

        init_p_res = p_res;
        findPeaks = true;
        p_lrt = 'reduced';
        
    elseif self.iter >= pv.stop_periodic_lrt_after

        % update reduced model based on effect size changes    
        init_p_res = p_res;
        p_lrt = 'none';
        findPeaks = false;

    end

    self.next();
    
end

results = self.results;
end % fit()

