function [performance, results, ap_fitter, p_fitter] = fit(self, freqs, spectrum, pv)%include_freq_range, exclude_freq_range, apriori_peak_range)

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
    pv.max_peak_frequency_offset = NaN % determines bounds of center frequency from the initial guesses
    pv.synch_tol = 2
    pv.includeKnee = false
    pv.lrt_knee = true % whether to compare aperiodic models w and w/o knee    
    pv.lrt_p_threshold = .05;

    pv.stop_lrt_after = 3;
    pv.periodic_prior_updates_after = 2;
    
    pv.plot = false
    pv.rng_seed = randi(2^8)
    pv.gof_div_tol = 1e-2 % divergence tolerance
    pv.param_div_tol = 1e-2
    pv.prop_param_conv_tol = .9; % proportion of parameters to accept convergence

    pv.convergence_criteria {mustBeMember(pv.convergence_criteria, {'gof', 'parameter'})} = 'gof'

    pv.useGPU = false

end
%% --- Name Value Validation --- 
assert(pv.stop_lrt_after >= pv.periodic_prior_updates_after);
%%

rng(pv.rng_seed);
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
            NaN; init_ap_res(2).fit];
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
p_fitter.useGPU = pv.useGPU;
ap_fitter = NLLFitter(ap_mdl);
ap_fitter.useGPU = pv.useGPU;

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
lastSuccessIter = self.iter;
refineIter = false;
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
        p0 = ap_params, refineIter=refineIter);
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
        synch_tol=pv.synch_tol, refineIter=refineIter);
    p_mdl = p_fitter.inner_model;
    p_fit = p_mdl.predict(X=analysis_freqs);
    ap_mdl.Y = original_spectrum - p_fit;   

    if pv.plot

        plot(tileN, analysis_freqs, ap_fit + p_fit, "-.", LineWidth=1.5);        
        drawnow();

    end

    %% --- Check Convergence ---
    true_residuals = original_spectrum - (ap_fit+p_fit);
    curr_aic = self.calculate_aic_(true_residuals, numel(p_res.fit)+numel(ap_res.fit));
    self.performance_{end+1} = struct(iter = self.iter, aic = curr_aic, ...
        n_periodic_param = numel(p_res.fit), ...
        n_aperiodic_param = numel(ap_res.fit), ...
        gof_divergence = NaN,...
        prop_ap_converged = NaN,...
        prop_p_converged = NaN,...
        ap_divergence = NaN,...
        p_divergence = NaN);
    if self.iter >= (1+~runInitialFit)
        %% --- check GOF convergence ---
        gof_divergence =  (prev_aic-curr_aic)/abs(prev_aic);  
        %if the current fit significantly worse
        % go back to the initial guesses and tweak   
        self.performance_{end}.gof_divergence = gof_divergence;
        lastSuccessIter = self.best_iter;
        hasGOFConverged = abs(gof_divergence) < pv.gof_div_tol;
                   
                
        %% --- Check Parameter Divergence ---
        [ap_div, p_div] = check_parameter_divergence_(ap_res, p_res, prev_ap_res, prev_p_res);
        divAP = ap_div >= pv.param_div_tol;
        divP = p_div >= pv.param_div_tol;
        n_ap_param_conv = sum(~divAP);
        n_ap_param = numel(divAP);
        prop_ap_param_conv = n_ap_param_conv/n_ap_param;
        
        n_p_param_conv = sum(~divP);
        n_p_param = numel(divP);
        prop_p_param_conv = n_p_param_conv/n_p_param;

        n_param_conv = n_ap_param_conv + n_p_param_conv;
        n_param = n_ap_param + n_p_param;
        prop_param_cov = n_param_conv/n_param;     

        self.performance_{end}.ap_divergence = ap_div;
        self.performance_{end}.p_divergence = p_div;
        self.performance_{end}.prop_ap_converged = prop_ap_param_conv;
        self.performance_{end}.prop_p_converged = prop_p_param_conv;

        hasAPConverged = prop_ap_param_conv >= pv.prop_param_conv_tol;
        hasPConverged = prop_p_param_conv >= pv.prop_param_conv_tol;
        hasParamConverged = hasAPConverged && hasPConverged;

        % Test convergence_criteria
        switch pv.convergence_criteria 

            case 'gof'

                hasConverged = hasGOFConverged;

            case 'parameter'

                hasConverged = hasParamConverged;

        end

        if self.verbose

            fprintf("\tIteration #%d\t Divergence rate: %05.2f%%\t, " + ...
                "GOF Converged = %d\n", self.iter, 100*gof_divergence, hasGOFConverged);
            
            if gof_divergence <= 0
                fprintf("\tPrevious Fit was better. Reverting back estimates...\n");
            end

            fprintf("\tIteration #%d\t Parameter divergence:\n" + ...
                "\t\tAperiodic:\t sum = %05.2f\t mean = %05.2f\t %05.2f%% (%02d of %02d)\t Convergence = %d\n" + ...
                "\t\tPeriodic:\t sum = %05.2f\t mean = %05.2f\t %05.2f%% (%02d of %02d)\t Convergence = %d\n " + ...
                "\t\tCombined:\t sum = %05.2f\t mean = %05.2f\t %05.2f%% (%02d of %02d)\t Convergence = %d\n ",...
                self.iter, ...
                sum(ap_div), mean(ap_div), 100*prop_ap_param_conv, n_ap_param_conv, n_ap_param,hasAPConverged,...
                sum(p_div), mean(p_div), 100*prop_p_param_conv, n_p_param_conv, n_p_param, hasPConverged,...
                sum([ap_div, p_div]), mean([ap_div, p_div]), 100*prop_param_cov,  n_param_conv, n_param, hasParamConverged);

            fprintf('ITERATION CONVERGENCE CRITERIA (%s) ACHIEVED? = %d.\n', ...
                upper(pv.convergence_criteria), hasConverged);

        end            
        
    end     

    %% --- Prep for next iter ---
    if self.iter <= pv.stop_lrt_after
        
        ap_params = self.retrieve('estimates', 'fit', type ='aperiodic', ...
            iter = lastSuccessIter);
        ap_params = [ap_params(1).fit,... % best model without knee
            NaN; ap_params(2).fit];  
       
    else

        ap_params = self.retrieve('survived', 'fit', type ='aperiodic', ...
            iter = lastSuccessIter);
        ap_params = ap_params.fit;
        ap_lrt = false;
        refineIter = true;
    end

    if self.iter >= pv.stop_lrt_after

        % update reduced model based on effect size changes    
        init_p_res = self.retrieve('survived', type ='periodic', ...
            iter = lastSuccessIter);
        p_lrt = 'none';
        findPeaks = false;
        refineIter = true;

    elseif self.iter >= pv.periodic_prior_updates_after
        
        % start synching previous guesses
        init_p_res = self.retrieve('estimates', type ='periodic', ...
            iter = lastSuccessIter);
        findPeaks = true;
        p_lrt = 'reduced';
               
    end


    if self.iter >= (1+~runInitialFit) && lastSuccessIter ~= self.iter
        
        dev_amount = init_p_res.fit.*.1;
        init_p_res.fit = init_p_res.fit + randn(size(dev_amount)).*dev_amount;

        dev_amount = ap_params.*.1;
        ap_params = ap_params + randn(size(dev_amount)).*dev_amount;

        % change aperiodic model data, it should be the same as
        % at the begining of lastSuccessIter
        ap_fit = p_mdl.predict(prev_p_res.fit(1:end-1));
        ap_mdl.Y = original_spectrum - ap_fit;

    else

        prev_aic = curr_aic;
        prev_ap_res = ap_res;
        prev_p_res = p_res;

    end

    self.next();    

    
    
end

results = self.results;
performance = self.performance_results;
end % fit()

%% LOCAL HELPER FUNCTIONS
function [ap_div, p_div] = check_parameter_divergence_(ap_res, p_res)

arguments (Input, Repeating)

    ap_res struct
    p_res struct

end

ap_div = check_ap_parameter_divergence_(ap_res{:});
p_div = check_p_parameter_divergence_(p_res{:});
end

function div = check_ap_parameter_divergence_(res0, res1)

    function p_out = parse_params_(p_in)
               
        p_out = [p_in(1:2), 0, p_in(end)];
        if numel(p_in) > 3
            p_out(3) = p_in(3);
        end
        p_out = p_out+eps; % avoid div by 0

    end

p0 = parse_params_(res0.fit);
p1 = parse_params_(res1.fit);

div = abs((p0-p1)./p0);
end

function div = check_p_parameter_divergence_(res0, res1)

    nonpeak0 = res0.fit(end-1:end);
    nonpeak1 = res1.fit(end-1:end);
    peak0 = res0.fit(1:end-2);
    peak1 = res1.fit(1:end-2);
    [~,~, mask0, mask1] = SumOfGaussians.synch_peak_parameters_(peak0, peak1);

    calc_div = @(p0,p1) abs((p0-p1)./p0);
    div = [calc_div(peak0(mask0), peak1(mask1)),...
        ones(1, sum(~mask0)), ones(1, sum(~mask1)),...
        calc_div(nonpeak0, nonpeak1)];

end