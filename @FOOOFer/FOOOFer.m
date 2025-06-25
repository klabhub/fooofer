classdef FOOOFer < handle

    % O1OFFitter Class for parameterizing neural power spectra.
    % Implements a FOOOF-like algorithm with multi-pass peak detection.
    % Peak amplitudes are fitted in log-space internally but stored and outputted in linear space.

    properties

        % Aperiodic Fit Settings

        max_irls_iterations (1,1) double = 10       % Max iterations for IRLS in robust aperiodic fit
        irls_tolerance (1,1) double = 1e-5        % Tolerance for IRLS convergence    

        % Peak Fit Settings
        flattened_spectrum_smoothing_factor_in_hz (1,1) double = 3 % to find the potential peaks smooths the flattened spectrum using median filtering
        
        peak_freq_range (1,2) double = [.75, 100]
        peak_threshold_sd (1,1) double = 1.0      % Std. dev. above mean of flattened spec to detect peak
        min_peak_height (1,1) double = 0.0        % Absolute minimum height of a peak on flattened spectrum (linear units)
        min_peak_width (1,1) double = 1
        peak_width_limits (:,2) double = [0, 4] % Each row is [min_width, max_width] for a pass
        max_n_peaks (1,1) double = 8              % Maximum number of peaks to fit per pass
        contains_narrow_peaks (1,1) logical = false % If true, peak bandwidth (BW) is fitted in log10 space.
        peak_proximity_threshold (1,1) double = 3.0 % Minimum Hz separation to keep distinct peaks
        
        % Step 4 Iterative Refinement Settings
        max_refit_iterations (1,1) double = 20    % Max iterations for Step 4 refinement loop
        convergence_tolerance_fits (1,1) double = 1e-3 
        convergence_tolerance_gof (1,1) double = 1e-4     

        % Fitting Options
        lsq_options % Options for lsqcurvefit

        % Properties to be updated within the iterative fitting loop
        iter = 0
        seeds = table(Size = [1, numel(FOOOFer.seeds_vars_)],...
            VariableNames = FOOOFer.seeds_vars_, ...
            VariableTypes = FOOOFer.seeds_var_types_)
        fits = table(Size = [1, numel(FOOOFer.fits_vars_)],...
            VariableNames = FOOOFer.fits_vars_, ...
            VariableTypes = FOOOFer.fits_var_types_)
        gof = table(Size = [1, numel(FOOOFer.gof_vars_)],...
            VariableNames = FOOOFer.gof_vars_, ...
            VariableTypes = FOOOFer.gof_var_types_)

        % gof table = table(...
        %     'VariableNames', {'R2', 'AIC', 'BIC'})

        % --- Results from the final iteration of the fit ---
        iteration (1,1) double = 0        
        R_squared (1,1) double = NaN
        MSE (1,1) double = NaN
        aperiodic_converged (1,1) logical = false
        periodic_converged (1,1) logical = false
        gof_converged (1,1) logical = false
        step4_converged_overall (1,1) logical = false
        error_message char = ''

    end

    properties (Constant, Hidden)

        seeds_vars_ = {'iter', 'intercept', 'knee', ...
            'exponent', 'n_peaks', 'amplitude', 'center', 'bandwidth', ...
            'baseline'};
        seeds_var_types_ = {'uint64', 'double', 'double',...
            'double', 'uint32', 'cell', 'cell', 'cell',...
            'double'}

        fits_vars_ = {'iter', 'n_ap_fit_iter', 'convergence', ...
            'intercept', 'exponent', 'knee', 'p_fit_bics', ....
            'n_peaks', 'amplitude', 'center', 'bandwidth', 'baseline'}

        fits_var_types_ = {'uint64', 'uint64', 'logical',...
            'double', 'double', 'double', 'cell',...
            'uint32', 'cell', 'cell', 'cell', 'double'}

        gof_vars_ = {'iter', 'n_param', 'r2', 'aic', 'bic'}
        gof_var_types_ = {'uint64', 'uint16', 'double', 'double', 'double'}

        free_ap_parameters = {'intercept', 'exponent', 'knee'}
        free_p_parameters = {'amplitude', 'center', 'bandwidth', 'baseline'}
    
    end

    properties (Access = protected)

        %%
        % Parameters estimates, rows are the results of each iteration
        intercept_ (:,1) = []
        knee_ (:,1) = []
        exponent_ (:,1) = []

        center_frequency_ (:,1) = {}
        amplitude_ (:,1) = {}
        bandwidth_ (:,1) = {}
        baseline_ (:,1) = {}
        parameter_convergence_ table %= table(...
            % 'VariableNames',{'intercept', 'exponent', 'knee','amplitude','center_frequency','bandwidth','baseline'})

        
    end

    properties (Dependent)

        aperiodic_seeds
        periodic_seeds
        aperiodic_fits
        periodic_fits

        parameter_convergence
        gof_convergence
        
        % current estimates from the last iteration
        intercept (1,1) double
        knee (1,1) double
        exponent (1,1) double
        center_frequency (1,:) double
        amplitude (1,:) double
        bandwidth (1,:) double    
        baseline (1,:) double
        isParametersConverged
        isGOFConverged

    end

    properties (Dependent, Access = private)

        current_row_idx_

    end

    methods

        function obj = FOOOFer(varargin)
            % Constructor for O1OFFitter
            obj.lsq_options = optimoptions('lsqcurvefit', 'Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6, 'MaxIterations', 1000, 'MaxFunctionEvaluations', 3000);
            if nargin > 0
                if mod(nargin, 2) ~= 0
                    error('FOOOFer:InvalidInput', 'Invalid number of input arguments. Must be name-value pairs.');
                end
                for i = 1:2:nargin
                    prop_name = varargin{i};
                    prop_val = varargin{i+1};
                    if isprop(obj, prop_name)
                        obj.(prop_name) = prop_val;
                    else
                        warning('FOOOFer:InvalidProperty', 'Property "%s" does not exist.', prop_name);
                    end
                end
            end
            
        end

        function fit( ...
                obj, ...
                freqs_full, ...
                spectrum_full, ...
                freq_range_to_fit, ...
                exclude_freq_range, ...
                apriori_peak_range...
                )
            % Main fitting method.
            
            if nargin < 4 || isempty(freq_range_to_fit)
                freq_range_to_fit = gen.range(freqs_full);
            end

            if nargin < 5
                exclude_freq_range = []; 
            end

            if nargin < 6
                apriori_peak_range = [];
            end

            % --- Preparations ---
            isFreqIn = freqs_full~=0;

            isFreqExcld = false(size(isFreqIn));
            for ii = 1:size(exclude_freq_range,1)
                isFreqExcld = isFreqExcld | gen.ifwithin(freqs_full, exclude_freq_range(ii,:));
            end         
            

            if ~isempty(freq_range_to_fit)

                isFreqIn = isFreqIn & gen.ifwithin(freqs_full, freq_range_to_fit);

            end

            analysis_freqs = freqs_full(isFreqIn & ~isFreqExcld);
            original_spectrum = spectrum_full(isFreqIn & ~isFreqExcld);
            analysis_spectrum = original_spectrum;

            isFreqExcldFromAp = false(size(analysis_freqs));
            for ii = 1:size(exclude_freq_range,1)
                isFreqExcldFromAp = isFreqExcldFromAp | gen.ifwithin(analysis_freqs, apriori_peak_range(ii,:));
            end

            obj.iter = 0; % Step 3 is the iterative fine-tuning of Step 1 and 2
            
            isConverged = false;
            while obj.iter <= obj.max_refit_iterations && ~isConverged

                
                % --- Step 1: Initial Aperiodic Fit (Robust) ---
                if obj.iter <= 1
                    % In Steps 0 & 1 re-estimate aperiodic params from
                    % data which is 'original_spectrum' in #S0 and
                    % 'original_spectrum/periodic_fit' in #S1
                    ap_param_guesses = obj.estimate_aperiodic_params_from_data(analysis_freqs, analysis_spectrum);
                    
                else

                    % For the subsequent fits, always use the previously
                    % fitted parameters.
                    ap_param_guesses = ap_param_fits;
                end

                ap_param_guesses = num2cell(ap_param_guesses);
                obj.update('seeds', 'aperiodic', ap_param_guesses{:});


                ap_param_fits = obj.perform_aperiodic_fit_(analysis_freqs(~isFreqExcldFromAp), analysis_spectrum(~isFreqExcldFromAp), true);
                ap_fit = obj.aperiodic_function(ap_param_fits,analysis_freqs);
                flattened_spectrum = log10(original_spectrum) - log10(ap_fit);

                if obj.iter > 1

                    p_param_guesses = p_param_fits;
                    % p_param_guesses = p_param_fits;                    
                    % flattened_spectrumN = flattened_spectrum - obj.predict(analysis_freqs, 'periodic', p_param_fits);
                    % n_peaks_fit_prev = obj.fits.n_peaks(obj.current_row_idx_-1);
                    % obj.max_n_peaks = obj.max_n_peaks - n_peaks_fit_prev;
                    % if obj.max_n_peaks
                    %     p_param_guesses = estimate_periodic_params_from_data(obj, analysis_freqs, flattened_spectrumN);
                    %     [pN, bN] = obj.rearrange_periodic_params(p_param_guesses);
                    %     [prev_pN, prev_bN] = obj.rearrange_periodic_params(p_param_fits);
                    %     pN = [prev_pN; pN];
                    %     p_param_guesses = [pN(:); prev_bN];
                    % else
                    %     p_param_guesses = p_param_fits;
                    % end                    
                    % 
                    % obj.max_n_peaks = obj.max_n_peaks + n_peaks_fit_prev;

                else

                    p_param_guesses = estimate_periodic_params_from_data(obj, analysis_freqs, flattened_spectrum);

                end


                [pN, bN] = obj.rearrange_periodic_params(p_param_guesses);
                obj.update('seeds', 'periodic', pN(:,1), pN(:,2), pN(:,3), bN);

                % --- Step 2: Fit Periodic Components Identification Iteratively, adding one peak at a time to the model ---
                p_param_fits = obj.perform_periodic_fit_(p_param_guesses, analysis_freqs, flattened_spectrum);
                
                p_fit = obj.predict(analysis_freqs, 'periodic');               
                p_fit_log = log10(p_fit);
                % Goodness of Fit and convergence of GOF
                n_param = numel(p_param_fits) + numel(ap_param_fits);
                [R2, AIC, BIC] = obj.calculate_gof(log10(original_spectrum), log10(ap_fit)+p_fit_log, n_param);
                obj.update('gof', n_param, R2, AIC, BIC);
                % --- Step 3: Check convergence and GOF
                if obj.iter > 1 
                    
                    params_conv = all(table2array(obj.isParametersConverged));
                    gof_conv = obj.isGOFConverged.aic;    

                    if params_conv && gof_conv

                        fprintf("Converged at Iteration %d.\n", obj.iter);
                        isConverged = true;
                        continue;

                    end
                    % discard non-converged peaks from the next estimate
                    % if diverging

                end                

                analysis_spectrum = 10.^(log10(original_spectrum) - p_fit_log);
                               
               obj.next();
               
            end

        end % fit()     
        
        function next(obj)

            if obj.iter >= obj.max_refit_iterations
                return;

            end
            obj.iter = obj.iter + 1;
            obj.seeds.iter(end+1) = obj.iter;
            obj.fits.iter(end+1) = obj.iter;
            obj.gof.iter(end+1) = obj.iter;
            % update tables

        end

        function update(obj, tbl_name, varargin)
           
            n_arg = numel(varargin);
            isAP = strcmp(varargin{1}, "aperiodic");
            idx = obj.current_row_idx_;

            switch tbl_name

                case 'seeds'                                      

                    if isAP
                        
                        obj.seeds.intercept(idx) = varargin{2};
                        obj.seeds.exponent(idx) = varargin{3};
                        if n_arg > 3
                            obj.seeds.knee(idx) = varargin{4};
                        end

                    else
                        
                        obj.seeds.amplitude{idx} = varargin{2};
                        obj.seeds.center{idx} = varargin{3};
                        obj.seeds.bandwidth{idx} = varargin{4};
                        obj.seeds.n_peaks(idx) = numel(varargin{2});                        
                        obj.seeds.baseline(idx) = varargin{5};

                    end

                case 'fits'


                    if isAP
                        
                        obj.fits.n_ap_fit_iter(idx) = varargin{2};
                        obj.fits.convergence(idx) = varargin{3};
                        obj.fits.intercept(idx) = varargin{4};
                        obj.fits.exponent(idx) = varargin{5};
                        if n_arg >= 6

                            obj.fits.knee(idx) = varargin{6};
                        else
                            obj.fits.knee(idx) = NaN;

                        end

                    else

                        obj.fits.p_fit_bics{idx} = varargin{2};
                        obj.fits.amplitude{idx} = varargin{3};
                        obj.fits.n_peaks(idx) = numel(varargin{3});
                        obj.fits.center{idx} = varargin{4};
                        obj.fits.bandwidth{idx} = varargin{5};
                        obj.fits.baseline(idx) = varargin{6};

                    end

                case 'gof'

                    obj.gof.n_param(idx) = varargin{1};
                    obj.gof.r2(idx) = varargin{2};
                    obj.gof.aic(idx) = varargin{3};
                    obj.gof.bic(idx) = varargin{4};
            end

        end

        function y_hat = predict(obj, x, component_name, p)
            

            if nargin < 3
                component_name = "";
            end

            
            y_hat = zeros(size(x));

            if ~strcmp(component_name, "periodic") % if 'aperiodic' or ''

                if nargin < 4
                    p = obj.aperiodic_fits;
                end
                y_hat = log10(obj.aperiodic_function(p, x));

            end

            if ~strcmp(component_name, "aperiodic") % if 'periodic' or ''

                if nargin < 4
                    p = obj.periodic_fits;
                end
                y_hat = y_hat + obj.sum_of_gaussians_function_(p, x);

            end

            y_hat = 10.^y_hat;

        end
                
        
    end

    % GET methods
    methods

        function p0 = get.aperiodic_seeds(obj)

            idx = obj.current_row_idx_;
            if isempty(idx); p0 = []; return; end
            p0 = [obj.seeds.intercept(idx), obj.seeds.exponent(idx),...
                obj.seeds.knee(idx)];
            p0 = p0(~isnan(p0));

        end

        function p = get.aperiodic_fits(obj)

            idx = obj.current_row_idx_;
            if isempty(idx); p = []; return; end

            p = [obj.fits.intercept(idx), obj.fits.exponent(idx),...
                obj.fits.knee(idx)];
            p = p(~isnan(p));

        end


        function p = get.periodic_fits(obj)

            idx = obj.current_row_idx_;
            if isempty(idx); p = []; return; end

            p = [gen.make_column(obj.fits.amplitude{idx}), ...
                gen.make_column(obj.fits.center{idx}),...
                gen.make_column(obj.fits.bandwidth{idx})];

            p = [p(:); obj.fits.baseline(idx)];

        end

        function idx = get.current_row_idx_(obj)

            idx = find(obj.seeds.iter == obj.iter);

        end
        

        function conv = get.parameter_convergence(obj)

            if obj.iter

                idx = obj.current_row_idx_ + (-1:0);
                ap = obj.fits(idx,obj.free_ap_parameters);
                ap_conv = abs(ap(1:end-1,:)-ap(2:end,:))./abs(ap(1:end-1,:));
                p = obj.fits(idx,obj.free_p_parameters);
                [amp_conv, cf_conv, bw_conv] = obj.calculate_periodic_convergence_(p);
                b_conv = diff(p.baseline)/abs(p.baseline(1));

                conv = [ap_conv, table(amp_conv, cf_conv, bw_conv, b_conv, VariableNames=obj.free_p_parameters)];

                if isnan(conv.knee), conv.knee = double(~all(isnan(ap.knee))); end
            else
                conv = [];
                return;
            end

        end

        function conv = get.gof_convergence(obj)

            if obj.iter
                
                idx = obj.current_row_idx_ + (-1:0);

                gofN = obj.gof(idx,3:end);

                conv = abs(gofN(2,:)-gofN(1,:))./abs(gofN(1,:));

            else
                conv = [];
            end


        end
        
        function y = get.intercept(obj)

            if isempty(obj.intercept_), y= []; return; end
            y = obj.intercept_(end);

        end

        function y = get.knee(obj)

            if isempty(obj.knee_), y= []; return; end
            y = obj.knee_(end);

        end

        function y = get.exponent(obj)

            if isempty(obj.exponent), y= []; return; end
            y = obj.exponent_(end);

        end

        function y = get.center_frequency(obj)

            if isempty(obj.center_frequency_), y= []; return; end
            y = obj.center_frequency_{end};

        end

        function y = get.amplitude(obj)

            if isempty(obj.amplitude_), y= []; return; end
            y = obj.amplitude_{end};

        end

        function y = get.bandwidth(obj)

            if isempty(obj.bandwidth_), y= []; return; end
            y = obj.bandwidth_{end};

        end

        function y = get.baseline(obj)

            if isempty(obj.baseline_), y= []; return; end
            y = obj.baseline_{end};

        end

        function y = get.isParametersConverged(obj)

            if ~obj.iter; y = []; return; end
            y = obj.parameter_convergence < obj.convergence_tolerance_fits;

        end

        function y = get.isGOFConverged(obj)
            if ~obj.iter; y = []; return; end
            y = obj.gof_convergence < obj.convergence_tolerance_gof;

        end

    end


    methods (Access = protected)
       
        function estimates = estimate_aperiodic_params_from_data(obj, x, y)

            x(x==0) = [];
            isInitialFreqs = x < 1;
            
            if sum(isInitialFreqs) == 0
                interceptN = mean(log10(y)) + 2*3.33*std(log10(y));
                

            else
                
                interceptN = max(log10(y(isInitialFreqs)));

            end

            mdl_for_exp = fitlm(log10(x),log10(y));
            exponentN = -mdl_for_exp.Coefficients.Estimate(2);
            interceptN = 10.^ interceptN;

            % diffs = diff(y);           

            kneeN = 0;
            estimates = [interceptN, exponentN,  kneeN];

            if kneeN < 3

                estimates(end) = [];

            end
            estimates_in = num2cell(estimates);
            obj.update('seeds', 'aperiodic', estimates_in{:});

        end
        
        function aperiodic_params = perform_aperiodic_fit_(obj, freqs_in, spectrum_in_linear, is_robust)

            initial_guesses_ap = obj.aperiodic_seeds;            
            
            spectrum_to_fit = max(spectrum_in_linear, eps);
            lower_bounds_ap = [0, 0, 0]; 
            upper_bounds_ap = [max(spectrum_in_linear), 100, max(freqs_in)];
            if numel(initial_guesses_ap) == 3 && (initial_guesses_ap(3) < 1 || initial_guesses_ap(3) > max(freqs_in)) % no knee

                obj_func = @obj.aperiodic_residual_function;
                initial_guesses_ap(3) = [];
                lower_bounds_ap(3) = [];
                upper_bounds_ap(3) = [];

                obj.seeds.knee(obj.current_row_idx_) = NaN;                

            else

                obj_func = @obj.aperiodic_residual_function_w_knee;

            end           

            isConverged = false;
            n_iter = 0;
            if is_robust
                [aperiodic_params, n_iter, isConverged] = obj.robust_nonlinear_fit_(obj.max_irls_iterations, obj_func, freqs_in, spectrum_to_fit, initial_guesses_ap, lower_bounds_ap, upper_bounds_ap, obj.irls_tolerance);
            else                
                aperiodic_params = lsqcurvefit(obj_func, initial_guesses_ap, freqs_in, spectrum_to_fit, lower_bounds_ap, upper_bounds_ap, obj.lsq_options);
            end

            aperiodic_params_in = num2cell(aperiodic_params);
            obj.update('fits','aperiodic',n_iter, isConverged, aperiodic_params_in{:});
            
        end     
     

        function [fitted_params, bics] = perform_periodic_fit_(obj, init_params, freqs_in, initial_flattened_spectrum)
            
            [init_params, b_est] = obj.rearrange_periodic_params(init_params);
            lower_bounds = [eps, obj.peak_freq_range(1), eps];
            lower_b = min(initial_flattened_spectrum)-2*std(initial_flattened_spectrum);
            upper_bounds = [max(initial_flattened_spectrum)+std(initial_flattened_spectrum), ... 
                obj.peak_freq_range(2),  max(init_params(:,3)) + 2*std(init_params(:,3))];
            upper_b = mean(initial_flattened_spectrum)+2*std(initial_flattened_spectrum);
            
            n_peaks = size(init_params,1);


            residuals = initial_flattened_spectrum;
            bic_prev = obj.calculate_aic_(residuals, 4);
            bics = nan(1,n_peaks+1);
            bics(1) = bic_prev;
            isPeakIn = false(1, n_peaks);
            params = sortrows(init_params,1,"descend");
            

            options = optimoptions(@lsqnonlin, 'SpecifyObjectiveGradient', true);
            for ii = 1:n_peaks               
                

                isPeakIn(ii) = true;
                pN = params(isPeakIn,:);
                pN = [pN(:); b_est];
                % make baselines additive in the order from the minimum to the
                % maximum

                lbN = repmat(lower_bounds,sum(isPeakIn),1);
                lbN = [lbN(:);lower_b];
                ubN = repmat(upper_bounds,sum(isPeakIn),1);
                ubN =[ubN(:);upper_b]';
                
                % Nonlinear fit with the residual function with weights
                fitted_params = lsqnonlin(@(p) obj.periodic_residual_function(p, freqs_in, residuals), pN, lbN, ubN, options);
                % fitted_params = lsqcurvefit(@obj.sum_of_gaussians_function_, pN, freqs_in, residuals, lbN, ubN)
                y_pred = obj.sum_of_gaussians_function_(fitted_params, freqs_in);
                resN = residuals-y_pred;
                
                bicN = obj.calculate_bic_(resN, numel(pN));
                % aicN = obj.calculate_r2(residuals, y_pred, numel(pN));
                bics(ii+1) = bicN;
                if ~(bicN < bic_prev && abs(bicN-bic_prev)/abs(bic_prev) > .01)

                    isPeakIn(ii) = false;
                
                else

                    [params(isPeakIn,:), b_est] = obj.rearrange_periodic_params(fitted_params);%reshape(fitted_params(1:end-1), [3,sum(isPeakIn)])';
                    % b_est = fitted_params(end);
                    bic_prev = bicN;
                
                end          
                              

            end

            fitted_params = params(isPeakIn,:);

            obj.update('fits', 'periodic', bics(isPeakIn), fitted_params(:,1), fitted_params(:,2), fitted_params(:,3), b_est);
            fitted_params = [fitted_params(:); b_est];
        end

        function init_params = estimate_periodic_params_from_data(obj, freqs_in, flattened_spectrum)

            isFreqInRange = gen.ifwithin(freqs_in, obj.peak_freq_range);
            
            [init_amp, init_cf, init_bw, init_p] = obj.find_peaks_(freqs_in(isFreqInRange), flattened_spectrum(isFreqInRange));
            init_b = mean(init_amp - init_p); % baseline

            init_amp = init_p; % baseline corrected amplitude
            init_sigma = init_bw / (2*sqrt(2*log(2))); % bw to sigma
            init_params = [init_amp, init_cf, init_sigma];
            init_params = sortrows(init_params, 1, "descend");
            
            init_params = [init_params(:); init_b];  

        end

        function [amp,cf,bw,p] = find_peaks_(obj, freqs_in, flattened_spectrum)

            
            % Linear interp if freqs contain breaks
            if ~isscalar(unique(diff(freqs_in)))

                [flattened_spectrum, freqs_in] = obj.interpolate_breaks_(flattened_spectrum, freqs_in);

            end
            fq_res = mode(diff(freqs_in));
            n_med_bins = floor(1/fq_res);
            flattened_spectrum = medfilt1(flattened_spectrum, n_med_bins); 
                        
            [amp, cf, bw,p] = findpeaks(flattened_spectrum, freqs_in, ...
                MinPeakDistance = obj.peak_proximity_threshold, ...
                MinPeakProminence=2*1.4826*mad(flattened_spectrum),...
                MinPeakWidth = obj.min_peak_width,...
                NPeaks=obj.max_n_peaks);


        end

        function [ap_conv, p_conv] = calculate_parameter_convergence(obj)

            prev_ap_vals = [obj.center_frequency_(end-1), obj.amplitude_(end-1), obj.knee_(end-1)];
            curr_ap_vals = [obj.center_frequency, obj.amplitude, obj.knee];

            ap_conv = abs(curr_ap_vals - prev_ap_vals) ./ (abs(prev_ap_vals) + eps);
            
            curr_peaks_list = [obj.center_frequency{:}; obj.amplitude{:}; obj.bandwidth{:}; obj.baseline{:}];
            curr_n_peaks = size(curr_peaks_list,2);

            prev_peaks_list = [obj.center_frequency_{end-1}; obj.amplitude_{end-1}; obj.bandwidth_{end-1}; obj.baseline_{end-1}];
            prev_n_peaks = size(prev_peaks_list,2);
            
            p_conv = nan(1,size(curr_peaks_list,1)); 
            if curr_n_peaks == prev_n_peaks && ~isempty(curr_peaks_list)
                if ~isempty(prev_peaks_list) 
                    prev_cf_vals = prev_peaks_list(1,:); 
                    [~, sort_idx_prev] = sort(prev_cf_vals);
                    prev_params_mat= prev_peaks_list(:, sort_idx_prev);
                else
                    prev_params_mat = {}; 
                end
                curr_cf_vals(1,:) = curr_peaks_list(1,:);
                [~, sort_idx_curr] = sort(curr_cf_vals);
                curr_params_mat = curr_peaks_list(:, sort_idx_curr);
                                                
                if ~isempty(prev_params_mat) && ~isempty(curr_params_mat) 
                    rel_changes = abs(curr_params_mat - prev_params_mat) ./ (abs(prev_params_mat) + eps);
                    p_conv = rel_changes(:);                
                end
                
            end

        end

        function conv = calculate_gof_convergence(obj, r2, aic, bic)

            if isempty(obj.gof_convergence_); conv = []; return; end

            obj.gof_convergence_(end+1,:) = [r2, aic, bic];
            gof = obj.gof_convergence_(end,:);
            prev_gof = obj.gof_convergence_(end-1,:);
            conv = abs(gof - prev_gof) ./ abs(prev_gof);

        end
        
        

    end

    methods (Static)

        
        function [fitted_params, iter, isConverged] = robust_nonlinear_fit_(max_iter, obj_fun, x_data, y_data, initial_guesses, lower_bounds, upper_bounds, conv_tol)

            if nargin <= 7
                conv_tol = 1e-28;
                warning("IRLS convergence tolerance is set to: %.2f", conv_tol)
            end
            y_data = y_data(:);% make column

            min_excld_freq = 1; % before this value, weights remain intact
                       
            isAboveMin = x_data > min_excld_freq;

            % res_scalar = 1.4826;
            current_params = initial_guesses;
            weights = 1./(x_data).^current_params(2);%ones(size(x_data));%ones(size(x_data));%1./x_data;%ones(size(y_data)); % weeights start equal
            isExcld = false(size(x_data));
            options = optimoptions(@lsqnonlin, 'SpecifyObjectiveGradient', true);            
            for iter = 1:max_iter % start iterations
                
                weights =  ones(size(x_data));%1./(x_data).^current_params(2);
                weights(isExcld) = 0;
                last_params = current_params;
                % Nonlinear fit with the residual function with weights
                current_params = lsqnonlin(@(p) obj_fun(p, x_data, y_data, weights), last_params, lower_bounds, upper_bounds, options);
                % actual residuals without weights
                residuals = obj_fun(current_params, x_data, y_data);
                mad_resid = mad(residuals);
                if mad_resid < eps, mad_resid = eps; end 
                residuals(isAboveMin) = gen.robust_z(residuals(isAboveMin));

                % residuals = residuals / (res_scalar * mad_resid); %scale residuals
                weights_to_zero = abs(residuals) > 4; %(1 - residuals.^2).^2; % recalculate weights based on residuals
                isExcld = weights_to_zero >= 1 & isAboveMin;
                % check for convergence between the first and last step
                isConverged = sum((current_params - last_params).^2) / (sum(last_params.^2) + eps) < conv_tol;
                if isConverged; break; end
            end
            fitted_params = current_params;

        end

        function varargout = aperiodic_function(estimates, freqs)

            varargout = cell(1, nargout);
            if numel(estimates) > 2
                [varargout{:}]= FOOOFer.aperiodic_function_w_knee(estimates,freqs);
                return;
            end
            estimates = num2cell(estimates);
            [interceptN, exponentN] = deal(estimates{:});
            
            % model in log_scale, result in linear
            varargout{1} = 10 .^ (log10(interceptN) - log10(freqs.^exponentN));

            if nargout == 2
                % Jacobian matrix
                varargout{2} = [1/(log(10)*interceptN)*ones(size(freqs)),...
                    -(freqs.^exponentN .* log(freqs))./(log(10).*(freqs.^exponentN))...
                    ];
            end

        end

        function varargout = aperiodic_function_w_knee(estimates, freqs)

            varargout = cell(1, nargout);
            if numel(estimates) < 3
                [varargout{:}] = FOOOFer.aperiodic_function(estimates,freqs);
                return;
            end
            estimates = num2cell(estimates);
            [interceptN, exponentN, kneeN] = deal(estimates{:});
            
            % model in log_scale, result in linear
            varargout{1} = 10 .^ (log10(interceptN) - log10(kneeN + freqs.^exponentN));

            if nargout == 2
                % Jacobian matrix
                varargout{2} = [1/(log(10)*interceptN)*ones(size(freqs)),...
                    -1./(log(10)*(kneeN+freqs.^exponentN)),...
                    -(freqs.^exponentN .* log(freqs))./(log(10).*(kneeN+freqs.^exponentN))...
                    ];
            end

        end

        function [residuals, J] = aperiodic_residual_function(estimates, freqs, y_data, weights)
            
            if nargin < 4; weights = ones(size(freqs)); end
            [y_hat, J] = FOOOFer.aperiodic_function(estimates, freqs);
            residuals = weights .* (log10(y_data) - log10(y_hat));

            % Jacobian matrix, first partial derivatives
            J = -J.* weights;
            

        end

        function [residuals, J] = aperiodic_residual_function_w_knee(estimates, freqs, y_data, weights)
            
            if nargin < 4; weights = ones(size(freqs)); end
            [y_hat, J] = FOOOFer.aperiodic_function_w_knee(estimates, freqs);
            residuals = weights .* (log10(y_data) - log10(y_hat));

            % Jacobian matrix, first partial derivatives
            J = -J.*weights;
            

        end

        

        function varargout = sum_of_gaussians_function_(p, x)
            
            % n_gauss = size(p,1);
            n_gauss = (numel(p)-1)/3;
            J = zeros(numel(x), numel(p));
            % b = p(end);
            [p, b] = FOOOFer.rearrange_periodic_params(p);%reshape(p(1:end-1), [3,n_gauss])';
            y_hat = zeros(size(x));
            

            for ii = 1:n_gauss

                [yN, JN] = FOOOFer.gaussian_function_(p(ii,:), x);    
                y_hat = y_hat + yN;
                J(:,(ii-1)*size(p,2) + (1:size(p,2))) = JN;

            end

            J(:,end) = ones(size(x)); % partial derivative w.r.t baseline 
            varargout{1} = y_hat + b;
            if nargout > 1
                varargout{2} = J;
            end

        end

        function [res, J] = periodic_residual_function(p, x, y)
        
            [y_hat, J] = FOOOFer.sum_of_gaussians_function_(p, x);

            [p, b] = FOOOFer.rearrange_periodic_params(p);

            % Penalties
            peak_locs = p(:,2);
            peak_dist = peak_locs - peak_locs';
            peak_dist = peak_dist(triu(peak_dist)>0);
            peak_prox_penalty = sum(1./peak_dist.^2); % penalize when peaks are too close together
            baseline_penalty  = b^2; % penalize when baseline is farther from 0
            peak_sd = p(:,3);
            q1 = norminv(.15, peak_locs, peak_sd);
            q3 = norminv(.85, peak_locs, peak_sd);
            sig_points = repmat(x,1,numel(q1))'>q1 & repmat(x,1,numel(q1))'<q3;
            undersampling_penalty = sum((sum(sig_points,2) - (q3-q1)/mode(diff(x))).^2);

            res = (y - y_hat) + (peak_prox_penalty + baseline_penalty + undersampling_penalty) / numel(y);

        end

        function [y_model, J] = gaussian_function_(p, x)
            % p = [amp, CF, bw]
            cf = p(2); amp = p(1); sd = p(3);% b = p(4);
            y_model = amp * exp(-(x - cf).^2 / (2 * sd.^2)); 

            % Jacobian
            J = [...
                exp(-(x - cf).^2 / (2 * sd.^2)), ...
                (amp .* (x - cf) ./ sd^2) .* exp(-(x - cf).^2 / (2 * sd.^2)),...
                (amp .* (x - cf).^2 ./ sd^3) .* exp(-(x - cf).^2 / (2 * sd.^2)) ...
                ];


        end

        function aic = calculate_aic_(residuals, n_param)

            n_sample = numel(residuals);
            rss = sum(residuals.^2);
            log_lik = (-n_sample/2) * (log(2*pi*rss/n_sample) + 1);
            aic = 2*n_param - 2*log(log_lik);
            if n_sample / n_param < 40
                % correction for low sample sizes
                aic = aic + (2*n_param*(n_param + 1)) / (n_sample - n_param + 1);
                
            end
        end

        function bic = calculate_bic_(residuals, n_param)

            n_sample = numel(residuals);
            bic = log(n_sample) * n_param + n_sample .* log(sum(residuals.^2) ./ n_sample);

        end

        function r2 = calculate_r2(y, y_hat, n_param)

            n_sample = numel(y);
            residuals = y - y_hat;
            ss_res = sum(residuals.^2);
            ss_total = sum((y(:) - mean(y(:))).^2);
            if ss_total < 1e-12; r2 = 1; else; r2 = 1 - (ss_res / ss_total); end

            % adjust according to no of params
            r2 = 1 - (((1 - r2) * (n_sample - 1)) / (n_sample - n_param - 1));

        end

        function [y_interp, x_interp] = interpolate_breaks_(y, x)

            step = mode(diff(x));

            x_interp = (x(1):step:x(end))';
            y_interp = interp1(x, y, x_interp, 'linear');
            

        end

        function [R2, AIC, BIC] = calculate_gof(y, y_hat, n_param)

            residuals = y - y_hat;
            R2 = FOOOFer.calculate_r2(y, y_hat, n_param);
            AIC = FOOOFer.calculate_aic_(residuals, n_param);
            BIC = FOOOFer.calculate_bic_(residuals, n_param);

        end

        function [p_mat, b] = rearrange_periodic_params(p)
            
            b = p(end);
            p(end) = [];
            n_peaks = numel(p)/3;

            p_mat = reshape(p, [n_peaks, 3]);

        end

        function [amp_conv, cf_conv, bw_conv] = calculate_periodic_convergence_(p)

            p0 = p(1,:);
            p1 = p(2,:);
            cf0 = p0.center{:};         
            amp0 = p0.amplitude{:};
            bw0 = p0.bandwidth{:};
            b0 = p0.baseline;
            n_peak0 = numel(cf0);
            
            cf1 = p1.center{:};         
            amp1 = p1.amplitude{:};
            bw1 = p1.bandwidth{:};
            b1 = p1.baseline;
            n_peak1 = numel(cf1);

            d_cf = (cf0-cf1').^2;
            [n_min, dim_min] = min([n_peak0, n_peak1]);


            [d_min, d_idx] = min(d_cf, [], dim_min);
            idx_selector = arrayfun(@(ii) {ii, d_idx(ii)},1:n_min, 'UniformOutput', false);
            idx_selector = cat(1,idx_selector{:});
            if dim_min == 2
                idx_selector = idx_selector';
                cf_base = abs(cf1);
                amp_base = abs(amp1);
                bw_base = abs(bw1);
            else
                cf_base = abs(cf0);
                amp_base = abs(amp0);
                bw_base = abs(bw0);
            end           

            % if numel(unique(d_idx)) == numel(d_idx)    
            try

                calc_conv = @(d, p) arrayfun(@(ii) d(idx_selector{ii,:})/p(ii), 1:n_min);

                cf_conv = calc_conv(d_cf, cf_base);
                amp_conv = calc_conv((amp0 - amp1').^2, amp_base);
                bw_conv = calc_conv((bw0 - bw1').^2, bw_base);

                n_diff = abs(n_peak1 - n_peak0);
                if n_diff

                    cf_conv = [cf_conv,ones(1, n_diff)];
                    amp_conv = [amp_conv, ones(1, n_diff)];
                    bw_conv = [bw_conv, ones(1, n_diff)];

                end

                cf_conv = mean(cf_conv);
                amp_conv = mean(amp_conv);
                bw_conv = mean(bw_conv);

            % else
            catch e

                %Idk yet
                aaaa

            end



        end
        

    end

end