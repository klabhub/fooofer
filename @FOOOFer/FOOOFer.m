classdef FOOOFer < handle

    % O1OFFitter Class for parameterizing neural power spectra.
    % Implements a FOOOF-like algorithm with multi-pass peak detection.
    % Peak amplitudes are fitted in log-space internally but stored and outputted in linear space.

    properties

        % Peak Fit Settings               
        max_refit_n_iter = 10
        min_peak_width (1,1) double = 1        
        max_n_peaks (1,1) double = 5              % Maximum number of peaks to fit per pass        
        min_peak_distance (1,1) double = 1.0 % Minimum Hz separation to keep distinct peaks
        min_peak_frequency (1,1) double
        max_peak_frequency (1,1) double        
        excluded_frequencies (:,2) double = []
                
        results = struct( ...            
            iter = [], type = [], seed = [], ...
            fit = [], fit_flag = [], ...
            fit_n_iter = [], fit_n_func_eval = [], fit_n_pcg_iter = [],...
            gof = [], gof_metric = [], modelSurvived = [])

        verbose = true;

        % fmincon options
        max_func_eval = 50
        max_fit_iter = 50;

    end

    properties (SetAccess = protected)

        iter = 0

    end
  
    methods
        
         [results, ap_fitter, p_fitter] = fit(obj, freqs, spectrum, include_freq_range, exclude_freq_range, apriori_peak_range)
        
    end
    
    methods

        function obj = FOOOFer(varargin)
            % Constructor for O1OFFitter
            
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
       
        function next(obj)

            if obj.iter >= obj.max_refit_n_iter
                return;

            end
            obj.iter = obj.iter + 1;
            % update tables

        end        
       
        function append_to_results(obj, pv)

            arguments

                obj

                pv.iter = []
                pv.type = []
                pv.seed = []
                pv.fit  = []
                pv.fit_flag = []
                pv.fit_n_iter = []
                pv.fit_n_func_eval = []
                pv.fit_n_pcg_iter = []
                pv.gof = []
                pv.gof_metric = []
                pv.modelSurvived = []

                pv.nextEntry = false  
                pv.step_from_current_entry = 0;
                
            end

            fld_names = fieldnames(pv);

            n_fld = numel(fld_names);
            allowed_fieldnames = fieldnames(obj.results);
            n_rows = numel(obj.results);

            % if nextEntry=true, append the next entry unless it is the first entry
            if pv.nextEntry & ~isempty(obj.results(1).iter)
                iRow = n_rows + 1;
            else
                iRow = n_rows + pv.step_from_current_entry;
            end

            for ii = 1:n_fld

                fldN = fld_names{ii};
                valN = pv.(fldN);
                if ~ismember(fldN, allowed_fieldnames) || isempty(valN) || all(ismissing(valN))
                    continue;
                end
                
                obj.results(iRow).(fldN) = valN;

            end



        end
                
        
    end


   methods (Static)
        function aic = calculate_aic_(residuals, n_param)
            
            residuals = residuals(:);
            n_sample = numel(residuals);
            rss = sum(residuals.^2);
            log_lik = -n_sample / 2 * (log(2*pi)+log(rss/n_sample) + 1);
            aic = 2*n_param - 2*log_lik;
            if n_sample / n_param < 40
                % correction for low sample sizes
                aic = aic + (2*n_param*(n_param + 1)) / (n_sample - n_param - 1);
                
            end
        end

        function bic = calculate_bic_(residuals, n_param)

            residuals = residuals(:);
            n_sample = numel(residuals);
            bic = log(n_sample) * n_param + n_sample .* log(sum(residuals.^2) ./ n_sample);

        end

        function r2 = calculate_r2(y, y_hat, n_param)

            n_sample = numel(y);
            residuals = y - y_hat;
            residuals = residuals(:);
            ss_res = sum(residuals.^2);
            ss_total = sum((y(:) - mean(y(:))).^2);
            if ss_total < 1e-12; r2 = 1; else; r2 = 1 - (ss_res / ss_total); end

            % adjust according to no of params
            r2 = 1 - (((1 - r2) * (n_sample - 1)) / (n_sample - n_param - 1));

        end

        

        function [R2, AIC, BIC] = calculate_gof(y, y_hat, n_param)

            residuals = y - y_hat;
            R2 = FOOOFer.calculate_r2(y, y_hat, n_param);
            AIC = FOOOFer.calculate_aic_(residuals, n_param);
            BIC = FOOOFer.calculate_bic_(residuals, n_param);

        end
      

    end

end