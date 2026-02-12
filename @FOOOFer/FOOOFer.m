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
        min_peak_frequency (1,1) double = NaN
        max_peak_frequency (1,1) double   = NaN    
        excluded_frequencies (:,2) double = []
                
        results = struct( ...            
            iter = [], type = [], seed = [], ...
            fit = [], fit_flag = [], ...
            fit_n_iter = [], fit_n_func_eval = [], fit_n_pcg_iter = [],...
            fit_dur = [], gof = [], gof_metric = [],...
            lrt_comparison_row = [], chi2 = [], p = [], pseudo_r2 = [], ...
            modelSurvived = [])

        performance_ = {};
        best_iter

        verbose = true;

        % fmincon options
        max_func_eval = 50
        max_fit_iter = 50;
        max_subproblem_iter = 100


    end

    properties (SetAccess = protected)

        iter = 0

    end

    properties (Dependent)
        performance_results
    end

    properties (Dependent, Access = protected)

        n_results_row
        
    end
  
    methods
        
         [performance, results, ap_fitter, p_fitter] = fit(obj, freqs, spectrum, include_freq_range, exclude_freq_range, apriori_peak_range)
        
    end

    methods (Access=protected)
        
        [fitter, res] = aperiodic_fit_(self, fitter, pv)
        [fitter, res] = periodic_fit_(self, fitter, pv)
               
    end

    methods (Static, Access = private)

        varargout = do_lrt_(aic0, aic1, n_params0, n_params1, n)

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
                pv.fit_dur = []
                pv.gof = []
                pv.gof_metric = []
                pv.lrt_comparison_row = [] % row index in the results 
                pv.lrt_comparison_row_rel = 0 % relative row index
                % struct to which model, current model is being compared to
                pv.chi2 = [] % chi-square stats
                pv.p = [] % p value
                pv.pseudo_r2 = [] % chi stat effect size (Cox & Snell Approach)
                pv.modelSurvived = []

                pv.nextEntry = false  
                pv.row_index = numel(obj.results)
                pv.step_from_current_entry = 0;
                
            end

            fld_names = fieldnames(pv);

            n_fld = numel(fld_names);
            allowed_fieldnames = fieldnames(obj.results);
            n_rows = pv.row_index;            

            % if nextEntry=true, append the next entry unless it is the first entry
            if pv.nextEntry & ~isempty(obj.results(1).iter)
                iRow = n_rows + 1;
            else
                iRow = n_rows + pv.step_from_current_entry;
            end

            if pv.lrt_comparison_row_rel
                pv.lrt_comparison_row = iRow+ pv.lrt_comparison_row_rel;
            end

            for ii = 1:n_fld

                fldN = fld_names{ii};
                valN = pv.(fldN);
                if ~ismember(fldN, allowed_fieldnames) || isempty(valN) || all(ismissing(valN))
                    continue;
                elseif isgpuarray(valN)
                    valN = gather(valN);
                end

                obj.results(iRow).(fldN) = valN;

            end

        end

        function res = retrieve(self, selection, restrictions)
            % retrieves results that meet a criteria from the results struct
            % Examples:
            %   res = self.retrieve()
            %       will retrieve last periodic and aperiodic models survived
            %   res = self.retrieve(indices) % retrieve self.results(indices)
            %   res = self.retrieve(field_name) returns specific fields in the struct
            %   res = self.retrieve(selection_mode) returns results struct
            %       that meets specific criteria:
            %       - 'best': returns best fits of each iteration (unless
            %              restrictions are specified). It includes surviving periodic
            %              model fit results, surviving aperiodic model fit
            %              results, the best alternative aperiodic model's,
            %              that is, the results row that is indexed by
            %              results.lrt_model_comparison_row of the surviving
            %              aperiodic model, if any.
            %               Criteria for best fit: modelSurvived = true,
            %               except for the alternative aperiodic model. If no                   
            %       - 'survived': returns all the survived models
            %               (modelSurvived = true) from each iteration unless
            %               other restrictions were specified.
            %   res = self.retrieve(iter = 2, type = 'aperiodic', fit_flag = 1)
            %       will return all rows whose fields meet the specified
            %       criteria
            %   selection arguments and name-value arguments could be used in
            %   conjunction to restrict results

            % retrieves results that meet a criteria from the results struct
            % Arguments (Input)
            %   self
            %   selection       % Repeating arguments: row indices (integer), field names (text), or selection modes (text)
            %   restrictions    % name-value arguments struct
            % Arguments (Output)
            %   res struct

            arguments (Input)
                self
            end

            arguments (Input, Repeating)
                selection
            end
            
            arguments (Input)
                restrictions.iter = []
                restrictions.type = []
                restrictions.seed = []
                restrictions.fit_flag = []
                restrictions.modelSurvived = []
                % Add other fields from self.results here if they need to be filterable
            end

            arguments (Output)
                res struct
            end

            % 1. Parse flexible repeating selection arguments using helper
            n_rows = numel(self.results);
            % Pass fieldnames(self.results) so the helper validates against all available fields
            parsed_selection = modifyRetrieveSelectionArgs_(selection, n_rows, fieldnames(self.results));

            req_fields = parsed_selection.fields;
            selection_mode = parsed_selection.mode;

            % Initialize row indices based on parsing results
            % parsed_selection.indices now contains either all rows or the specific intersection
            row_indices = parsed_selection.indices;

            % Initial subset extraction
            res_struct = self.results(row_indices);

            % 2. Validate and Modify Restrictions based on Mode
            % This handles negative iterations and mode-specific constraints (e.g., survived=true)
            restrictions = modifyRetrieveRestrictions_(res_struct, restrictions, selection_mode);

            % Convert to table for easier column-wise operations
            res_table = struct2table(res_struct, 'AsArray', true);

            % Store original indices if needed for 'best' mode logic (LRT cross-referencing)
            % We map the current table rows back to the global self.results indices
            global_indices = row_indices(:);

            % 3. Apply restrictions by variable
            % We iterate only over the specified restriction fields (variables), not rows.
            restriction_fields = fieldnames(restrictions);
            mask = true(height(res_table), 1);

            for i = 1:numel(restriction_fields)
                fn = restriction_fields{i};
                val = restrictions.(fn);

                col_data = res_table.(fn);

                % Handle filtering based on column type (Cell vs Array)
                % This avoids explicit loops over rows
                if iscell(col_data)
                    if isnumeric(val) || islogical(val)
                        % Numeric/Logical match in cell array (handling empties)
                        is_match = cellfun(@(x) ~isempty(x) && any(ismember(x, val)), col_data);
                    else
                        % String/Char match in cell array
                        is_match = cellfun(@(x) ~isempty(x) && any(matches(string(x), string(val))), col_data);
                    end
                else
                    % Standard array column
                    if isnumeric(val) || islogical(val)
                        is_match = ismember(col_data, val);
                    else
                        is_match = matches(string(col_data), string(val));
                    end
                end
                mask = mask & is_match;

            end

            % 4. Apply special logic for 'best' mode
            % 'best' includes survivors AND the specific aperiodic models they were compared against (LRT)
            if strcmpi(selection_mode, 'best') || strcmpi(selection_mode, 'estimates')
                % First, apply the standard mask to find the 'surviving' candidates within restrictions
                % Note: modifyRetrieveRestrictions_ does NOT enforce modelSurvived=true for 'best',
                % so we check it here.

                % Check survival status column
                if iscell(res_table.modelSurvived)
                    survived_col = cellfun(@(x) ~isempty(x) && x==true, res_table.modelSurvived);
                else
                    survived_col = res_table.modelSurvived == true;
                end

                % Candidates are those that pass restrictions AND survived
                survivor_mask = mask & survived_col;

                % Now find the 'alternatives' (LRT comparison rows) for these survivors
                % 1. Get types of survivors
                types = string(res_table.type);
                is_aperiodic = matches(types, "aperiodic", 'IgnoreCase', true);

                % 2. Filter for aperiodic survivors
                aperiodic_survivor_mask = survivor_mask & is_aperiodic;

                % 3. Extract the 'lrt_comparison_row' values from these rows
                lrt_targets = [];
                if any(aperiodic_survivor_mask)
                    raw_lrt = res_table.lrt_comparison_row(aperiodic_survivor_mask);
                    if iscell(raw_lrt)
                        lrt_targets = [raw_lrt{:}];
                    else
                        lrt_targets = raw_lrt;
                    end
                    lrt_targets = unique(lrt_targets);
                end

                % 4. Update the final mask
                % We keep a row if:
                %   (It is a Survivor AND passes restrictions)
                %   OR
                %   (It is one of the targeted LRT rows AND passes restrictions?)
                %   *Usually strict Best logic implies we just grab the LRT row regardless,
                %    but here we ensure it exists in our current selection set.*

                is_lrt_target = ismember(global_indices, lrt_targets);

                % Final mask for 'best': Survivors + their specific alternatives
                mask = survivor_mask | is_lrt_target;

                if strcmpi(selection_mode, 'estimates')
                    % retrieves best final estimates for
                    % 1. aperiodic without knee
                    % 2. aperiodic with knee
                    % 3. periodic
                    n_params = cellfun(@(x) numel(x), res_table.fit);
                    hasKnee = n_params == 4;
                    noKnee = n_params == 3;
                    typeAp = strcmp(res_table.type,"aperiodic");
                    typeP = strcmp(res_table.type,"periodic");
                    
                    if ~isfield(restrictions,'iter')
                        bestIter = res_table.iter == self.best_iter;
                    else
                        bestIter = true(size(hasKnee));
                    end

                    bestApKnee = find(mask & typeAp & hasKnee & bestIter,1, 'last');
                    bestAp = find(mask & typeAp & noKnee & bestIter,1, 'last');
                    if isempty(bestApKnee)
                        % release iter restriction
                        bestApKnee = find(mask & typeAp & hasKnee,1, 'last');
                    end

                    if isempty(bestAp)
                        bestAp = find(mask & typeAp & noKnee,1, 'last');
                    end
                    bestP = find(mask & typeP,1, 'last');
                    mask = false(size(mask));
                    mask([bestAp, bestApKnee,bestP]) = true;

                end
            end

            % Apply the final mask to the table
            res_table = res_table(mask, :);

            % 5. Apply selection by exporting only those columns specified by parsed_selection.fields
            % Ensure we only ask for fields that exist in the table
            valid_fields = intersect(res_table.Properties.VariableNames, req_fields,'stable');
            res_table = res_table(:, valid_fields);

            % Convert back to struct
            res = table2struct(res_table);

        end


    end

    %% Get Methods
    methods
        function n = get.n_results_row(self)
            n = numel(self.results);
        end

        function res = get.performance_results(self)

            res = struct2table([self.performance_{:}]);

        end

        function i = get.best_iter(self)

            res = self.performance_results;
            [~, idx] = min(res.aic);
            i = res.iter(idx);

        end
    end
    %% Static Methods
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

function restrictions = modifyRetrieveRestrictions_(res, restrictions, mode)

    n_iter = res(end).iter;
    % No need for n_rows in this specific logic, but passed in case needed later
    
    if ~isempty(restrictions.iter)
        isRelIter = restrictions.iter < 0;
        if any(isRelIter)
            % Check if n_iter is empty (e.g., empty results), prevent error
            if isempty(n_iter)
                error('retrieve:EmptyResults', 'Cannot use relative iterations on empty results.');
            end
            restrictions.iter(isRelIter) = n_iter + restrictions.iter(isRelIter);
        end
        mustBeNonnegative(restrictions.iter); %assert non-negativity
        mustBeInteger(restrictions.iter);
    end

    % Validates and modifies restrictions based on selection mode
    if strcmpi(mode, 'survived') 
        % Compatibility fix: Check if user provided restriction (arguments default to [])
        assert(isempty(restrictions.modelSurvived) || restrictions.modelSurvived, "If selection is 'survived', cannot restrict modelSurvived!")
        restrictions.modelSurvived = true;        
    elseif strcmpi(mode, 'best') || strcmp(mode, "estimates")
        % Compatibility fix: Check if user provided restriction
        assert(isempty(restrictions.modelSurvived), "If selection is 'best', cannot restrict modelSurvived!")
    end

    % remove empty fields from restrictions to avoid looping over them later
    fns = fieldnames(restrictions);
    for i = 1:numel(fns)
        if isempty(restrictions.(fns{i}))
            restrictions = rmfield(restrictions, fns{i});
        end
    end
end

function s = modifyRetrieveSelectionArgs_(selection, n_rows, results_fields)
    % Helper to parse repeating selection arguments
    % Returns struct with indices (default 1:n_rows or intersected selection), fields, and mode
    
    s.indices = 1:n_rows; % Default to all rows
    s.fields = results_fields;
    s.mode = 'all';
    s.final = 0;

    n_select = numel(selection);
    modes = ["best", "survived", "estimates", "all"];
    assert(n_select <= numel(modes), "You must only select indices, variables, mode and each of them once!");

    [selectedIndices, selectedFields, selectedMode] = deal(false);
    

    for i = 1:n_select
        item = selection{i};
        if isempty(item)
            continue;
        end
        
        if isnumeric(item) && ~selectedIndices
            % If multiple numeric arrays are provided, intersect them
            % Note: logic ensures we always respect the 1:n_rows boundary
            isRelIdx = item < 0;
            if any(isRelIdx)
                item(isRelIdx) = n_rows + item(isRelIdx) + 1;
            end
            mustBeInteger(item);
            mustBePositive(item); % assert positivity
            s.indices = intersect(s.indices, item);
            selectedIndices = true;           
        elseif ischar(item) || isstring(item) || iscell(item)
            %make string
            item = string(item);
            % Check if it is a mode or a field name
            if any(matches(item, modes, 'IgnoreCase', true)) && ~selectedMode
                assert(isscalar(item), "Provide only one selection mode!")
                s.mode = lower(item);
                selectedMode = true;
            elseif all(matches(item, results_fields)) && ~selectedFields
                s.fields = intersect(s.fields, cellstr(item));
                selectedFields = true;
            else
                error("Selection Error: selection does not exist or duplicate type provided!")
            end
        else
            error("Selection Error: Unknown type!")
        end
    end

end