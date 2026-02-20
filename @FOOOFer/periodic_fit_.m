function [fitter, res] = periodic_fit_(self, fitter, pv)

arguments

    self FOOOFer
    fitter NLLFitter

    pv.findPeaks = false % if true, data-driven peak finding algrithm is
    % used to determine initial parameter estimates.

    pv.p0 (1,:) double = [] % if provided uses these initial parameter
    % if findPeaks = true, it first attempts to match prior_p0 to empirical
    % peaks, then appends no-match peak parameters to test_p0    

    pv.apriori_fooofer FOOOFer = FOOOFer.empty() % if provided a FOOOFer object from a previous 
    % model fit (e.g., a neighbpring channel fit), the algorithm will use
    % the results to inform current fitting procedure by populating base_p0 
    % If findPeaks = true, the algorithm will match the peak parameters from 
    % the apriori model and the peak finding algorithm estimates based on
    % the center frequency. Then appends no-match peak parameters to test_p0

    pv.synch_tol = 2 % when synching apriori peaks with empirical peaks
    % periodic parameters from apriori_fit to data-driven estimates 
    % match the center frequencies that are within this tol

    pv.test_p0 (1,:) double = [] % if provided, tests these initial 
    % parameters in stepwise LRT    
    
    pv.lrt {validateLRTInp_(pv.lrt)} = 'none' % if true it performs LRT
    pv.reduced_model {validateReducedModel_(pv.reduced_model)} = struct([])
    pv.p_threshold (1,1) double {mustBePositive, mustBeLessThan(pv.p_threshold,1)} = .05

    pv.refineIter = false; % if true, modelSurvived = 1; pv.lrt must be none and findPeaks must be false

end
%% --- Input Validation ---
pv = configure_pv_(pv);

isLRT = ~isempty(pv.test_p0) || pv.findPeaks;

mdl = fitter.inner_model;

%% --- Retrieve initial guesses ---
%
% Scenario 1: no apriori_fooofer or p0 provided => completely data-driven
%   guesses.
% Scenario 2A: apriori_fooofer or p0 provided and findPeaks = true
%   match data-driven estimates to apriori_fooofer results
% Scenario 2B: apriori_fooofer or p0 provided and findPeaks = false
%   use the initial estimates as is
% Scenario 3: apriori_fooofer or p0 provided and pv.lrt=false
%   direct fit

% If requested, empirically search for peaks and estimate parameters
emp_p0 = [];
if pv.findPeaks
    % data-driven estimates
    mdl.n_peaks = self.max_n_peaks; % evokes reconstructing the model with maximum no of peaks
    emp_p0 = fitter.estimate();    
end
test_p0 = emp_p0;

% If provided, retrieve the user provided initial guesses
prior_p0 = retrieve_prior_p0_(pv.apriori_fooofer, pv.p0, pv.reduced_model);
if ~isempty(prior_p0) && ~isempty(emp_p0)

    [prior_peak_p0, prior_nonpeak_p0] = arrange_params_(prior_p0);
    [emp_p0, ~] = arrange_params_(emp_p0);

    % match initial guesses with prior
    [p0, test_p0] = mdl.synch_peak_parameters_( ...
        prior_peak_p0(:), emp_p0(:), pv.synch_tol);
    % append baseline and sigma
    if ~isempty(p0)
        p0 = [p0, prior_nonpeak_p0];
    end

else

    p0 = prior_p0;

end

% if there are test peaks, LRT is activated
isLRT = (isLRT || ~isempty(test_p0)) && ...
    ~(strcmp(pv.lrt, 'reduced') && isempty(test_p0));

%% --- Optimization ---
% --- Reconfigure if needed --
n_base_peaks = floor(numel(p0)/3);
n_test_peaks = floor(numel(test_p0)/3);

if isLRT

    if ~n_base_peaks && ~isstruct(pv.lrt) && strcmp(pv.lrt, 'reduced')
        warning("No reliable peaks matching the reduced/empirical priors " + ...
            "were detected in data. Switching to lrt = 'null'");
        pv.lrt = 'null';
    end

    if ~n_test_peaks && ~n_base_peaks
       
        assert(pv.findPeaks, "BUG in previous input validation!");
        warning("Nothing to test survived peak finding algorithm! " + ...
            "Returning the null model.");
        res = retrieve_null_model_(self);
        self.append_to_results(modelSurvived=true);
        return;
    end

else
    
    assert(~n_test_peaks && n_base_peaks, "BUG in previous input validation!")

end

if ~isLRT
    % Scenario 1
    % PROMPT: Check if this assertion could ever be violated.
    assert(~isempty(p0) && isempty(test_p0));

    % reconfigure the model
    mdl.n_peaks = n_base_peaks;
    % estimate parameter boundaries
    mdl.P = p0(1:end-1); % if max_frequency offset provided, this will constrain center_frequency search per peak
    [lb, ub] = fitter.estimate_parameter_bounds();
    % Set up optimizer options
    H_pattern = double(fitter.compute_hessian(p0) ~= 0); %speeds up memory alloc
    optimopts = optimoptions('fmincon',...
        'Display', 'off',...
        'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
        'SpecifyObjectiveGradient', true, ...
        'EnableFeasibilityMode', true,...
        'SubproblemAlgorithm', 'cg', ...
        'HessianFcn', @(p, ~) fitter.compute_hessian(p), ...
        'HessPattern', H_pattern, ...
        'MaxFunctionEvaluations', self.max_func_eval, ...
        'MaxIterations', self.max_fit_iter);

    % Fit the model
    onset = tic;
    if self.verbose
        
        fprintf('Fitting the periodic model with %d peaks.\n', n_base_peaks);
    end

    [params, ~, flag, op] = fmincon(@(p) fitter.objective_function(p), ...
        p0, [], [], [], [], lb, ub, [], optimopts);
    dur = toc(onset);
    if self.verbose, fprintf("\tTook %.2f seconds.\n", dur); end

    % gof
    aic1 = self.calculate_aic_(mdl.wR, numel(params));

    % Save the results

    res = struct(iter = self.iter,...
        type = 'periodic',...
        seed = p0, ...
        fit = params, ...
        fit_flag=flag, ...
        fit_n_iter = op.iterations, ...
        fit_n_func_eval = op.funcCount, ...
        fit_n_pcg_iter = op.cgiterations, ...
        fit_dur = dur,...
        gof=aic1, ...
        gof_metric='aic', ...
        nextEntry = true);
    res_args = namedargs2cell(res);
    self.append_to_results(res_args{:});

    if pv.refineIter || (~strcmp(pv.lrt, "none") && pv.findPeaks)
        % if a refinement iteration or if it was initially set to do LRT 
        % test but initial estimates were reliably matching all of the 
        % empirical estimates LRT test is skipped.
        self.append_to_results(modelSurvived=true);
    end

else
%%
   if ~strcmp(pv.lrt, 'reduced')
       % NULL model
       res = retrieve_null_model_(self, mdl);
   elseif isempty(p0) % findPeaks=false 
       res = pv.reduced_model;
   else
       res.fit = p0; % update apriori estimates
   end
%%
   % --- Stepwise LRT ---
   %
   % We will fit the model one step at a time and compare the new model
   % to the reduced model. Steps:
   % - If lrt = 'reduced' and ~isempty(p0), first iteration ii = 0 and
   %    first predict_fit_(...,p0 = p0); else, res must be non-empty, 
   %    ii=1 and first predict_fit_(...,p0 = first_test_peaks)
   % - lrt is only done when ii > 0

   ii = ~(strcmp(pv.lrt,'reduced') && ~isempty(p0));
   % params in each cell will be tested at each iteration
   test_param_order = cell(n_test_peaks + ~ii,1);   
   
   [test_peak_params, test_nonpeak_params] = arrange_params_(test_p0);
   test_param_order(1+~ii:end) = mat2cell(test_peak_params, ...
       ones(1, n_test_peaks), 3);
   if ~ii       
       test_param_order{1} = [];       
   else
       test_param_order{1} = [test_param_order{1}, test_nonpeak_params];
   end

   %%   
   skipFirstTest = ii == 0;
   last_surviving_model_index = self.n_results_row;
   for ii = ii:double(n_test_peaks)

       % determine the parameters to fit
       % use previous parameter fit if any, and append the testing
       % parameters, test_nonpeak_params will only be used if res.fit does
       % not contain baseline and sigma params
       pN = join_params_(res.fit, test_param_order{ii+skipFirstTest});

       if self.verbose
           fprintf('\n %d. ', ii+skipFirstTest);
       end
       % Fit the model
       [alt_fitter, alt_res] = self.periodic_fit_(fitter.copy(), ...
           p0=pN, lrt = 'none');
       % Test
       if ii % LRT only when the null model stats are stored in res
           % compare the reduced model to the alternative model
           [p, chi_stat, eff] = self.do_lrt_(res.gof, alt_res.gof, ...
               numel(res.fit), numel(alt_res.fit), mdl.sample_size);

           % alternative model survives if:
           % p < p_threshold
           % fit_flag is positive (success)
           % alt_aic < reduced_aic
           altModelSurvived = ...
               p < pv.p_threshold && res.fit_flag > 0  && alt_res.gof < res.gof;
           
           % update current model results
           self.append_to_results(chi2 = chi_stat, ...
               p=p, pseudo_r2=eff,...
               modelSurvived = altModelSurvived,...
               lrt_comparison_row = last_surviving_model_index)
           % update simple model results
           self.append_to_results( ...
               modelSurvived = ~altModelSurvived, ...
               row_index = last_surviving_model_index, ...
               lrt_comparison_row = self.n_results_row);

           if altModelSurvived               
               last_surviving_model_index = self.n_results_row;
           end

       elseif skipFirstTest
           last_surviving_model_index = self.n_results_row; % null model
       end

       % update the initial null model || the surviving model
       if ~ii || altModelSurvived 
           res = alt_res;
           fitter = alt_fitter;           
       end       

   end

end
end

%% LOCAL HELPER FUNCTIONS
function p0 = retrieve_prior_p0_(prior_fooofer, prior_p0, prior_results)

if ~isempty(prior_p0)
    p0 = prior_p0;
elseif ~isempty(prior_fooofer)
    p0 =[]; %retrieve func
elseif ~isempty(prior_results)
    p0 = prior_results.fit;
else
    p0 = [];
    return;
end

end

function res = retrieve_null_model_(self,mdl)

res = struct(iter = self.iter, ...
    type = "periodic",...
    fit= mdl.null_model.P,...
    fit_flag = 1,...
    gof = mdl.null_model.aic,...
    gof_metric='aic',...
    nextEntry=true);
res_cfg = namedargs2cell(res);
self.append_to_results(res_cfg{:});

end

function [peak_params, nonpeak_params] = arrange_params_(p0)

nonpeak_params = [];
peak_params = [];
if numel(p0) < 3, return;
elseif numel(p0)==3, peak_params = p0; return
end

n_nonpeak_params = mod(numel(p0),3);
if n_nonpeak_params
    nonpeak_params = p0(end-1:end);
    p0 = p0(1:end-2);    
end

peak_params = reshape(p0,[],3);
end

function p_all = join_params_(p0, p1)

 [peak_p0, nonpeak_p0] = arrange_params_(p0);
 [peak_p1, nonpeak_p1] = arrange_params_(p1);

 nonpeak_p = nonpeak_p0;
 if isempty(nonpeak_p)
     nonpeak_p = nonpeak_p1;
 end

 p_all = [peak_p0; peak_p1];
 p_all = [p_all(:)', nonpeak_p(:)'];

end

%% INPUT VALIDATION FUNCTIONS
function validateReducedModel_(inp)
% null_model must be either a res struct or empty
assert(isempty(inp) || (isstruct(inp) && all(isfield(inp, {'gof','type','fit'}))))
end

function validateLRTInp_(inp)

mustBeMember(char(inp), {'none', 'noise', 'intercept', 'null', 'reduced'});

end

function pv = configure_pv_(pv)

if pv.findPeaks && strcmp(pv.lrt, 'none')
    pv.lrt = 'null';    
    warning("Changing lrt to 'null' since findPeaks is set to true.")
end

assert(strcmp(pv.lrt, 'none') || pv.findPeaks || ( ...
    isempty(pv.reduced_model) && ...
    isempty(pv.p0) && isempty(pv.apriori_fooofer)), ...
    "If provided apriori guesses for the reduced model or results from an " + ...
    "apriori fit, MUST SET lrt to 'null' or 'reduced'.")

assert( ...
    sum(cellfun(@(x) ~isempty(x), {pv.p0, pv.apriori_fooofer, pv.reduced_model})) <= 1);
end
