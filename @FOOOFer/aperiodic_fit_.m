function [fitter, res] = aperiodic_fit_(self, fitter, pv)

arguments (Input)

    self FOOOFer
    fitter NLLFitter
    pv.apriori_fooofer FOOOFer = FOOOFer.empty() % if provided estimates are made from the results struct

    pv.p0 (:,:) double = [] % if provided use directly without estimating
    % if pv.lrt = true, pv.p0 = [intercept1, exponent1, NaN;
    % intercept2, exponent2, knee2]

    pv.lrt (1,1) logical = false % LRT to determine knee    
    pv.lrt_p_threshold {mustBePositive, mustBeLessThan(pv.lrt_p_threshold, 1)} = .05
    pv.refineIter = false % model always survives, if true lrt must be set to false
end

%% --- Input Validation ---
assert(isempty(pv.p0) || isempty(pv.apriori_fooofer), ...
    "Either provide p0 or apriori_fooofer, not both.")

%% --- Configure model ---
mdl = fitter.inner_model;
if pv.lrt
    % First round of LRT must have no knee
    mdl.includeKnee = false;
end

%% --- Retrieve initial guesses ---
if isempty(pv.apriori_fooofer) && isempty(pv.p0)
    % Scenario 1: data-driven guesses
    p0 = fitter.estimate();
elseif ~isempty(pv.apriori_fooofer)
    % Scenario 2: guesses from apriori results
else
    % Scenario 3: user provided guesses   
    
    assert(~pv.lrt || (all(size(pv.p0) == [2,4]) && isnan(pv.p0(1,end))), ...
        "If directly providing initial estimates for LRT procedure p0 must be [intercept1, exponent1, NaN; intercept2, exponent2, knee2]");
    p0 = pv.p0(1,:);
    p0 = p0(~isnan(p0));
end
% Estimate parameter boundaries
[lb, ub] = fitter.estimate_parameter_bounds();

%% --- Optimization ---
% Set up optimizer options
H_pattern = double(fitter.compute_hessian(p0) ~= 0); %speeds up memory alloc
optimopts = optimoptions('fmincon',...
    'Display', 'off',...
    'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
    'SpecifyObjectiveGradient', true, ...
    'SubproblemAlgorithm', 'cg', ...
    'EnableFeasibilityMode',true,...
    'HessianFcn', @(p, ~) fitter.compute_hessian(p), ...
    'HessPattern', H_pattern, ...
    'MaxFunctionEvaluations', self.max_func_eval, ...
    'MaxIterations', self.max_fit_iter);

% Fit the model
onset = tic;
if self.verbose    
    msg = 'Fitting the aperiodic component with';
    if mdl.includeKnee, msg = [msg, ' knee...\n\t'];
    else, msg = [msg, 'out knee...\n\t'];
    end
    fprintf(msg);
end
[params, ~, flag, op] = fmincon(@(p) fitter.objective_function(p), ...
    p0, [], [], [], [], lb, ub, [], optimopts);
fit_dur = toc(onset);
if self.verbose, fprintf("\t Took %.2f secs.\n", fit_dur); end

%% --- Goodness of Fit ---
n_params = numel(params);
prev_aic = self.calculate_aic_(mdl.wR, n_params);

%% --- Save results ---
res = struct(iter = self.iter,...
    type = 'aperiodic',...
    seed = p0, ...
    fit = params, ...
    fit_flag=flag, ...
    fit_n_iter = op.iterations, ...
    fit_n_func_eval = op.funcCount, ...
    fit_n_pcg_iter = op.cgiterations, ...
    fit_dur = fit_dur,...
    gof=prev_aic, ...
    gof_metric='aic', ...
    modelSurvived = true, ...
    nextEntry = true);
res_args = namedargs2cell(res);
self.append_to_results(res_args{:});

%% --- LRT ---
if ~pv.lrt, return; end
% 1. Call aperiodic_fit_ again with knee model
reduced_fitter = fitter.copy(); % Save the reduced fitter object (without knee)
mdl.includeKnee = true; % evokes re-constructing the model with knee
if isempty(pv.apriori_fooofer) && isempty(pv.p0)
    p0 = [];
else
    p0 = pv.p0(2,:);
end

% Fit the model, it will also update the results structure
[fitter, full_res] = self.aperiodic_fit_(fitter, lrt=false, p0=p0);
% 2. Compare the reduced and the full model
[p, chi2, r2] = self.do_lrt_(res.gof, full_res.gof, n_params, n_params + 1, mdl.n_sample);
reducedModelSurvived = ~((p < pv.lrt_p_threshold) && (full_res.gof < res.gof)) || full_res.fit_flag <= 0;
% 3. Save results
self.append_to_results(chi2=chi2, ...
                pseudo_r2=r2, p=p, ...
                modelSurvived = ~reducedModelSurvived, ...
                lrt_comparison_row_rel=-1);

self.append_to_results( ...
                modelSurvived = reducedModelSurvived, ...
                step_from_current_entry = -1, ... % update the previous entry
                lrt_comparison_row_rel=1); % relative row index from the previous entry

% Update output with the surviving fitter and result structure
if reducedModelSurvived

    fitter = reduced_fitter;

else

    res = full_res;

end


end