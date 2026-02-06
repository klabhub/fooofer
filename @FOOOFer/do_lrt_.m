function varargout = do_lrt_(aic0, aic1, n_params0, n_params1, n)
% DO_LRT Performs a Likelihood Ratio Test using AIC values.
%
%   [p, chi_stat] = DO_LRT(aic0, aic1, n_params0, n_params1)
%
%   Inputs:
%       aic0      : AIC of the Null (simpler) model
%       aic1      : AIC of the Alternative (complex) model
%       n_params0 : Number of parameters in the Null model
%       n_params1 : Number of parameters in the Alternative model
%       n: sample size, if provided returns effect size
%
%   Outputs:
%       p         : P-value of the test (probability of observing the stat 
%                   under the null hypothesis)
%       chi_stat  : The Chi-squared test statistic
%       eff: effect size as Likelihood Ratio R-squared
%
%   Note: This assumes the models are nested.
arguments
    aic0
    aic1
    n_params0
    n_params1
    n
end
assert(nargout<3 || ~isnan(n), "To output effect size estimation (3rd output), provide the sample size.");

% 1. Calculate Degrees of Freedom
df = abs(n_params1 - n_params0);

% 2. Calculate the Test Statistic
% Formula derived from: AIC = 2*k - 2*ln(L)
% Statistic = 2*(LL_alternative - LL_null)
%           = (AIC_null - AIC_alt) + 2*(k_alt - k_null)
chi_stat = (aic0 - aic1) + 2 * df;

% Ensure statistic is not negative (can happen due to floating point noise
% or if the alternative model fits strictly worse than null)
chi_stat = max(0, chi_stat);


% 3. Calculate P-value
% Using chi2cdf from Statistics Toolbox, or gammainc as fallback
p = chi2cdf(chi_stat, df, 'upper');
varargout = {p, chi_stat};

% Calculate effect size
if ~isnan(n)
    eff = 1 - exp(-chi_stat/n);
    varargout{end+1} = eff;
end

end