#include <armadillo>

using namespace arma;

namespace {

void ecdf(vec& v) {
  vec sorted = sort(v);

  for (vec::size_type i = 0; i < v.size(); ++i) {
    vec::size_type num_lte = 0;
    for (; num_lte < sorted.size() && sorted[num_lte] <= v[i]; ++num_lte) {
    }
    v[i] = static_cast<double>(num_lte) / static_cast<double>(v.size());
  }
}

double skew(const vec& v) {
  double n = static_cast<double>(v.size());
  vec diff = v - mean(v);
  double numerator = (1 / n) * sum(pow(diff, 3));
  double denominator = pow((1 / (n - 1)) * sum(pow(diff, 2)), 1.5);
  return numerator / denominator;
}

double calculate_outlier_score(const mat& data, mat::size_type sample,
                               const mat& left_tail_ecdf,
                               const mat& right_tail_ecdf,
                               const vec& feature_skewness) {
  double score_left = 0.0;
  double score_right = 0.0;
  double score_auto = 0.0;

  // Aggregate tail probabilities for the sample
  for (mat::size_type feature = 0; feature < data.n_cols; ++feature) {
    score_left -= log(left_tail_ecdf(sample, feature));
    score_right -= log(right_tail_ecdf(sample, feature));
    score_auto -= (feature_skewness(feature) < 0.0)
                      ? log(left_tail_ecdf(sample, feature))
                      : log(right_tail_ecdf(sample, feature));
  }

  // Calculate the outlier score for the sample
  return std::max({score_left, score_right, score_auto});
}

}  // namespace

vec ecod(const mat& data) {
  // Estimate left and right tail ECDFs
  mat left_tail_ecdf(data);
  left_tail_ecdf.each_col([](vec& c) { ecdf(c); });
  mat right_tail_ecdf(-data);
  right_tail_ecdf.each_col([](vec& c) { ecdf(c); });

  // Calculate features skewness
  vec feature_skewness(data.n_cols);
  for (mat::size_type feature = 0; feature < data.n_cols; ++feature) {
    feature_skewness[feature] = skew(data.col(feature));
  }

  vec outlier_scores(data.n_rows);
  for (mat::size_type sample = 0; sample < data.n_rows; ++sample) {
    outlier_scores(sample) = calculate_outlier_score(
        data, sample, left_tail_ecdf, right_tail_ecdf, feature_skewness);
  }

  return outlier_scores;
}

