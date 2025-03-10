// Copyright (c) Sleipnir contributors

#include <concepts>
#include <initializer_list>
#include <functional>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/util/small_vector.hpp>

template <std::size_t NumDecisionVars>
class ToyOptimizationProblem {
 public:
  template <typename F>
    requires requires(F callback) {
      { callback() } -> std::same_as<sleipnir::OptimizationProblem>;
    }
  ToyOptimizationProblem(std::string&& problemName, F&& makeProblemFunc,
                         std::initializer_list<double> argumentMinimum)
      : m_problemName{problemName},
        m_makeProblemFunc{std::forward<F>(makeProblemFunc)},
        m_argumentMinimum{argumentMinimum} {};

  sleipnir::OptimizationProblem MakeProblem() const {
    return m_makeProblemFunc();
  }
  std::string_view ProblemName() const { return m_problemName; }
  std::span<const double> ArgumentMinimum() const { return m_argumentMinimum; }

 private:
  std::string m_problemName;
  std::function<sleipnir::OptimizationProblem(void)> m_makeProblemFunc;
  sleipnir::small_vector<double> m_argumentMinimum;
};

/*static const auto MishrasBird = ToyOptimizationProblem{"Mishra's Bird Function", []() {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    //x.SetValue(input.x);
    auto y = problem.DecisionVariable();
    //y.SetValue(input.y);

    // https://en.wikipedi}, {a.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    problem.Minimize(sleipnir::sin(y) *
                         sleipnir::exp(sleipnir::pow(1 - sleipnir::cos(x), 2)) +
                     sleipnir::cos(x) *
                         sleipnir::exp(sleipnir::pow(1 - sleipnir::sin(y), 2)) +
                     sleipnir::pow(x - y, 2));

    problem.SubjectTo(sleipnir::pow(x + 5, 2) + sleipnir::pow(y + 5, 2) < 25);
 	
    return problem;
}, {1, 2, 3}};*/
