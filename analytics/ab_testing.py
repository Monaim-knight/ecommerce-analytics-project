"""
A/B Testing and Statistical Analysis Module
Demonstrates statistical analysis, hypothesis testing, and experiment design skills
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ABTestingFramework:
    """
    Comprehensive A/B testing framework for e-commerce experiments
    Demonstrates statistical analysis and hypothesis testing skills
    """
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
        self.significance_level = 0.05
    
    def design_experiment(self, 
                         experiment_name: str,
                         control_group: pd.DataFrame,
                         treatment_group: pd.DataFrame,
                         metric: str,
                         hypothesis: str = "two_sided") -> Dict:
        """
        Design and execute A/B test experiment
        """
        logger.info(f"Designing experiment: {experiment_name}")
        
        experiment = {
            'name': experiment_name,
            'metric': metric,
            'hypothesis': hypothesis,
            'control_size': len(control_group),
            'treatment_size': len(treatment_group),
            'start_date': datetime.now(),
            'status': 'running'
        }
        
        self.experiments[experiment_name] = experiment
        
        return experiment
    
    def run_conversion_test(self, 
                           control_data: pd.DataFrame,
                           treatment_data: pd.DataFrame,
                           experiment_name: str = "conversion_test") -> Dict:
        """
        Run conversion rate A/B test
        """
        # Calculate conversion rates
        control_conversion = control_data['converted'].mean()
        treatment_conversion = treatment_data['converted'].mean()
        
        # Perform statistical test
        chi2_stat, p_value, dof, expected = chi2_contingency([
            [control_data['converted'].sum(), len(control_data) - control_data['converted'].sum()],
            [treatment_data['converted'].sum(), len(treatment_data) - treatment_data['converted'].sum()]
        ])
        
        # Calculate effect size
        effect_size = treatment_conversion - control_conversion
        relative_improvement = (effect_size / control_conversion) * 100 if control_conversion > 0 else 0
        
        # Determine significance
        is_significant = p_value < self.significance_level
        
        results = {
            'experiment_name': experiment_name,
            'control_conversion': control_conversion,
            'treatment_conversion': treatment_conversion,
            'effect_size': effect_size,
            'relative_improvement': relative_improvement,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': 1 - self.significance_level,
            'test_type': 'chi_square',
            'timestamp': datetime.now()
        }
        
        self.results[experiment_name] = results
        return results
    
    def run_revenue_test(self, 
                        control_data: pd.DataFrame,
                        treatment_data: pd.DataFrame,
                        experiment_name: str = "revenue_test") -> Dict:
        """
        Run revenue per user A/B test
        """
        # Calculate average revenue
        control_revenue = control_data['revenue'].mean()
        treatment_revenue = treatment_data['revenue'].mean()
        
        # Perform t-test
        t_stat, p_value = ttest_ind(control_data['revenue'], treatment_data['revenue'])
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_data['revenue'].var() + 
                             (len(treatment_data) - 1) * treatment_data['revenue'].var()) / 
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (treatment_revenue - control_revenue) / pooled_std
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            control_data['revenue'], treatment_data['revenue']
        )
        
        results = {
            'experiment_name': experiment_name,
            'control_revenue': control_revenue,
            'treatment_revenue': treatment_revenue,
            'effect_size': treatment_revenue - control_revenue,
            'relative_improvement': ((treatment_revenue - control_revenue) / control_revenue) * 100,
            'p_value': p_value,
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': confidence_interval,
            'test_type': 't_test',
            'timestamp': datetime.now()
        }
        
        self.results[experiment_name] = results
        return results
    
    def run_pricing_experiment(self, 
                              control_prices: pd.DataFrame,
                              treatment_prices: pd.DataFrame,
                              experiment_name: str = "pricing_test") -> Dict:
        """
        Run pricing strategy A/B test
        """
        # Calculate key metrics
        control_metrics = self._calculate_pricing_metrics(control_prices)
        treatment_metrics = self._calculate_pricing_metrics(treatment_prices)
        
        # Perform statistical tests
        revenue_test = self._test_revenue_impact(control_prices, treatment_prices)
        margin_test = self._test_margin_impact(control_prices, treatment_prices)
        
        results = {
            'experiment_name': experiment_name,
            'control_metrics': control_metrics,
            'treatment_metrics': treatment_metrics,
            'revenue_test': revenue_test,
            'margin_test': margin_test,
            'overall_significant': revenue_test['is_significant'] or margin_test['is_significant'],
            'timestamp': datetime.now()
        }
        
        self.results[experiment_name] = results
        return results
    
    def _calculate_pricing_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate pricing-related metrics"""
        return {
            'avg_price': df['Final MRP Old'].mean(),
            'avg_margin': df['Margin'].mean() if 'Margin' in df.columns else 0,
            'margin_percentage': df['Margin_Percentage'].mean() if 'Margin_Percentage' in df.columns else 0,
            'price_variance': df['Final MRP Old'].var(),
            'total_revenue': (df['Final MRP Old'] * df['Stock']).sum() if 'Stock' in df.columns else 0
        }
    
    def _test_revenue_impact(self, control: pd.DataFrame, treatment: pd.DataFrame) -> Dict:
        """Test revenue impact between control and treatment"""
        control_revenue = control['Final MRP Old'] * control['Stock']
        treatment_revenue = treatment['Final MRP Old'] * treatment['Stock']
        
        t_stat, p_value = ttest_ind(control_revenue, treatment_revenue)
        
        return {
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'effect_size': treatment_revenue.mean() - control_revenue.mean(),
            'relative_change': ((treatment_revenue.mean() - control_revenue.mean()) / control_revenue.mean()) * 100
        }
    
    def _test_margin_impact(self, control: pd.DataFrame, treatment: pd.DataFrame) -> Dict:
        """Test margin impact between control and treatment"""
        if 'Margin' not in control.columns or 'Margin' not in treatment.columns:
            return {'p_value': 1.0, 'is_significant': False, 'effect_size': 0, 'relative_change': 0}
        
        t_stat, p_value = ttest_ind(control['Margin'], treatment['Margin'])
        
        return {
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'effect_size': treatment['Margin'].mean() - control['Margin'].mean(),
            'relative_change': ((treatment['Margin'].mean() - control['Margin'].mean()) / control['Margin'].mean()) * 100
        }
    
    def _calculate_confidence_interval(self, control_data: pd.Series, treatment_data: pd.Series, 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        diff = treatment_data.mean() - control_data.mean()
        pooled_se = np.sqrt(control_data.var() / len(control_data) + treatment_data.var() / len(treatment_data))
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = diff - z_score * pooled_se
        upper_bound = diff + z_score * pooled_se
        
        return (lower_bound, upper_bound)
    
    def calculate_sample_size(self, 
                            baseline_conversion: float,
                            mde: float,  # Minimum Detectable Effect
                            alpha: float = 0.05,
                            power: float = 0.8) -> int:
        """
        Calculate required sample size for A/B test
        """
        from statsmodels.stats.power import NormalIndPower
        
        power_analysis = NormalIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=mde,
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=1.0
        )
        
        return int(sample_size)
    
    def generate_experiment_report(self, experiment_name: str) -> Dict:
        """Generate comprehensive experiment report"""
        if experiment_name not in self.results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        result = self.results[experiment_name]
        
        report = {
            'experiment_summary': {
                'name': experiment_name,
                'status': 'completed',
                'timestamp': result['timestamp'],
                'test_type': result.get('test_type', 'unknown')
            },
            'statistical_results': {
                'p_value': result['p_value'],
                'is_significant': result['is_significant'],
                'confidence_level': 1 - self.significance_level
            },
            'business_impact': {
                'effect_size': result.get('effect_size', 0),
                'relative_improvement': result.get('relative_improvement', 0)
            },
            'recommendations': self._generate_recommendations(result)
        }
        
        return report
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        if result['is_significant']:
            if result.get('relative_improvement', 0) > 0:
                recommendations.append("Implement treatment group strategy - significant positive impact detected")
                recommendations.append(f"Expected improvement: {result.get('relative_improvement', 0):.2f}%")
            else:
                recommendations.append("Reject treatment group - significant negative impact detected")
        else:
            recommendations.append("No significant difference detected - consider larger sample size or different approach")
        
        if result.get('p_value', 1) > 0.1:
            recommendations.append("Consider running experiment longer to achieve statistical significance")
        
        return recommendations
    
    def visualize_results(self, experiment_name: str, save_path: Optional[str] = None):
        """Create visualization of A/B test results"""
        if experiment_name not in self.results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        result = self.results[experiment_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Control vs Treatment comparison
        if 'control_conversion' in result and 'treatment_conversion' in result:
            labels = ['Control', 'Treatment']
            values = [result['control_conversion'], result['treatment_conversion']]
            colors = ['#ff9999', '#66b3ff']
            
            axes[0, 0].bar(labels, values, color=colors)
            axes[0, 0].set_title('Conversion Rate Comparison')
            axes[0, 0].set_ylabel('Conversion Rate')
            
            # Add significance indicator
            if result['is_significant']:
                axes[0, 0].text(0.5, max(values) * 1.1, '***', ha='center', fontsize=20)
        
        # P-value distribution
        axes[0, 1].hist([result['p_value']], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].axvline(self.significance_level, color='red', linestyle='--', label=f'α = {self.significance_level}')
        axes[0, 1].set_title('P-value Distribution')
        axes[0, 1].set_xlabel('P-value')
        axes[0, 1].legend()
        
        # Effect size visualization
        if 'effect_size' in result:
            effect_size = result['effect_size']
            axes[1, 0].bar(['Effect Size'], [effect_size], color='lightgreen' if effect_size > 0 else 'lightcoral')
            axes[1, 0].set_title('Effect Size')
            axes[1, 0].set_ylabel('Effect Size')
        
        # Confidence interval
        if 'confidence_interval' in result:
            ci_lower, ci_upper = result['confidence_interval']
            axes[1, 1].errorbar(['Difference'], [ci_upper - ci_lower], yerr=[[ci_lower], [ci_upper]], 
                               fmt='o', capsize=5)
            axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('Confidence Interval')
            axes[1, 1].set_ylabel('Difference')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_sample_data():
    """Create sample data for A/B testing demonstration"""
    np.random.seed(42)
    
    # Sample conversion data
    n_control = 1000
    n_treatment = 1000
    
    control_conversion = np.random.binomial(1, 0.15, n_control)
    treatment_conversion = np.random.binomial(1, 0.18, n_treatment)
    
    control_data = pd.DataFrame({
        'user_id': range(n_control),
        'converted': control_conversion,
        'revenue': np.random.exponential(50, n_control) * control_conversion
    })
    
    treatment_data = pd.DataFrame({
        'user_id': range(n_treatment),
        'converted': treatment_conversion,
        'revenue': np.random.exponential(55, n_treatment) * treatment_conversion
    })
    
    return control_data, treatment_data

def main():
    """Demonstrate A/B testing capabilities"""
    print("=== A/B Testing Framework Demonstration ===")
    
    # Initialize framework
    ab_framework = ABTestingFramework()
    
    # Create sample data
    control_data, treatment_data = create_sample_data()
    
    # Run conversion test
    print("\n1. Running Conversion Rate A/B Test...")
    conversion_results = ab_framework.run_conversion_test(control_data, treatment_data, "conversion_test")
    
    print(f"Control Conversion Rate: {conversion_results['control_conversion']:.3f}")
    print(f"Treatment Conversion Rate: {conversion_results['treatment_conversion']:.3f}")
    print(f"P-value: {conversion_results['p_value']:.4f}")
    print(f"Significant: {conversion_results['is_significant']}")
    print(f"Relative Improvement: {conversion_results['relative_improvement']:.2f}%")
    
    # Run revenue test
    print("\n2. Running Revenue A/B Test...")
    revenue_results = ab_framework.run_revenue_test(control_data, treatment_data, "revenue_test")
    
    print(f"Control Revenue: ${revenue_results['control_revenue']:.2f}")
    print(f"Treatment Revenue: ${revenue_results['treatment_revenue']:.2f}")
    print(f"P-value: {revenue_results['p_value']:.4f}")
    print(f"Cohen's d: {revenue_results['cohens_d']:.3f}")
    
    # Generate report
    print("\n3. Generating Experiment Report...")
    report = ab_framework.generate_experiment_report("conversion_test")
    
    print("\n=== Experiment Report ===")
    print(f"Experiment: {report['experiment_summary']['name']}")
    print(f"Significant: {report['statistical_results']['is_significant']}")
    print(f"P-value: {report['statistical_results']['p_value']:.4f}")
    print(f"Business Impact: {report['business_impact']['relative_improvement']:.2f}%")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # Calculate sample size for future experiments
    print("\n4. Sample Size Calculation...")
    sample_size = ab_framework.calculate_sample_size(
        baseline_conversion=0.15,
        mde=0.03,  # 3% minimum detectable effect
        alpha=0.05,
        power=0.8
    )
    print(f"Required sample size per group: {sample_size}")
    
    return ab_framework

if __name__ == "__main__":
    ab_framework = main() 