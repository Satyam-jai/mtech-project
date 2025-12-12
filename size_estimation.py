"""
Size Estimation for Heart Disease Prediction System
Uses Function Point Analysis and COCOMO Model
"""


class SizeEstimation:
    """Estimate project size using various metrics"""
    
    def __init__(self):
        self.estimates = {}
    
    def function_point_analysis(self):
        """Calculate Function Points"""
        # Count components
        inputs = 11  # 11 patient data input fields
        outputs = 5  # prediction, probability, risk, message, confidence
        inquiries = 4  # health, model-info, feature-info, root
        files = 3  # model, scaler, encoders
        interfaces = 2  # Frontend-Backend, Backend-Model
        
        # Complexity weights (Simple/Average/Complex)
        weights = {
            'inputs': 4,  # Average
            'outputs': 5,  # Average
            'inquiries': 4,  # Average
            'files': 10,  # Average
            'interfaces': 7  # Average
        }
        
        unadjusted_fp = (
            inputs * weights['inputs'] +
            outputs * weights['outputs'] +
            inquiries * weights['inquiries'] +
            files * weights['files'] +
            interfaces * weights['interfaces']
        )
        
        # Technical Complexity Factor (0.65 - 1.35)
        tcf = 1.0  # Average complexity
        
        function_points = unadjusted_fp * tcf
        
        self.estimates['function_points'] = {
            'unadjusted_fp': unadjusted_fp,
            'technical_complexity_factor': tcf,
            'adjusted_fp': function_points
        }
        
        return function_points
    
    def loc_estimation(self):
        """Estimate Lines of Code from Function Points"""
        fp = self.function_point_analysis()
        
        # Language-specific conversion rates (LOC per FP)
        conversion_rates = {
            'python': 54,  # Average for Python
            'javascript': 47,
            'sql': 13
        }
        
        estimated_loc = {
            'python': fp * conversion_rates['python'],
            'javascript': fp * conversion_rates['javascript'],
            'total': fp * conversion_rates['python']
        }
        
        self.estimates['estimated_loc'] = estimated_loc
        return estimated_loc
    
    def cocomo_estimation(self):
        """COCOMO Model for effort and duration estimation"""
        loc = self.loc_estimation()['total']
        kloc = loc / 1000  # Convert to KLOC
        
        # Basic COCOMO (Organic project type)
        # Effort = 2.4 * (KLOC)^1.05 person-months
        # Duration = 2.5 * (Effort)^0.38 months
        
        effort_pm = 2.4 * (kloc ** 1.05)  # Person-months
        duration_months = 2.5 * (effort_pm ** 0.38)
        
        # Average team size
        avg_team_size = effort_pm / duration_months
        
        self.estimates['cocomo'] = {
            'kloc': kloc,
            'effort_person_months': effort_pm,
            'duration_months': duration_months,
            'average_team_size': avg_team_size,
            'estimated_cost_usd': effort_pm * 8000  # Assuming $8000 per person-month
        }
        
        return self.estimates['cocomo']
    
    def generate_estimation_report(self):
        """Generate comprehensive size estimation report"""
        self.function_point_analysis()
        self.loc_estimation()
        self.cocomo_estimation()
        
        report = f"""
{'='*80}
HEART DISEASE PREDICTION SYSTEM - SIZE ESTIMATION REPORT
{'='*80}

1. FUNCTION POINT ANALYSIS
{'─'*80}
Input Forms:                11
Output Reports:             5
Inquiries:                  4
Files:                      3
Interfaces:                 2

Unadjusted FP:              {self.estimates['function_points']['unadjusted_fp']}
Technical Complexity:       {self.estimates['function_points']['technical_complexity_factor']}
Adjusted Function Points:   {self.estimates['function_points']['adjusted_fp']:.1f}

2. LINES OF CODE ESTIMATION
{'─'*80}
Estimated Python LOC:       {self.estimates['estimated_loc']['python']:.0f}
Estimated JavaScript LOC:   {self.estimates['estimated_loc']['javascript']:.0f}
Total Estimated LOC:        {self.estimates['estimated_loc']['total']:.0f}

3. COCOMO MODEL (Basic - Organic)
{'─'*80}
KLOC:                       {self.estimates['cocomo']['kloc']:.2f}
Effort:                     {self.estimates['cocomo']['effort_person_months']:.2f} person-months
Duration:                   {self.estimates['cocomo']['duration_months']:.2f} months
Average Team Size:          {self.estimates['cocomo']['average_team_size']:.1f} persons
Estimated Cost:             ${self.estimates['cocomo']['estimated_cost_usd']:,.0f}

4. PROJECT BREAKDOWN
{'─'*80}
Phase                       Effort (PM)     Duration (Months)
Requirements                {self.estimates['cocomo']['effort_person_months']*0.10:.2f}          {self.estimates['cocomo']['duration_months']*0.15:.2f}
Design                      {self.estimates['cocomo']['effort_person_months']*0.20:.2f}          {self.estimates['cocomo']['duration_months']*0.20:.2f}
Implementation              {self.estimates['cocomo']['effort_person_months']*0.40:.2f}          {self.estimates['cocomo']['duration_months']*0.40:.2f}
Testing                     {self.estimates['cocomo']['effort_person_months']*0.20:.2f}          {self.estimates['cocomo']['duration_months']*0.20:.2f}
Deployment                  {self.estimates['cocomo']['effort_person_months']*0.10:.2f}          {self.estimates['cocomo']['duration_months']*0.05:.2f}

5. ACTUAL vs ESTIMATED
{'─'*80}
Estimated LOC:              {self.estimates['estimated_loc']['total']:.0f}
Actual LOC:                 ~3500 (measured)
Accuracy:                   {(3500/self.estimates['estimated_loc']['total']*100):.1f}%

Estimated Duration:         {self.estimates['cocomo']['duration_months']:.1f} months
Actual Duration:            ~2.5 months
Efficiency:                 {(2.5/self.estimates['cocomo']['duration_months']*100):.1f}%

{'='*80}
CONCLUSION
{'='*80}
The project size estimation using Function Point Analysis and COCOMO model
provides a baseline for project planning. The actual implementation was more
efficient due to:
  - Use of modern frameworks (FastAPI, Streamlit)
  - Pre-trained ML models
  - Reusable components
  - Agile development methodology

{'='*80}
"""
        return report
if __name__ == "__main__":
    print("Generating Size Estimation Report...")
    estimation = SizeEstimation()
    report = estimation.generate_estimation_report()
    print(report)
    
    with open("size_estimation_report.txt", "w",encoding="utf-8") as f:
        f.write(report)
    
    print("\n✅ Size Estimation Report saved as 'size_estimation_report.txt'")

