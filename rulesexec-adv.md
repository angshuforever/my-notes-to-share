Extended Rule Validation & Recommendation System
Complete Architecture with Rule Recommendation Engine
Executive Summary
This document extends the existing ML-powered rule validation system to include intelligent rule recommendation capabilities. The system now provides two core functions:

Rule Validation: Determines if provided rules are applicable to a dataset
Rule Recommendation: Suggests relevant rules for datasets, including custom dimensions
System Overview
Frontend Application
        ↓
[Dataset Upload] → [Data Profiling] → [Dual Engine Processing]
                                            ↓
                    ┌─────────────────────────────────────┐
                    │     Rule Validation Engine          │
                    │  (Existing - Classification)        │
                    └─────────────────────────────────────┘
                                            ↓
                    ┌─────────────────────────────────────┐
                    │    Rule Recommendation Engine       │
                    │    (New - Recommendation)           │
                    └─────────────────────────────────────┘
                                            ↓
                    [Unified Rule Dashboard] → [Execute Selected Rules]
Core Architecture
1. Data Flow Architecture
Input Dataset (20 columns)
        ↓
┌─────────────────────────────────────────────────────────┐
│              Data Profiling Engine                      │
│  • Column analysis (types, patterns, distributions)     │
│  • Statistical profiling                                │
│  • Domain detection                                     │
│  • Quality indicators                                   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│              Rule Processing Pipeline                   │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ Rule Validation │    │   Rule Recommendation       │ │
│  │    Engine       │    │       Engine                │ │
│  │                 │    │                             │ │
│  │ Input: Rules +  │    │ Input: Dataset Profile      │ │
│  │        Dataset  │    │ Output: Recommended Rules   │ │
│  │ Output: Valid/  │    │         + Custom Dimensions │ │
│  │         Invalid │    │                             │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│                 Unified Dashboard                       │
│  • Show validation results                              │
│  • Display recommended rules                            │
│  • Allow rule selection/modification                    │
│  • Execute final rule set                               │
└─────────────────────────────────────────────────────────┘
Phase 1: Simple POV Implementation (Weeks 1-2)
1.1 Enhanced Data Profiling
class DatasetProfiler:
    def __init__(self):
        self.standard_dimensions = ['COMPLETENESS', 'ACCURACY', 'CONSISTENCY', 
                                  'VALIDITY', 'UNIQUENESS', 'INTEGRITY']
    
    def profile_dataset(self, dataset):
        """Create comprehensive dataset profile for recommendation"""
        
        profile = {
            'dataset_metadata': self.extract_metadata(dataset),
            'columns': [],
            'relationships': self.detect_relationships(dataset),
            'domain_indicators': self.detect_domain(dataset),
            'quality_baseline': self.calculate_baseline_quality(dataset)
        }
        
        for column in dataset.columns:
            column_profile = self.profile_column(column, dataset)
            profile['columns'].append(column_profile)
        
        return profile
    
    def profile_column(self, column, dataset):
        """Detailed column profiling for rule recommendation"""
        
        column_data = dataset[column]
        
        return {
            'name': column,
            'data_type': str(column_data.dtype),
            'basic_stats': self.calculate_basic_stats(column_data),
            'pattern_analysis': self.analyze_patterns(column_data),
            'domain_hints': self.detect_column_domain(column, column_data),
            'quality_indicators': self.assess_column_quality(column_data),
            'custom_dimension_candidates': self.suggest_custom_dimensions(column_data)
        }
    
    def suggest_custom_dimensions(self, column_data):
        """Suggest custom dimensions based on column characteristics"""
        
        suggestions = []
        
        # Business logic patterns
        if self.is_email_column(column_data):
            suggestions.append({
                'dimension': 'EMAIL_DELIVERABILITY',
                'description': 'Validates email deliverability and format',
                'confidence': 0.9
            })
        
        if self.is_date_column(column_data):
            suggestions.append({
                'dimension': 'TEMPORAL_RELEVANCE',
                'description': 'Ensures dates are within business-relevant timeframes',
                'confidence': 0.85
            })
        
        if self.is_numeric_id(column_data):
            suggestions.append({
                'dimension': 'ID_SEQUENTIAL_INTEGRITY',
                'description': 'Validates ID sequence and gap analysis',
                'confidence': 0.8
            })
        
        return suggestions
1.2 Simple Rule Recommendation Engine
class RuleRecommendationEngine:
    def __init__(self):
        self.rule_templates = self.load_rule_templates()
        self.dimension_rules = self.load_dimension_rules()
    
    def recommend_rules(self, dataset_profile):
        """Main recommendation function"""
        
        recommendations = {
            'standard_dimension_rules': [],
            'custom_dimension_rules': [],
            'confidence_scores': {}
        }
        
        # Standard dimension recommendations
        for column in dataset_profile['columns']:
            column_rules = self.recommend_for_column(column)
            recommendations['standard_dimension_rules'].extend(column_rules)
        
        # Custom dimension recommendations
        custom_rules = self.recommend_custom_dimensions(dataset_profile)
        recommendations['custom_dimension_rules'] = custom_rules
        
        # Calculate overall confidence
        recommendations['confidence_scores'] = self.calculate_confidence(recommendations)
        
        return recommendations
    
    def recommend_for_column(self, column_profile):
        """Recommend rules for a specific column"""
        
        recommended_rules = []
        
        # Completeness rules
        if column_profile['basic_stats']['null_percentage'] > 0:
            recommended_rules.append({
                'rule_type': 'COMPLETENESS',
                'rule_definition': f"NULL_CHECK({column_profile['name']}) < 5%",
                'reasoning': f"Column has {column_profile['basic_stats']['null_percentage']:.1f}% nulls",
                'priority': 'HIGH' if column_profile['basic_stats']['null_percentage'] > 10 else 'MEDIUM',
                'confidence': 0.9
            })
        
        # Uniqueness rules for potential keys
        if column_profile['basic_stats']['unique_percentage'] > 95:
            recommended_rules.append({
                'rule_type': 'UNIQUENESS',
                'rule_definition': f"DUPLICATE_CHECK({column_profile['name']}) = 0",
                'reasoning': f"Column has {column_profile['basic_stats']['unique_percentage']:.1f}% unique values",
                'priority': 'HIGH',
                'confidence': 0.85
            })
        
        # Validity rules based on patterns
        if column_profile['domain_hints']:
            for hint in column_profile['domain_hints']:
                if hint['type'] == 'email':
                    recommended_rules.append({
                        'rule_type': 'VALIDITY',
                        'rule_definition': f"EMAIL_FORMAT_CHECK({column_profile['name']})",
                        'reasoning': f"Detected email pattern with {hint['confidence']} confidence",
                        'priority': 'HIGH',
                        'confidence': hint['confidence']
                    })
        
        return recommended_rules
    
    def recommend_custom_dimensions(self, dataset_profile):
        """Recommend custom dimensions based on dataset analysis"""
        
        custom_recommendations = []
        
        # Business domain specific recommendations
        domain = dataset_profile.get('domain_indicators', {}).get('primary_domain')
        
        if domain == 'customer':
            custom_recommendations.extend(self.customer_domain_rules(dataset_profile))
        elif domain == 'financial':
            custom_recommendations.extend(self.financial_domain_rules(dataset_profile))
        elif domain == 'product':
            custom_recommendations.extend(self.product_domain_rules(dataset_profile))
        
        # Cross-column relationship rules
        relationship_rules = self.recommend_relationship_rules(dataset_profile)
        custom_recommendations.extend(relationship_rules)
        
        return custom_recommendations
    
    def customer_domain_rules(self, dataset_profile):
        """Customer-specific custom dimension rules"""
        
        rules = []
        
        # Customer lifecycle dimension
        if self.has_date_columns(dataset_profile):
            rules.append({
                'custom_dimension': 'CUSTOMER_LIFECYCLE_CONSISTENCY',
                'rule_definition': 'created_date <= last_activity_date <= current_date',
                'description': 'Ensures customer lifecycle dates are logical',
                'business_value': 'Prevents impossible customer journey timelines',
                'confidence': 0.8
            })
        
        # Customer reachability dimension
        contact_columns = self.find_contact_columns(dataset_profile)
        if contact_columns:
            rules.append({
                'custom_dimension': 'CUSTOMER_REACHABILITY',
                'rule_definition': 'has_email OR has_phone OR has_address',
                'description': 'Ensures customer has at least one contact method',
                'business_value': 'Guarantees ability to reach customers',
                'confidence': 0.9
            })
        
        return rules
1.3 Unified Processing Engine
class UnifiedRuleEngine:
    def __init__(self):
        self.validator = RuleValidationEngine()  # Existing from Phase 1
        self.recommender = RuleRecommendationEngine()
        self.profiler = DatasetProfiler()
    
    def process_dataset(self, dataset, provided_rules=None):
        """Main processing function combining validation and recommendation"""
        
        # Step 1: Profile the dataset
        dataset_profile = self.profiler.profile_dataset(dataset)
        
        results = {
            'dataset_profile': dataset_profile,
            'validation_results': None,
            'recommendations': None,
            'unified_dashboard_data': None
        }
        
        # Step 2: Validate provided rules (if any)
        if provided_rules:
            validation_results = []
            for rule in provided_rules:
                validation = self.validator.validate_rule_applicability(rule, dataset_profile)
                validation_results.append(validation)
            results['validation_results'] = validation_results
        
        # Step 3: Generate recommendations
        recommendations = self.recommender.recommend_rules(dataset_profile)
        results['recommendations'] = recommendations
        
        # Step 4: Create unified dashboard data
        results['unified_dashboard_data'] = self.create_dashboard_data(results)
        
        return results
    
    def create_dashboard_data(self, processing_results):
        """Create data structure for frontend dashboard"""
        
        dashboard_data = {
            'dataset_summary': {
                'total_columns': len(processing_results['dataset_profile']['columns']),
                'estimated_quality_score': self.calculate_estimated_quality(processing_results),
                'domain': processing_results['dataset_profile']['domain_indicators']
            },
            'rule_sections': {
                'provided_rules': self.format_validation_results(processing_results.get('validation_results')),
                'recommended_standard': processing_results['recommendations']['standard_dimension_rules'],
                'recommended_custom': processing_results['recommendations']['custom_dimension_rules']
            },
            'action_items': self.generate_action_items(processing_results)
        }
        
        return dashboard_data
Phase 2: Frontend Application (Week 3)
2.1 Simple Frontend Architecture
# Using Streamlit for rapid prototyping
import streamlit as st
import pandas as pd

class RuleRecommendationApp:
    def __init__(self):
        self.engine = UnifiedRuleEngine()
    
    def run(self):
        st.title("Data Quality Rule Validation & Recommendation System")
        
        # Step 1: Dataset Upload
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])
        
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
            st.success(f"Dataset uploaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
            
            # Step 2: Show dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(dataset.head())
            
            # Step 3: Optional rule upload
            st.subheader("Existing Rules (Optional)")
            provided_rules = self.get_provided_rules()
            
            # Step 4: Process button
            if st.button("Analyze Dataset & Generate Recommendations"):
                with st.spinner("Processing..."):
                    results = self.engine.process_dataset(dataset, provided_rules)
                    self.display_results(results)
    
    def get_provided_rules(self):
        """Allow users to input existing rules"""
        
        rules = []
        num_rules = st.number_input("Number of existing rules to validate", min_value=0, max_value=10, value=0)
        
        for i in range(num_rules):
            with st.expander(f"Rule {i+1}"):
                rule_type = st.selectbox(f"Rule Type {i+1}", 
                                       ['COMPLETENESS', 'ACCURACY', 'CONSISTENCY', 'VALIDITY', 'UNIQUENESS', 'INTEGRITY'])
                rule_definition = st.text_input(f"Rule Definition {i+1}")
                target_column = st.selectbox(f"Target Column {i+1}", ['All'] + list(st.session_state.get('columns', [])))
                
                if rule_definition:
                    rules.append({
                        'rule_type': rule_type,
                        'rule_definition': rule_definition,
                        'target_column': target_column
                    })
        
        return rules if rules else None
    
    def display_results(self, results):
        """Display unified results dashboard"""
        
        dashboard_data = results['unified_dashboard_data']
        
        # Dataset Summary
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Columns", dashboard_data['dataset_summary']['total_columns'])
        with col2:
            st.metric("Estimated Quality Score", f"{dashboard_data['dataset_summary']['estimated_quality_score']:.1f}%")
        with col3:
            st.metric("Domain", dashboard_data['dataset_summary']['domain'].get('primary_domain', 'Unknown'))
        
        # Validation Results (if provided rules exist)
        if results['validation_results']:
            st.subheader("Rule Validation Results")
            self.display_validation_results(results['validation_results'])
        
        # Recommendations
        st.subheader("Recommended Rules")
        
        # Standard dimension rules
        st.markdown("### Standard Data Quality Dimensions")
        self.display_rule_recommendations(dashboard_data['rule_sections']['recommended_standard'], "standard")
        
        # Custom dimension rules
        st.markdown("### Custom Business Dimensions")
        self.display_rule_recommendations(dashboard_data['rule_sections']['recommended_custom'], "custom")
        
        # Action Items
        st.subheader("Recommended Actions")
        for action in dashboard_data['action_items']:
            st.info(f"**{action['priority']}**: {action['description']}")
    
    def display_validation_results(self, validation_results):
        """Display rule validation results"""
        
        validation_df = pd.DataFrame([
            {
                'Rule': result['rule']['rule_definition'][:50] + "...",
                'Type': result['rule']['rule_type'],
                'Applicable': "✅" if result['is_applicable'] else "❌",
                'Confidence': f"{result['confidence']:.2f}",
                'Reasoning': result['reasoning'][:100] + "..."
            }
            for result in validation_results
        ])
        
        st.dataframe(validation_df)
    
    def display_rule_recommendations(self, rules, rule_category):
        """Display recommended rules with selection capability"""
        
        if not rules:
            st.info(f"No {rule_category} rules recommended for this dataset.")
            return
        
        for i, rule in enumerate(rules):
            with st.expander(f"{rule.get('rule_type', rule.get('custom_dimension', 'Unknown'))} - Priority: {rule.get('priority', 'MEDIUM')}"):
                
                # Rule details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Rule Definition**: {rule['rule_definition']}")
                    st.write(f"**Reasoning**: {rule.get('reasoning', rule.get('description', 'N/A'))}")
                    if 'business_value' in rule:
                        st.write(f"**Business Value**: {rule['business_value']}")
                
                with col2:
                    st.metric("Confidence", f"{rule['confidence']:.2f}")
                    selected = st.checkbox(f"Select Rule {i+1}", key=f"{rule_category}_rule_{i}")
                    
                    if selected:
                        st.session_state[f'selected_{rule_category}_rules'] = st.session_state.get(f'selected_{rule_category}_rules', []) + [rule]

# Run the app
if __name__ == "__main__":
    app = RuleRecommendationApp()
    app.run()
Phase 3: ML Model Training (Week 4)
3.1 Training Data Generation for Recommendation
class RecommendationTrainingDataGenerator:
    def __init__(self):
        self.rule_patterns = self.load_rule_patterns()
    
    def generate_training_data(self, idmc_projects):
        """Generate training data for recommendation model"""
        
        training_records = []
        
        for project in idmc_projects:
            for dataset in project.datasets:
                # Profile dataset
                dataset_profile = self.profile_dataset(dataset)
                
                # Get all rules that were actually applied
                applied_rules = dataset.get_applied_rules()
                
                # Generate positive examples (rules that were applied)
                for rule in applied_rules:
                    training_records.append({
                        'dataset_features': self.extract_dataset_features(dataset_profile),
                        'rule_features': self.extract_rule_features(rule),
                        'should_recommend': True,
                        'success_score': rule.execution_stats.success_rate
                    })
                
                # Generate negative examples (rules that could have been applied but weren't)
                potential_rules = self.generate_potential_rules(dataset_profile)
                for rule in potential_rules:
                    if rule not in applied_rules:
                        training_records.append({
                            'dataset_features': self.extract_dataset_features(dataset_profile),
                            'rule_features': self.extract_rule_features(rule),
                            'should_recommend': False,
                            'success_score': 0.0
                        })
        
        return training_records
3.2 Simple Recommendation Model
class RuleRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.feature_engineer = RecommendationFeatureEngineer()
    
    def train(self, training_data):
        """Train recommendation model"""
        
        # Prepare features and labels
        X = []
        y = []
        
        for record in training_data:
            features = self.feature_engineer.combine_features(
                record['dataset_features'],
                record['rule_features']
            )
            X.append(features)
            y.append(record['should_recommend'])
        
        # Train model
        self.model.fit(X, y)
        
        return self.model
    
    def recommend_rules_for_dataset(self, dataset_profile):
        """Use trained model to recommend rules"""
        
        # Generate all possible rules for this dataset
        candidate_rules = self.generate_candidate_rules(dataset_profile)
        
        recommendations = []
        
        for rule in candidate_rules:
            # Extract features
            features = self.feature_engineer.combine_features(
                self.extract_dataset_features(dataset_profile),
                self.extract_rule_features(rule)
            )
            
            # Get recommendation probability
            recommendation_prob = self.model.predict_proba([features])[0][1]
            
            if recommendation_prob > 0.6:  # Threshold for recommendation
                recommendations.append({
                    'rule': rule,
                    'confidence': recommendation_prob,
                    'reasoning': self.generate_reasoning(features, rule)
                })
        
        # Sort by confidence and return top recommendations
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:10]  # Top 10 recommendations
Simple API Implementation
API Endpoints
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Rule Validation & Recommendation API")

class DatasetProcessingRequest(BaseModel):
    dataset_profile: dict
    provided_rules: list = []

class ProcessingResponse(BaseModel):
    validation_results: list = []
    recommendations: dict
    dashboard_data: dict

@app.post("/process-dataset", response_model=ProcessingResponse)
async def process_dataset(request: DatasetProcessingRequest):
    """Main endpoint for dataset processing"""
    
    engine = UnifiedRuleEngine()
    results = engine.process_dataset(
        dataset_profile=request.dataset_profile,
        provided_rules=request.provided_rules
    )
    
    return ProcessingResponse(
        validation_results=results.get('validation_results', []),
        recommendations=results['recommendations'],
        dashboard_data=results['unified_dashboard_data']
    )

@app.post("/validate-rule")
async def validate_single_rule(rule: dict, dataset_profile: dict):
    """Validate a single rule against dataset"""
    
    validator = RuleValidationEngine()
    result = validator.validate_rule_applicability(rule, dataset_profile)
    return result

@app.post("/recommend-rules")
async def recommend_rules(dataset_profile: dict):
    """Get rule recommendations for dataset"""
    
    recommender = RuleRecommendationEngine()
    recommendations = recommender.recommend_rules(dataset_profile)
    return recommendations
Deployment Strategy
Simple Deployment Architecture
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                 │
│                    Port: 8501                           │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────┐
│                API Gateway (FastAPI)                    │
│                    Port: 8000                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Processing Engine                          │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ Rule Validation │    │   Rule Recommendation       │ │
│  │    Service      │    │       Service               │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
Success Metrics
Technical Metrics
Rule Validation Accuracy: >85%
Recommendation Precision: >70% (recommended rules are actually useful)
Recommendation Recall: >60% (don't miss important rules)
API Response Time: <2 seconds for full dataset processing
System Availability: >99%
Business Metrics
User Adoption: >50% of datasets processed use recommended rules
Rule Discovery: 30% increase in useful rules identified per dataset
Custom Dimension Value: 20% of custom dimensions become standard practice
Data Quality Improvement: 15% average increase in quality scores
Implementation Timeline
Week 1: Core Foundation
Implement enhanced data profiling
Build basic rule recommendation engine
Create simple custom dimension suggestions
Week 2: Integration & Testing
Integrate validation and recommendation engines
Build unified processing pipeline
Create basic test suite
Week 3: Frontend Development
Build Streamlit application
Implement dashboard functionality
Add rule selection capabilities
Week 4: Deployment & Optimization
Deploy API and frontend
Performance optimization
User testing and feedback
Key Success Factors
Start Simple: Begin with pattern-based recommendations before ML complexity
User Feedback Loop: Collect feedback on recommendations from day one
Iterative Improvement: Add sophistication based on user needs
Clear Value Proposition: Focus on time savings and quality improvements
Easy Integration: Ensure smooth integration with existing IDMC workflows
This architecture provides a clear, simple path to extend your existing rule validation system with intelligent recommendation capabilities while maintaining simplicity and deliverability.
