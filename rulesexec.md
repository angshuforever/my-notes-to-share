# Complete End-to-End Prompt Sequence
## From Initial Question to Final Solution

This document shows the complete conversation flow that led to the ML-powered rule applicability system for IDMC.

---

## Initial Human Query

**Human:** "Informatica has the tool called IDMC. In the tool we are trying to build a small capability where we will ingest data from in system csv file or table in database or table in database and this will go through the ingested data and based on some defined rules on the particular column of the csv say, it will try to build a dimension. there are 6 standard dimensions apart from that that can be a custom dimension where it will build as score about the quality of the data and eventually over a time it will show us scores of the data based on the recovery of the data quality so it will create a graph now we are trying to build and external machine learning component that engine will help us to ideally some shape or form give indication if a particular rule is applicable to that data set or not. How to go about it and build it. If we want to start simple."

---

## Initial AI Response (Misunderstood Architecture)

**AI Initial Response:** 
The AI initially interpreted this as IDMC sending data to the ML component for rule recommendations, suggesting:

- IDMC → Extract basic statistics → Send to ML API
- ML component analyzes data patterns → Recommends applicable rules
- IDMC applies recommended rules → Generates quality scores

**Suggested Architecture:**
```
[IDMC Profiling] → [ML API] → [Rule Recommendations] → [Apply Rules]
```

**Technical approach suggested:**
- Rule-based classifier for simple patterns
- Basic ML classifier using Random Forest/XGBoost
- REST API integration
- Features: data type, null%, unique%, pattern matches

---

## Human Clarification

**Human:** "No. Little change. IDMC does not send anything. We probably first feed some test data out of existing rules from multiple profiles spread across many projects. May be we generate the rules for data quality through LLMs. Once the model the trained with the training data of rules, then before we try to feed any rule we need to see if that rule is applicable to the given dataset or not."

---

## Key Insight Recognition

The AI recognized the fundamental architectural change:

**Corrected Understanding:**
- This is about **rule validation**, not rule recommendation
- Training data comes from **existing IDMC projects**
- **LLMs generate synthetic rules** for training augmentation
- ML model answers: **"Should this rule be applied to this dataset?"**
- **Validation happens before rule application**, not after

---

## Revised Architecture

**Training Phase:**
```
[Extract IDMC Rules] → [LLM Rule Generation] → [Training Dataset] → [ML Model]
```

**Inference Phase:**
```
[Candidate Rule + Dataset Profile] → [ML Validation] → [Apply/Skip Decision]
```

---

## Solution Evolution

### 1. Data Collection Strategy
**Original approach:** Wait for IDMC to send data
**Revised approach:** Proactively extract from existing IDMC projects

```python
# Extract existing rules from IDMC projects
for project in idmc_projects:
    for profile in project.data_profiles:
        for rule in profile.applied_rules:
            training_data.append({
                'rule_definition': rule.definition,
                'dataset_profile': profile.statistics,
                'is_applicable': True  # Since it was applied
            })
```

### 2. LLM Integration for Synthetic Data
**New requirement:** Generate training rules using LLMs

```python
def generate_rules_with_llm(dataset_profile):
    prompt = f"""
    Given this dataset profile:
    - Column: {dataset_profile.column_name}
    - Data Type: {dataset_profile.data_type}
    - Sample Values: {dataset_profile.sample_values}
    
    Generate 5 potential data quality rules that could apply to this column.
    """
    return llm_client.generate(prompt)
```

### 3. Binary Classification Problem
**Problem reframed:** Predict rule applicability (Yes/No) for rule-dataset pairs

```python
# Features: dataset_profile + rule characteristics
# Label: is_applicable (boolean)
model = RandomForestClassifier()
model.fit(features, labels)
```

### 4. API Design for Pre-Application Validation
**New API purpose:** Validate before applying, not recommend after profiling

```python
@app.post("/validate-rule-applicability")
def validate_rule(rule, dataset_profile):
    probability = model.predict_proba([features])[0][1]
    return {
        "is_applicable": probability > 0.7,
        "confidence": probability
    }
```

---

## Final Solution Components

### 1. Training Data Pipeline
- **Source 1:** Extract existing rule applications from IDMC projects
- **Source 2:** LLM-generated synthetic rules for dataset profiles
- **Source 3:** Negative examples (rules that shouldn't apply to datasets)

### 2. Feature Engineering
- **Dataset features:** Row count, column types, quality indicators, domain patterns
- **Rule features:** Complexity, type, target data type, business domain
- **Compatibility features:** Type matching, pattern alignment, statistical fit

### 3. ML Model
- **Algorithm:** Random Forest / XGBoost (interpretable, handles mixed data types)
- **Input:** Feature vector of (rule + dataset_profile)
- **Output:** Binary classification + confidence score
- **Training:** Balanced dataset with positive/negative examples

### 4. Integration Architecture
```
[IDMC Rule Engine] → [ML API Validation] → [Apply/Skip Decision] → [Quality Assessment]
```

### 5. Continuous Learning
- **Feedback collection:** Track rule application success/failure
- **Model updates:** Retrain with new feedback data
- **Performance monitoring:** Track prediction accuracy over time

---

## Key Architectural Insights

### Problem Type Recognition
**Initial misunderstanding:** Recommendation system
**Correct understanding:** Validation/filtering system

### Data Flow Direction
**Initial assumption:** IDMC → ML (reactive)
**Correct flow:** ML validates before IDMC applies (proactive)

### Training Data Strategy
**Initial approach:** Wait for new data
**Correct approach:** Mine existing IDMC implementations + LLM augmentation

### Value Proposition
**Initial focus:** Suggest new rules
**Correct focus:** Prevent inappropriate rule applications

---

## Prompt Engineering Lessons

### 1. Domain Context is Critical
The initial prompt required deep understanding of:
- IDMC workflow and architecture
- Data quality rule systems
- Enterprise data management processes

### 2. Iterative Clarification Process
- **First response:** Based on common ML patterns (recommendation)
- **Clarification:** Revealed true requirement (validation)
- **Final solution:** Aligned with actual business need

### 3. Technical Architecture Implications
Understanding the difference between:
- **Recommendation:** "What rules should we apply?"
- **Validation:** "Should we apply this specific rule?"

Changes the entire technical approach, data requirements, and integration strategy.

---

## Complete Prompt Sequence Summary

### Stage 1: Initial Problem Description
```
"Informatica IDMC... build external machine learning component... 
give indication if particular rule is applicable to dataset or not"
```
**AI Response:** Recommendation system architecture

### Stage 2: Architectural Correction
```
"IDMC does not send anything... feed test data from existing rules... 
generate rules through LLMs... before we feed any rule, see if applicable"
```
**AI Response:** Validation system architecture

### Stage 3: Solution Elaboration
```
"can you build a small documentation on this. Also please write 
for me a complete end to end prompt to arrive here."
```
**AI Response:** Complete technical documentation + this prompt analysis

---

## Meta-Learning from this Conversation

### 1. Requirements Gathering
- Initial problem statements may be ambiguous
- Domain expertise affects interpretation
- Iterative clarification is essential

### 2. Solution Architecture
- Small wording changes can completely change system design
- "Send data for recommendations" vs "Validate before applying"
- Architecture follows from understanding the true business process

### 3. Technical Implementation
- Training data strategy changes based on problem type
- API design reflects the user interaction pattern
- Integration points depend on workflow understanding

### 4. Prompt Effectiveness
The most effective prompts:
- Include specific domain context
- Clarify the business process flow
- Specify the integration requirements
- Provide concrete use cases

**Example of improved initial prompt:**
```
"We have Informatica IDMC with existing data quality rules applied 
across multiple projects. We want to build an ML system that validates 
whether a specific rule should be applied to a new dataset BEFORE 
we actually apply it. The training data should come from existing 
rule applications in IDMC, augmented with LLM-generated synthetic 
rules. The system should integrate as a validation step in our 
rule application workflow."
```

This would have led directly to the correct solution architecture from the start.

---

## Conclusion

This conversation demonstrates how iterative refinement and clarification lead to precise technical solutions. The key lesson is that understanding the business process flow and integration requirements is more important than the specific ML techniques used.

The evolution from "rule recommendation system" to "rule validation system" completely changed:
- Data collection strategy
- Model training approach  
- API design
- Integration architecture
- Value proposition

This highlights the importance of thorough requirements gathering and iterative clarification in AI-assisted solution design.



# ML-Powered Rule Applicability System for IDMC
## Technical Documentation

### Executive Summary

This document outlines the design and implementation of an external machine learning component that validates whether data quality rules should be applied to specific datasets in Informatica IDMC. The system prevents inappropriate rule application by predicting rule applicability before execution, improving data quality assessment accuracy and reducing false positives.

### Problem Statement

In Informatica IDMC, data quality rules are often applied broadly across datasets without considering their contextual relevance. This leads to:
- False positive quality issues
- Irrelevant rule executions
- Decreased confidence in quality scores
- Manual effort in rule selection

### Solution Overview

**Core Concept:** Build an ML model that answers "Should this rule be applied to this dataset?" before rule execution.

**Key Components:**
1. Training data generation from existing IDMC projects
2. LLM-assisted synthetic rule creation
3. ML model for rule-dataset compatibility prediction
4. REST API for real-time rule validation

---

## Architecture

### High-Level System Flow

```
[IDMC Projects] → [Rule Extraction] → [Training Data Generation]
                                    ↓
[LLM Rule Generator] → [Feature Engineering] → [ML Model Training]
                                    ↓
[Dataset Profile] + [Candidate Rule] → [ML API] → [Applicability Decision]
                                    ↓
[IDMC Rule Engine] ← [Apply/Skip Rule] ← [Confidence Score]
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IDMC Environment                         │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Data Profiling  │    │     Rule Engine                 │ │
│  │ & Dataset Stats │    │  (Before applying any rule)     │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
                  │ Dataset Profile       │ Rule Validation Request
                  │                       │
┌─────────────────▼───────────────────────▼───────────────────┐
│              ML Applicability Service                       │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Feature         │  │ ML Model        │  │ Confidence  │ │
│  │ Extraction      │  │ (Random Forest/ │  │ Scoring     │ │
│  │                 │  │  XGBoost)       │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Training Data Pipeline                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │ IDMC Rule   │  │ LLM Rule    │  │ Negative Sample │ │ │
│  │  │ Extraction  │  │ Generation  │  │ Generation      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Guide

### Phase 1: Data Collection & Preparation (Weeks 1-2)

#### 1.1 Extract Existing Rules from IDMC

**Objective:** Build initial training dataset from existing IDMC implementations

**Implementation:**
```python
class IDMCRuleExtractor:
    def __init__(self, idmc_connection):
        self.connection = idmc_connection
    
    def extract_applied_rules(self):
        """Extract rules that were actually applied to datasets"""
        training_records = []
        
        projects = self.connection.get_all_projects()
        for project in projects:
            profiles = project.get_data_profiles()
            
            for profile in profiles:
                dataset_stats = self.extract_dataset_profile(profile)
                applied_rules = profile.get_applied_rules()
                
                for rule in applied_rules:
                    training_records.append({
                        'dataset_profile': dataset_stats,
                        'rule': self.extract_rule_definition(rule),
                        'is_applicable': True,  # Since it was applied
                        'project_context': project.metadata,
                        'success_rate': rule.execution_stats.success_rate
                    })
        
        return training_records
    
    def extract_dataset_profile(self, profile):
        """Extract relevant dataset characteristics"""
        return {
            'dataset_id': profile.dataset_id,
            'row_count': profile.statistics.row_count,
            'column_count': len(profile.columns),
            'columns': [self.extract_column_profile(col) for col in profile.columns],
            'domain': self.infer_domain(profile),
            'data_freshness': profile.last_updated
        }
    
    def extract_column_profile(self, column):
        """Extract column-level statistics"""
        return {
            'name': column.name,
            'data_type': column.data_type,
            'null_percentage': column.null_count / column.total_count * 100,
            'unique_percentage': column.distinct_count / column.total_count * 100,
            'sample_values': column.sample_values[:10],
            'pattern_analysis': self.analyze_patterns(column.sample_values),
            'statistical_profile': {
                'min': column.min_value,
                'max': column.max_value,
                'mean': column.mean_value,
                'std_dev': column.std_deviation
            }
        }
```

#### 1.2 Generate Negative Training Examples

**Objective:** Create examples where rules should NOT be applied

**Strategy:**
- Cross-apply rules to datasets where they weren't originally used
- Generate obviously incompatible rule-dataset pairs
- Manual validation of negative examples

```python
def generate_negative_examples(positive_examples):
    negative_examples = []
    
    all_rules = [example['rule'] for example in positive_examples]
    all_datasets = [example['dataset_profile'] for example in positive_examples]
    
    for rule in all_rules:
        for dataset in all_datasets:
            # Check if this combination exists in positive examples
            if not combination_exists(rule, dataset, positive_examples):
                compatibility_score = calculate_compatibility(rule, dataset)
                
                if compatibility_score < 0.3:  # Low compatibility threshold
                    negative_examples.append({
                        'dataset_profile': dataset,
                        'rule': rule,
                        'is_applicable': False,
                        'incompatibility_reason': analyze_incompatibility(rule, dataset)
                    })
    
    return negative_examples
```

### Phase 2: LLM-Powered Rule Generation (Weeks 2-3)

#### 2.1 LLM Integration for Synthetic Rule Creation

**Objective:** Expand training data with LLM-generated rules

**Implementation:**
```python
class LLMRuleGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_rules_for_dataset(self, dataset_profile):
        """Generate potential rules for a given dataset"""
        rules = []
        
        for column in dataset_profile['columns']:
            column_rules = self.generate_column_rules(column, dataset_profile)
            rules.extend(column_rules)
        
        return rules
    
    def generate_column_rules(self, column_profile, dataset_context):
        """Generate rules for a specific column"""
        
        prompt = self.build_rule_generation_prompt(column_profile, dataset_context)
        
        llm_response = self.llm_client.generate_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        return self.parse_generated_rules(llm_response, column_profile)
    
    def build_rule_generation_prompt(self, column_profile, dataset_context):
        """Create structured prompt for rule generation"""
        
        prompt = f"""
        You are a data quality expert. Generate appropriate data quality rules for the following column:
        
        DATASET CONTEXT:
        - Dataset Domain: {dataset_context.get('domain', 'Unknown')}
        - Total Rows: {dataset_context.get('row_count', 'Unknown')}
        - Business Context: {dataset_context.get('business_context', 'General')}
        
        COLUMN DETAILS:
        - Column Name: {column_profile['name']}
        - Data Type: {column_profile['data_type']}
        - Null Percentage: {column_profile['null_percentage']}%
        - Unique Values: {column_profile['unique_percentage']}%
        - Sample Values: {column_profile['sample_values']}
        - Detected Patterns: {column_profile['pattern_analysis']}
        
        Generate 3-5 data quality rules that would be appropriate for this column.
        For each rule, provide:
        1. Rule Type (COMPLETENESS, VALIDITY, CONSISTENCY, ACCURACY, UNIQUENESS, INTEGRITY)
        2. Rule Definition (SQL-like or regex format)
        3. Applicability Confidence (0.0 to 1.0)
        4. Business Justification
        
        Format your response as JSON:
        [
          {{
            "rule_type": "COMPLETENESS",
            "rule_definition": "NULL_CHECK(column_name) < 5%",
            "confidence": 0.9,
            "justification": "Column appears to be required based on low null rate",
            "expected_outcome": "Flag records with missing values"
          }}
        ]
        
        Consider the column's characteristics carefully. Don't generate rules that obviously don't fit.
        """
        
        return prompt
    
    def parse_generated_rules(self, llm_response, column_profile):
        """Parse and validate LLM-generated rules"""
        try:
            rules_json = json.loads(llm_response)
            validated_rules = []
            
            for rule in rules_json:
                # Validate rule structure
                if self.validate_rule_structure(rule):
                    # Add metadata
                    rule['generated_by'] = 'llm'
                    rule['target_column'] = column_profile['name']
                    rule['target_data_type'] = column_profile['data_type']
                    rule['generation_timestamp'] = datetime.now().isoformat()
                    
                    validated_rules.append(rule)
            
            return validated_rules
            
        except json.JSONDecodeError:
            # Fallback parsing or retry logic
            return self.fallback_rule_parsing(llm_response, column_profile)
```

#### 2.2 Rule Quality Validation

**Objective:** Ensure generated rules are meaningful and applicable

```python
class RuleValidator:
    def validate_generated_rule(self, rule, column_profile):
        """Validate if a generated rule makes sense for the column"""
        
        validation_checks = {
            'data_type_compatibility': self.check_data_type_match(rule, column_profile),
            'logical_consistency': self.check_rule_logic(rule),
            'practical_applicability': self.check_practical_sense(rule, column_profile),
            'syntax_validity': self.check_rule_syntax(rule['rule_definition'])
        }
        
        # Rule is valid if all checks pass
        is_valid = all(validation_checks.values())
        
        return {
            'is_valid': is_valid,
            'validation_details': validation_checks,
            'confidence_adjustment': self.calculate_confidence_adjustment(validation_checks)
        }
```

### Phase 3: Feature Engineering (Week 3)

#### 3.1 Feature Extraction Pipeline

**Objective:** Convert rule-dataset pairs into ML features

```python
class FeatureEngineer:
    def __init__(self):
        self.feature_extractors = [
            DatasetFeatureExtractor(),
            RuleFeatureExtractor(),
            CompatibilityFeatureExtractor(),
            ContextualFeatureExtractor()
        ]
    
    def extract_features(self, rule, dataset_profile):
        """Extract all features for a rule-dataset pair"""
        
        features = {}
        
        for extractor in self.feature_extractors:
            extractor_features = extractor.extract(rule, dataset_profile)
            features.update(extractor_features)
        
        # Add interaction features
        interaction_features = self.create_interaction_features(features)
        features.update(interaction_features)
        
        return features

class DatasetFeatureExtractor:
    def extract(self, rule, dataset_profile):
        """Extract dataset-level features"""
        
        return {
            # Size features
            'dataset_row_count': dataset_profile['row_count'],
            'dataset_column_count': dataset_profile['column_count'],
            'dataset_size_category': self.categorize_size(dataset_profile['row_count']),
            
            # Quality features
            'avg_null_percentage': self.calculate_avg_null_rate(dataset_profile),
            'avg_unique_percentage': self.calculate_avg_unique_rate(dataset_profile),
            'data_quality_score': self.calculate_overall_quality(dataset_profile),
            
            # Diversity features
            'data_type_diversity': len(set(col['data_type'] for col in dataset_profile['columns'])),
            'domain_indicators': self.extract_domain_features(dataset_profile),
            
            # Complexity features
            'schema_complexity': self.calculate_schema_complexity(dataset_profile),
            'relationship_indicators': self.detect_relationships(dataset_profile)
        }

class RuleFeatureExtractor:
    def extract(self, rule, dataset_profile):
        """Extract rule-specific features"""
        
        return {
            # Rule characteristics
            'rule_type': rule['rule_type'],
            'rule_complexity': self.calculate_rule_complexity(rule['rule_definition']),
            'rule_specificity': self.calculate_specificity(rule),
            
            # Target features
            'targets_specific_column': 'target_column' in rule,
            'targets_data_type': rule.get('target_data_type', 'any'),
            'requires_reference_data': self.check_reference_requirement(rule),
            
            # Pattern features
            'uses_regex': 'REGEX' in rule['rule_definition'],
            'uses_sql_functions': self.detect_sql_functions(rule['rule_definition']),
            'threshold_based': self.has_thresholds(rule['rule_definition'])
        }

class CompatibilityFeatureExtractor:
    def extract(self, rule, dataset_profile):
        """Extract compatibility-specific features"""
        
        target_column = self.find_target_column(rule, dataset_profile)
        
        compatibility_features = {
            'data_type_match': self.check_data_type_compatibility(rule, target_column),
            'pattern_alignment': self.check_pattern_alignment(rule, target_column),
            'statistical_compatibility': self.check_statistical_fit(rule, target_column),
            'business_logic_fit': self.assess_business_logic_fit(rule, dataset_profile)
        }
        
        return compatibility_features
```

#### 3.2 Feature Importance Analysis

**Objective:** Understand which features drive rule applicability decisions

```python
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Create interpretable feature groups
    feature_groups = {
        'dataset_characteristics': ['dataset_row_count', 'dataset_column_count', 'data_quality_score'],
        'rule_properties': ['rule_complexity', 'rule_specificity', 'rule_type'],
        'compatibility_signals': ['data_type_match', 'pattern_alignment', 'statistical_compatibility']
    }
    
    return feature_importance_df, feature_groups
```

### Phase 4: Model Training & Validation (Week 4)

#### 4.1 Model Training Pipeline

```python
class RuleApplicabilityModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'ensemble': VotingClassifier(estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ], voting='soft')
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models and select the best one"""
        
        results = {}
        
        for model_name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Validate
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            results[model_name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'auc_roc': roc_auc_score(y_val, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_val, y_pred)
            }
        
        # Select best model based on F1 score (balanced for precision/recall)
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        self.best_model = self.models[best_model_name]
        
        return results, best_model_name
```

#### 4.2 Model Validation & Testing

```python
class ModelValidator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def comprehensive_validation(self):
        """Run comprehensive validation tests"""
        
        validation_results = {
            'performance_metrics': self.calculate_performance_metrics(),
            'robustness_tests': self.test_robustness(),
            'bias_analysis': self.analyze_bias(),
            'interpretability': self.explain_predictions()
        }
        
        return validation_results
    
    def test_robustness(self):
        """Test model robustness with edge cases"""
        
        edge_cases = [
            self.create_very_small_dataset_cases(),
            self.create_very_large_dataset_cases(),
            self.create_unusual_data_type_cases(),
            self.create_extreme_quality_cases()
        ]
        
        robustness_scores = []
        
        for case_group in edge_cases:
            predictions = self.model.predict_proba(case_group)
            # Check if predictions are reasonable (not too extreme)
            robustness_scores.append(self.evaluate_prediction_reasonableness(predictions))
        
        return {
            'average_robustness': np.mean(robustness_scores),
            'edge_case_performance': robustness_scores
        }
```

### Phase 5: API Development (Week 5)

#### 5.1 REST API Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Rule Applicability API", version="1.0.0")

# Load trained model
model = joblib.load('models/rule_applicability_model.pkl')
feature_engineer = joblib.load('models/feature_engineer.pkl')

class RuleValidationRequest(BaseModel):
    rule: dict
    dataset_profile: dict
    context: dict = {}

class RuleValidationResponse(BaseModel):
    is_applicable: bool
    confidence: float
    reasoning: str
    recommendations: list = []

@app.post("/validate-rule", response_model=RuleValidationResponse)
async def validate_rule_applicability(request: RuleValidationRequest):
    """
    Validate if a rule should be applied to a dataset
    """
    try:
        # Extract features
        features = feature_engineer.extract_features(
            rule=request.rule,
            dataset_profile=request.dataset_profile
        )
        
        # Convert to model input format
        feature_vector = feature_engineer.vectorize(features)
        
        # Get prediction
        prediction_proba = model.predict_proba([feature_vector])[0]
        confidence = float(prediction_proba[1])  # Probability of being applicable
        is_applicable = confidence > 0.6  # Configurable threshold
        
        # Generate reasoning
        reasoning = generate_reasoning(features, model, confidence)
        
        # Generate recommendations if not applicable
        recommendations = []
        if not is_applicable:
            recommendations = generate_alternative_recommendations(
                request.rule, 
                request.dataset_profile
            )
        
        return RuleValidationResponse(
            is_applicable=is_applicable,
            confidence=confidence,
            reasoning=reasoning,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/batch-validate")
async def batch_validate_rules(rules: list, dataset_profile: dict):
    """
    Validate multiple rules against a single dataset
    """
    results = []
    
    for rule in rules:
        request = RuleValidationRequest(rule=rule, dataset_profile=dataset_profile)
        result = await validate_rule_applicability(request)
        results.append({
            'rule_id': rule.get('id', 'unknown'),
            'validation_result': result
        })
    
    return {
        'dataset_id': dataset_profile.get('dataset_id', 'unknown'),
        'total_rules_evaluated': len(rules),
        'applicable_rules': sum(1 for r in results if r['validation_result'].is_applicable),
        'results': results
    }

def generate_reasoning(features, model, confidence):
    """Generate human-readable reasoning for the decision"""
    
    # Get feature importance for this prediction
    if hasattr(model, 'feature_importances_'):
        important_features = get_top_features(features, model.feature_importances_)
        
        reasoning_parts = []
        
        for feature, importance in important_features[:3]:  # Top 3 features
            feature_value = features.get(feature, 'unknown')
            reasoning_parts.append(f"{feature}: {feature_value} (importance: {importance:.2f})")
        
        reasoning = f"Decision based on: {'; '.join(reasoning_parts)}. Confidence: {confidence:.2f}"
        
    else:
        reasoning = f"Model confidence: {confidence:.2f}"
    
    return reasoning
```

#### 5.2 Integration Documentation

```python
# Example IDMC Integration Code
class IDMCRuleValidator:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
    
    def validate_before_applying_rule(self, rule_definition, dataset_profile):
        """
        Check with ML service before applying any rule
        """
        
        # Prepare request
        validation_request = {
            'rule': {
                'id': rule_definition.rule_id,
                'type': rule_definition.rule_type,
                'definition': rule_definition.sql_expression,
                'target_column': rule_definition.target_column,
                'parameters': rule_definition.parameters
            },
            'dataset_profile': {
                'dataset_id': dataset_profile.dataset_id,
                'row_count': dataset_profile.row_count,
                'columns': [col.to_dict() for col in dataset_profile.columns],
                'domain': dataset_profile.inferred_domain
            }
        }
        
        # Call ML API
        response = requests.post(
            f"{self.api_endpoint}/validate-rule",
            json=validation_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result['is_applicable'] and result['confidence'] > 0.8:
                return "APPLY", result['reasoning']
            elif result['is_applicable'] and result['confidence'] > 0.6:
                return "REVIEW", f"Medium confidence: {result['reasoning']}"
            else:
                return "SKIP", f"Rule not applicable: {result['reasoning']}"
        
        else:
            # Fallback to applying rule if API fails
            return "APPLY", "ML validation unavailable - applying rule"

# Usage in IDMC workflow
rule_validator = IDMCRuleValidator("http://ml-service:8000")

def execute_data_quality_assessment(dataset, rule_catalog):
    """Modified DQ assessment with ML validation"""
    
    dataset_profile = profile_dataset(dataset)
    applicable_rules = []
    
    for rule in rule_catalog:
        decision, reasoning = rule_validator.validate_before_applying_rule(rule, dataset_profile)
        
        if decision == "APPLY":
            applicable_rules.append(rule)
            log_info(f"Applying rule {rule.rule_id}: {reasoning}")
        
        elif decision == "REVIEW":
            # Could trigger manual review workflow
            log_warning(f"Rule {rule.rule_id} needs review: {reasoning}")
            if auto_approval_enabled():
                applicable_rules.append(rule)
        
        else:  # SKIP
            log_info(f"Skipping rule {rule.rule_id}: {reasoning}")
    
    # Apply only validated rules
    quality_results = apply_rules_to_dataset(dataset, applicable_rules)
    
    return quality_results
```

---

## Monitoring & Maintenance

### Performance Monitoring

```python
class ModelMonitor:
    def __init__(self, model_api):
        self.api = model_api
        self.metrics = MetricsCollector()
    
    def monitor_predictions(self):
        """Monitor model performance in production"""
        
        # Collect prediction statistics
        recent_predictions = self.api.get_recent_predictions(days=7)
        
        metrics = {
            'total_predictions': len(recent_predictions),
            'average_confidence': np.mean([p['confidence'] for p in recent_predictions]),
            'applicability_rate': sum(1 for p in recent_predictions if p['is_applicable']) / len(recent_predictions),
            'low_confidence_rate': sum(1 for p in recent_predictions if p['confidence'] < 0.6) / len(recent_predictions)
        }
        
        # Check for drift indicators
        drift_indicators = self.detect_drift(recent_predictions)
        
        return {
            'performance_metrics': metrics,
            'drift_indicators': drift_indicators,
            'alert_conditions': self.check_alert_conditions(metrics, drift_indicators)
        }
```

### Continuous Learning Pipeline

```python
class ContinuousLearning:
    def __init__(self, model, training_pipeline):
        self.model = model
        self.training_pipeline = training_pipeline
        self.feedback_buffer = []
    
    def collect_feedback(self, prediction_id, actual_outcome, user_feedback):
        """Collect feedback on predictions"""
        
        self.feedback_buffer.append({
            'prediction_id': prediction_id,
            'predicted_applicable': prediction_id.is_applicable,
            'actual_applicable': actual_outcome,
            'user_feedback': user_feedback,
            'timestamp': datetime.now()
        })
        
        # Trigger retraining if enough feedback collected
        if len(self.feedback_buffer) >= 1000:
            self.trigger_retraining()
    
    def trigger_retraining(self):
        """Retrain model with new feedback data"""
        
        # Combine feedback with original training data
        new_training_data = self.prepare_retraining_data(self.feedback_buffer)
        
        # Retrain model
        updated_model = self.training_pipeline.retrain(new_training_data)
        
        # Validate updated model
        if self.validate_updated_model(updated_model):
            self.deploy_updated_model(updated_model)
            self.feedback_buffer = []  # Clear buffer
```

---

## Success Metrics & KPIs

### Technical Metrics
- **Model Accuracy**: >85% on test set
- **Precision**: >80% (minimize false positives)
- **Recall**: >75% (don't miss applicable rules)
- **API Response Time**: <500ms for single rule validation
- **System Uptime**: >99.5%

### Business Metrics
- **Rule Application Efficiency**: 30% reduction in inappropriate rule applications
- **Data Quality Score Accuracy**: Improved consistency in quality assessments
- **Manual Review Reduction**: 50% decrease in manual rule selection effort
- **User Adoption**: >70% of IDMC users utilizing the validation API

### Operational Metrics
- **False Positive Rate**: <15% for rule applicability predictions
- **Model Drift Detection**: Weekly monitoring with automated alerts
- **Feedback Collection Rate**: >60% of predictions receive user feedback within 30 days

---

## Risk Assessment & Mitigation

### Technical Risks
1. **Model Drift**: Continuous monitoring and retraining pipeline
2. **API Downtime**: Graceful degradation with fallback to rule application
3. **Feature Engineering Complexity**: Comprehensive testing and validation
4. **Training Data Quality**: Data validation and cleaning processes

### Business Risks
1. **Over-reliance on ML**: Maintain manual override capabilities
2. **User Resistance**: Comprehensive training and change management
3. **Integration Complexity**: Phased rollout and extensive testing

### Mitigation Strategies
- Robust testing framework with edge cases
- Feature flags for gradual rollout
- Comprehensive logging and monitoring
- Regular model performance reviews
- User feedback integration

---

## Future Enhancements

### Phase 2 Enhancements
- **Advanced Rule Generation**: More sophisticated LLM integration
- **Context-Aware Predictions**: Consider business domain and use case
- **Multi-Model Ensemble**: Combine different ML approaches
- **Real-time Learning**: Immediate feedback integration

### Phase 3 Vision
- **Automated Rule Optimization**: ML-suggested rule improvements
- **Anomaly Detection Integration**: Identify unusual data patterns
- **Cross-Dataset Learning**: Learn from similar datasets across projects
- **Natural Language Rule Interface**: Generate rules from business descriptions

---

## Conclusion

This ML-powered rule applicability system provides a sophisticated approach to improving data quality assessment in IDMC. By validating rule relevance before application, the system reduces noise in quality reporting and improves overall assessment accuracy.

The phased implementation approach ensures manageable development while delivering incremental value. The system's foundation on existing IDMC data, enhanced by LLM-generated synthetic examples, provides a robust training base for accurate rule applicability predictions.

Key success factors include maintaining high model accuracy, seamless IDMC integration, and continuous learning from user feedback. The system's modular architecture supports future enhancements and scaling as data quality requirements evolve.

---

## Appendix A: Sample Data Structures

### Training Record Format
```json
{
  "record_id": "tr_001",
  "dataset_profile": {
    "dataset_id": "customer_master_v2",
    "domain": "customer_data",
    "row_count": 1500000,
    "column_count": 25,
    "columns": [
      {
        "name": "email",
        "data_type": "string",
        "null_percentage": 2.1,
        "unique_percentage": 98.7,
        "sample_values": ["john@example.com", "jane@company.org"],
        "pattern_analysis": {
          "email_format": 0.96,
          "domain_diversity": 234,
          "common_domains": ["gmail.com", "yahoo.com", "company.com"]
        },
        "statistical_profile": {
          "avg_length": 24.5,
          "min_length": 8,
          "max_length": 64
        }
      }
    ],
    "data_quality_indicators": {
      "overall_completeness": 0.94,
      "schema_consistency": 0.99,
      "referential_integrity": 0.87
    }
  },
  "rule": {
    "rule_id": "email_format_validation_001",
    "rule_type": "VALIDITY",
    "rule_definition": "REGEX_MATCH(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})",
    "target_column": "email",
    "target_data_type": "string",
    "complexity_score": 0.6,
    "business_domain": "communication",
    "expected_impact": "format_validation"
  },
  "is_applicable": true,
  "confidence": 0.95,
  "metadata": {
    "source": "idmc_extraction",
    "project_id": "proj_customer_quality",
    "extraction_date": "2024-08-01",
    "rule_success_rate": 0.94
  }
}
```

### API Request/Response Examples
```json
// Request
{
  "rule": {
    "rule_id": "completeness_check_001",
    "rule_type": "COMPLETENESS",
    "rule_definition": "NULL_CHECK(customer_id) = 0",
    "target_column": "customer_id",
    "parameters": {
      "threshold": 0,
      "severity": "high"
    }
  },
  "dataset_profile": {
    "dataset_id": "new_customer_data",
    "row_count": 50000,
    "columns": [
      {
        "name": "customer_id",
        "data_type": "integer",
        "null_percentage": 0.0,
        "unique_percentage": 100.0,
        "pattern_analysis": {
          "sequential": true,
          "numeric_range": [1, 50000]
        }
      }
    ]
  }
}

// Response
{
  "is_applicable": true,
  "confidence": 0.92,
  "reasoning": "High confidence due to: data_type_match=perfect, null_percentage=0.0 (supports completeness rule), unique_percentage=100.0 (indicates key column)",
  "recommendations": [],
  "metadata": {
    "prediction_id": "pred_20240814_001",
    "model_version": "v1.2.0",
    "response_time_ms": 145
  }
}
```

---

## Appendix B: Deployment Configuration

### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Set environment variables
ENV MODEL_PATH=/app/models/rule_applicability_model.pkl
ENV FEATURE_ENGINEER_PATH=/app/models/feature_engineer.pkl
ENV API_PORT=8000
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rule-applicability-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rule-applicability-api
  template:
    metadata:
      labels:
        app: rule-applicability-api
    spec:
      containers:
      - name: api
        image: rule-applicability-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/rule_applicability_model.pkl"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rule-applicability-service
spec:
  selector:
    app: rule-applicability-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```
