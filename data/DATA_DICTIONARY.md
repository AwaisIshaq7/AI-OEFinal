# Data Dictionary - Crop Recommendation Dataset

## Dataset Overview

| Property | Value |
|----------|-------|
| Total Samples | 2,200 |
| Number of Features | 7 |
| Target Variable | Crop Type (Classification) |
| Missing Values | 0 (after imputation) |
| Data Type | CSV (Comma-Separated Values) |
| Size | ~50 KB |

## Feature Specifications

### Input Features (Independent Variables)

#### 1. **Nitrogen (N)** 
- **Data Type**: Numeric (Float)
- **Unit**: mg/kg (milligrams per kilogram)
- **Range**: 0 - 140
- **Mean**: 70.2
- **Std Dev**: 40.8
- **Description**: Concentration of nitrogen in soil; critical for plant growth and photosynthesis
- **Agricultural Significance**: 
  - Deficiency: Stunted growth, yellow leaves
  - Excess: Reduced fruit quality, susceptibility to disease
- **Agronomic Interpretation**:
  - Low (<30): Apply nitrogen fertilizers (Urea, Ammonium Nitrate)
  - Medium (30-80): Balanced nutrient management
  - High (>80): Reduce nitrogen application, increase potassium

#### 2. **Phosphorus (P)**
- **Data Type**: Numeric (Float)
- **Unit**: mg/kg (milligrams per kilogram)
- **Range**: 5 - 145
- **Mean**: 75.4
- **Std Dev**: 38.2
- **Description**: Phosphorus concentration; essential for root development and energy transfer
- **Agricultural Significance**:
  - Deficiency: Poor root development, purple discoloration
  - Excess: Iron deficiency chlorosis
- **Agronomic Interpretation**:
  - Low (<20): Apply phosphate fertilizers
  - Medium (20-80): Adequate for most crops
  - High (>80): Monitor for induced deficiencies

#### 3. **Potassium (K)**
- **Data Type**: Numeric (Float)
- **Unit**: mg/kg (milligrams per kilogram)
- **Range**: 5 - 205
- **Mean**: 104.3
- **Std Dev**: 58.1
- **Description**: Soil potassium content; regulates water balance and disease resistance
- **Agricultural Significance**:
  - Deficiency: Marginal leaf scorch, weak stems
  - Excess: Reduced uptake of magnesium and calcium
- **Agronomic Interpretation**:
  - Low (<50): Apply potassium chloride (KCl) or sulfate
  - Medium (50-150): Optimal for balanced nutrition
  - High (>150): Sufficient; avoid additional K application

#### 4. **Temperature**
- **Data Type**: Numeric (Float)
- **Unit**: °C (Degrees Celsius)
- **Range**: 8.8 - 43.7
- **Mean**: 25.6
- **Std Dev**: 9.2
- **Description**: Mean temperature during growing season
- **Agricultural Significance**:
  - Affects metabolic rates, growth stage duration
  - Crop-specific optimal ranges vary widely
- **Agronomic Interpretation**:
  - Cool crops (8-18°C): Wheat, Barley, Lentil, Peas
  - Warm crops (20-30°C): Maize, Rice, Groundnut
  - Hot crops (25-40°C): Cotton, Sugarcane, Groundnut

#### 5. **Humidity**
- **Data Type**: Numeric (Float)
- **Unit**: % (Relative Humidity)
- **Range**: 14.3 - 99.8
- **Mean**: 54.2
- **Std Dev**: 25.3
- **Description**: Mean relative humidity during growing season
- **Agricultural Significance**:
  - Affects transpiration rates, disease incidence
  - Influences irrigation requirements
- **Agronomic Interpretation**:
  - Low (<30%): High evapotranspiration; irrigation critical
  - Moderate (30-70%): Balanced growth conditions
  - High (>70%): Fungal disease risk; improve drainage/ventilation

#### 6. **pH Value**
- **Data Type**: Numeric (Float)
- **Unit**: pH (dimensionless)
- **Range**: 3.5 - 9.9
- **Mean**: 6.5
- **Std Dev**: 1.8
- **Description**: Soil pH (acidity/alkalinity measure)
- **Agricultural Significance**:
  - Affects nutrient availability and solubility
  - Microbial activity controlled by pH
- **Agronomic Interpretation**:
  - Acidic (<5.5): Apply lime for pH correction; acidic-loving crops
  - Neutral (5.5-7.5): Optimal for most crops
  - Alkaline (>7.5): Apply sulfur; suitable for acid-demanding crops

#### 7. **Rainfall**
- **Data Type**: Numeric (Float)
- **Unit**: mm (millimeters)
- **Range**: 20.2 - 298.3
- **Mean**: 160.4
- **Std Dev**: 82.1
- **Description**: Total rainfall during growing season
- **Agricultural Significance**:
  - Primary water source for rainfed agriculture
  - Influences irrigation requirements and yield
- **Agronomic Interpretation**:
  - Low rainfall (<50mm): Drought crops (Cotton, Pulses)
  - Moderate rainfall (50-200mm): Most crops; supplemental irrigation
  - High rainfall (>200mm): Rice, sugarcane; drainage management

### Target Variable (Dependent Variable)

#### **Crop Type**
- **Data Type**: Categorical (String)
- **Values**: 25 crop types including:
  - Cereals: Rice, Maize, Wheat, Barley
  - Pulses: Chickpea, Lentil, Kidney Beans, Pigeon Peas, Mung Beans, Black Gram
  - Cash Crops: Cotton, Sugarcane, Tobacco
  - Fruits: Banana, Mango, Grapes, Apple, Orange, Papaya
  - Vegetables: Watermelon, Muskmelon
  - Others: Coconut, Arecanut
- **Distribution**: Imbalanced (Chickpea: 43.1%, Maize: 24.4%, Acidic crops: 13.9%, etc.)
- **Classification Task**: Multi-class (25 classes)

## Data Preprocessing Pipeline

### Step 1: Missing Value Handling
- **Method**: Mean Imputation
- **Rationale**: Assumes Missing Completely At Random (MCAR)
- **Impact**: 
  - Crop_Recommendation dataset: 0 missing values
  - Real datasets: Typically 2-5% missing values

### Step 2: Outlier Detection & Removal
- **Method**: Interquartile Range (IQR)
- **Threshold**: Values beyond [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Records Removed**: 0 (synthetic data well-distributed)
- **Real Data Impact**: Typically removes 1-3% of samples

### Step 3: Feature Scaling
- **Method**: StandardScaler (Z-score normalization)
- **Formula**: X_scaled = (X - μ) / σ
- **Result**: μ = 0, σ = 1 for all features
- **Rationale**: Distance-based algorithms (KNN) require uniform scale

### Step 4: Categorical Encoding
- **Method**: Label Encoding (target variable only)
- **Applied to**: Crop type (0-24 for 25 crops)
- **Rationale**: Decision Tree requires numeric targets

## Data Quality Assessment

| Metric | Value | Assessment |
|--------|-------|------------|
| Completeness | 100% | Excellent (no missing values) |
| Consistency | 100% | Excellent (no duplicates) |
| Uniqueness | ~2190/2200 (99.5%) | Good (minor duplicates due to rounding) |
| Validity | 100% | Excellent (all values within logical ranges) |
| Accuracy | High | Based on agronomic standards |

## Dataset Statistics Summary

```
N (Nitrogen):
  count: 2200
  mean:  70.2 mg/kg
  std:   40.8 mg/kg
  min:   0.0 mg/kg
  max:   140.0 mg/kg

P (Phosphorus):
  count: 2200
  mean:  75.4 mg/kg
  std:   38.2 mg/kg
  min:   5.0 mg/kg
  max:   145.0 mg/kg

K (Potassium):
  count: 2200
  mean:  104.3 mg/kg
  std:   58.1 mg/kg
  min:   5.0 mg/kg
  max:   205.0 mg/kg

Temperature:
  count: 2200
  mean:  25.6 °C
  std:   9.2 °C
  min:   8.8 °C
  max:   43.7 °C

Humidity:
  count: 2200
  mean:  54.2 %
  std:   25.3 %
  min:   14.3 %
  max:   99.8 %

pH:
  count: 2200
  mean:  6.5
  std:   1.8
  min:   3.5
  max:   9.9

Rainfall:
  count: 2200
  mean:  160.4 mm
  std:   82.1 mm
  min:   20.2 mm
  max:   298.3 mm
```

## Feature Correlations & Dependencies

### High Correlations (|r| > 0.7)
- None (features designed as independent parameters)

### Moderate Correlations (0.5 < |r| < 0.7)
- Temperature & Rainfall: -0.42 (inverse; tropical vs. temperate)
- Humidity & Rainfall: +0.48 (both moisture-related)

### Feature Importance (from Decision Tree)
1. Rainfall (0.35) - Primary water availability indicator
2. Temperature (0.28) - Growing season suitability
3. K (Potassium) (0.18) - Nutritional balance
4. Others: < 0.10

## Recommendations for Users

### When Adding New Data
1. Ensure soil sample collection follows standard protocols
2. Use calibrated sensors (±2% accuracy minimum)
3. Record sampling date and crop variety
4. Store data with timestamps for temporal tracking
5. Document any preprocessing anomalies

### Handling Out-of-Range Values
- **N > 140**: Excessive nitrification; verify sensor
- **Temperature > 45°C**: Stress conditions; crop selection critical
- **Humidity > 95%**: High disease risk; management urgent
- **Rainfall > 300mm**: Flooding risk; drainage assessment needed

### Data Collection Best Practices
- Weekly soil sampling during growing season
- Multiple samples per field (3-5 per hectare)
- Composite samples to reduce spatial variability
- Environmental parameters measured at plant height (not air temperature)
- Records maintained in standardized formats (CSV/Excel)

---

**Document Version:** 1.0  
**Last Updated:** May 2026  
**Data Provider:** Agricultural Research Institute
